from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# with this, to use diffusers the library, we sadly *really* need to import torch for now.
# wip, trying to get this out
from torch import from_numpy
from tinygrad import Tensor, dtypes, TinyJit
import tinygrad

import einops

from config import DiffusionConfig
from models import DiffusionRgbEncoder, DiffusionConditionalUnet1d

import numpy as np


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")

class DiffusionModel():
    def __init__(self, config: DiffusionConfig):
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = config.input_shapes["observation.state"][0]
        num_images = len([k for k in config.input_shapes if k.startswith("observation.image")])
        self._use_images = False
        self._use_env_state = False
        if num_images > 0:
            self._use_images = True
            self.rgb_encoder = DiffusionRgbEncoder(config)
            global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if "observation.environment_state" in config.input_shapes:
            self._use_env_state = True
            global_cond_dim += config.input_shapes["observation.environment_state"][0]

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None
    ) -> Tensor:
        # Sample prior.
        sample = Tensor.randn(
            # shape
            batch_size, self.config.horizon, self.config.output_shapes["action"][0],
            dtype=dtypes.float
        )
        print(f'sample: {sample}')
        print(f'sample shape: {sample.shape}')
        print(f'self.num_inference_steps: {self.num_inference_steps}')

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            # Kinda tricky: t, because it's a noise scheduler, is using DDPM. This is supplied
            # by `diffusers`, which runs in Torch. This is from Hugging Face.
            #
            # This repo's moreso to prove that Tinygrad can do diffusion on-chip, so
            # I don't mind if some of these are bridged to third party libraries
            # to prove a point.
            sample = Tensor(sample.numpy(), dtype=dtypes.float)
            print(f'sample.shape[:1]: {sample.shape[:1][0]}')
            print(f'sample.t.numpy(): {t.numpy()}')
            print(f'sample: {sample}')
            model_output = self.unet(
                sample,
                Tensor.full(shape=sample.shape[:1], fill_value=t.numpy(), dtype=dtypes.long),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(
                from_numpy(model_output.numpy()), t, from_numpy(sample.numpy())
            ).prev_sample

        sample = Tensor(sample.numpy(), dtype=dtypes.float)
        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        global_cond_feats = [batch["observation.state"]]
        # Extract image feature (first combine batch, sequence, and camera index dims).
        if self._use_images:
            b, s, n, *rest = batch["observation.images"].shape
            img_features = batch["observation.images"].reshape((b * s * n, *rest))
            img_features.requires_grad = False
            img_features = self.rgb_encoder(
                img_features
            )
            # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
            # feature dim (effectively concatenating the camera features).
            total_first_dim, *rest = img_features.shape
            n = total_first_dim // (batch_size * n_obs_steps)
            img_features = img_features.reshape((batch_size, n_obs_steps, n, *rest))
            rest_product = 1
            for dim in rest:
                rest_product *= dim
            final_shape = (batch_size, n_obs_steps, n * rest_product)
            img_features = img_features.reshape(final_shape)
            global_cond_feats.append(img_features)

        if self._use_env_state:
            global_cond_feats.append(batch["observation.environment_state"].cast(dtype=dtypes.float))

        # Concatenate features then flatten to (B, global_cond_dim).
        return Tensor.cat(*(global_cond_feats), dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss_pre(self, batch: dict[str, Tensor]) -> (Tensor, Tensor, Tensor, Tensor):
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        eps = Tensor.randn(trajectory.shape, dtype=dtypes.float).realize()
        # Sample a random noising timestep for each item in the batch.
        print(f'trajectory.shape[0]: {trajectory.shape[0]}')
        timesteps = Tensor.randint(
            (trajectory.shape[0],),
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            dtype=dtypes.long
        )
        print(f'timesteps: {timesteps.numpy()}')
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        # gotta convert to torch here. noise_scheduler needs it
        noisy_trajectory = self.noise_scheduler.add_noise(
            from_numpy(trajectory.realize().numpy()), from_numpy(eps.realize().numpy()), from_numpy(timesteps.realize().numpy())
        ).numpy()

        print(f'noisy_trajectory: {noisy_trajectory.shape}')

        # convert back to tinygrad
        noisy_trajectory = Tensor(noisy_trajectory, dtype=dtypes.float)
        
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")


        
        return (noisy_trajectory, timesteps, global_cond, target)

    def compute_loss(self, noisy_trajectory:Tensor, timesteps:Tensor, global_cond:Tensor, target:Tensor) -> Tensor:
        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        #pred_cpu = pred.numpy()
        #blue_channel_zero = np.zeros(pred_cpu.shape[:2]).reshape(pred_cpu.shape[0], pred_cpu.shape[1], 1)
        #pred_cpu = np.concatenate((pred_cpu, blue_channel_zero), axis=-1)
        #plt.imshow(pred_cpu, interpolation='nearest')
        #plt.show()

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        # MSE loss, without the mean. Reduction = none
        loss = (target - pred).square()

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


