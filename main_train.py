from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

import tinygrad
from tinygrad import Tensor, nn, TinyJit

from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

from config import DiffusionConfig
from diffusion_policy import DiffusionPolicy

# Start of training code

# Create a directory to store the training checkpoint.
output_directory = Path("outputs/train/example_pusht_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

# Number of offline training steps (we'll only do offline training for this example.)
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
training_steps = 40002
log_freq = 1

# Set up the dataset.
delta_timestamps = {
    # Load the previous image and state at -0.1 seconds before current frame,
    # then load current image and state corresponding to 0.0 second.
    "observation.image": [-0.1, 0.0],
    "observation.state": [-0.1, 0.0],
    # Load the previous action (-0.1), the next action to be executed (0.0),
    # and 14 future actions with a 0.1 seconds spacing. All these actions will be
    # used to supervise the policy.
    "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
}
dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)

# Set up the the policy.
# Policies are initialized with a configuration class, in this case `DiffusionConfig`.
# For this example, no arguments need to be passed because the defaults are set up for PushT.
# If you're doing something different, you will likely need to change at least some of the defaults.
cfg = DiffusionConfig()
policy = DiffusionPolicy(cfg, dataset_stats=dataset.stats)

opt = nn.optim.Adam(nn.state.get_parameters(policy), lr=1e-4)

@TinyJit
@Tensor.train()
def train_step(batch:(Tensor, Tensor, Tensor, Tensor)) -> Tensor:
    Tensor.training = True
    batch_outputs = policy.normalize_inputs_pre_call(batch)
    output_dict = policy(batch_outputs)
    loss = output_dict["loss"]
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss

if __name__ == "__main__":
    # Run training loop.
    print(f'Starting training loop')
    # Create dataloader for offline training.
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=64,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
    )

    step = 0
    done = False
    with Tensor.train():
        jit_train_step = TinyJit(train_step)
        while not done:
            for batch in dataloader:
                batch = {k: Tensor(v.numpy(), requires_grad=False) for k, v in batch.items()}
                loss = train_step(batch)
            
                if step % log_freq == 0:
                    print(f"step: {step} loss: {loss.numpy():.3f}")
                step += 1

                if step % 5000 == 0:
                    try:
                        state_dict = get_state_dict(policy)
                        safe_save(state_dict, f'{output_directory}/model_{step}.safetensors')
                    except:
                        print(f'Exception with safe save occured')
                if step >= training_steps:
                    done = True
                    break

    # Save a policy checkpoint.
    state_dict = get_state_dict(policy)
    safe_save(state_dict, f'{output_directory}/model_final.safetensors')
