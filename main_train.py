from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

import tinygrad
from tinygrad import Tensor, nn, TinyJit, dtypes

from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

from config import DiffusionConfig
from diffusion_policy import DiffusionPolicy

# Start of training code

# Create a directory to store the training checkpoint.
output_directory = Path("outputs/train/example_pusht_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

# Number of offline training steps (we'll only do offline training for this example.)
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
training_steps = 40000
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

policy_parameters = nn.state.get_parameters(policy)

opt = nn.optim.AdamW(policy_parameters, lr=1e-4, b1=0.95, b2=0.999, weight_decay=1e-6)

def clip_grad_norm_(parameters, max_norm) -> Tensor:
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    
    max_norm = float(max_norm)
    
    if len(parameters) == 0:
        return Tensor(0.0)
    
    # For Metal, we need this: Metal can only support 32 memory buffers
    is_32 = 0
    total_norm = Tensor.zeros((), dtype=dtypes.float32)
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.flatten().square().sum().contiguous().realize()
            total_norm += param_norm
            is_32 += 1
            if is_32 > 29:
                total_norm = total_norm.contiguous().realize()
                is_32 = 0
    
    # We need one extra realize here: make sure there's nothing left in the parameter calc
    total_norm = (total_norm.realize() + 1e-12).sqrt()
    print(f'total_norm: {total_norm.numpy()}')
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = Tensor.minimum(clip_coef, Tensor.ones_like(clip_coef)).contiguous().realize()
    print(f'clip_coef: {clip_coef}')
    
    for p in parameters:
        if p.grad is not None:
            p.grad = p.grad.contiguous() * clip_coef
    
    return total_norm.realize()

@TinyJit
@Tensor.train()
def train_step(
    observation_image: Tensor,
    observation_state: Tensor,
    action: Tensor,
    episode_index: Tensor,
    frame_index: Tensor,
    timestamp: Tensor,
    next_reward: Tensor,
    next_done: Tensor,
    next_success: Tensor,
    index: Tensor,
    observation_image_is_pad: Tensor,
    observation_state_is_pad: Tensor,
    action_is_pad: Tensor
) -> Tensor:
    Tensor.training = True
    batch = {
        'observation.image': observation_image,
        'observation.state': observation_state,
        'action': action,
        'episode_index': episode_index,
        'frame_index': frame_index,
        'timestamp': timestamp,
        'next.reward': next_reward,
        'index': index,
        'observation.image_is_pad': observation_image_is_pad,
        'observation.state_is_pad': observation_state_is_pad,
        'action_is_pad': action_is_pad
    }
    output_dict = policy(batch)
    loss = output_dict["loss"]
    opt.zero_grad()
    loss.backward()
    grad_norm = clip_grad_norm_(policy_parameters, 10.0)
    opt.step()
    return (loss, grad_norm)

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
                print(f'batch: {batch}')
                loss_gradnorm_tuple = train_step(
                    batch['observation.image'].realize(),
                    batch['observation.state'].realize(),
                    batch['action'].realize(),
                    batch['episode_index'].realize(),
                    batch['frame_index'].realize(),
                    batch['timestamp'].realize(),
                    batch['next.reward'].realize(),
                    batch['next.done'].realize(),
                    batch['next.success'].realize(),
                    batch['index'].realize(),
                    batch['observation.image_is_pad'].realize(),
                    batch['observation.state_is_pad'].realize(),
                    batch['action_is_pad'].realize()
                )

                loss = loss_gradnorm_tuple[0]
                gradnorm = loss_gradnorm_tuple[1]
            
                if step % log_freq == 0:
                    print(f"step: {step} loss: {loss.numpy():.3f} gradnorm: {gradnorm.numpy():.3f}")
                step += 1

                if step % 1000 == 0:
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
