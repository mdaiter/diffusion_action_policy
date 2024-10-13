from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import tinygrad
from tinygrad import Tensor, nn, TinyJit, dtypes
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from huggingface_hub import snapshot_download

from config import DiffusionConfig
from diffusion_policy import DiffusionPolicy

# Create a directory to store the video of the evaluation
output_directory = Path("outputs/eval/example_pusht_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

# Download the diffusion policy for pusht environment
pretrained_policy_path = Path(snapshot_download("lerobot/diffusion_pusht"))
# OR uncomment the following to evaluate a policy from the local outputs/train folder.
# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

# load the dict of safe_tensors
state_dict = safe_load("/Users/msd/Desktop/model2.safetensors")
policy = DiffusionPolicy(DiffusionConfig())
load_state_dict(policy, state_dict)

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position of the agent. The environment
# also automatically stops running after 300 interactions/steps.
env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=300,
)

# Reset the policy and environmens to prepare for rollout
policy.reset()
numpy_observation, info = env.reset(seed=42)

# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state
frames.append(env.render())


#@TinyJit
#@Tensor.test()
def test(numpy_observation) -> Tensor:
    state = Tensor(numpy_observation["agent_pos"], dtype=dtypes.float)
    image = Tensor(numpy_observation["pixels"], dtype=dtypes.float)
    
    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [0,1]
    image = image / 255.0
    image = image.permute(2, 0, 1)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    # Create the policy input dictionary
    observation = {
        "observation.state": state,
        "observation.image": image,
    }

    # Predict the next action with respect to the current observation
    prev_tensor_no_grad = Tensor.no_grad
    Tensor.no_grad = True
    action = policy.select_action(observation)
    Tensor.no_grad = prev_tensor_no_grad
    
    # Prepare the action for the environment
    return action.squeeze(0)

if __name__ == "__main__":
    step = 0
    done = False
    while not done:
        squeezed_action = test(numpy_observation)
    
        # Prepare the action for the environment
        numpy_action = squeezed_action.numpy()

        # Step through the environment and receive a new observation
        numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
        print(f"{step=} {reward=} {terminated=}")

        # Keep track of all the rewards and frames
        rewards.append(reward)
        frames.append(env.render())

        # The rollout is considered done when the success state is reach (i.e. terminated is True),
        # or the maximum number of iterations is reached (i.e. truncated is True)
        done = terminated | truncated | done
        step += 1

    if terminated:
        print("Success!")
    else:
        print("Failure!")

    # Get the speed of environment (i.e. its number of frames per second).
    fps = env.metadata["render_fps"]

    # Encode all frames into a mp4 video.
    video_path = output_directory / "rollout.mp4"
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

    print(f"Video of the evaluation is available in '{video_path}'.")
