from tinygrad import Tensor, nn, dtypes

def create_stats_buffers(
    shapes: dict[str, list[int]],
    modes: dict[str, str],
    stats: dict[str, dict[str, Tensor]] | None = None,
) -> dict[str, dict[str, dict]]:
    """
    Create buffers per modality (e.g. "observation.image", "action") containing their mean, std, min, max
    statistics.

    Args: (see Normalize and Unnormalize)

    Returns:
        dict: A dictionary where keys are modalities and values are `nn.ParameterDict` containing
            `nn.Parameters` set to `requires_grad=False`, suitable to not be updated during backpropagation.
    """
    stats_buffers = {}

    for key, mode in modes.items():
        assert mode in ["mean_std", "min_max"]

        shape = tuple(shapes[key])

        if "image" in key:
            # sanity checks
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            # override image shape to be invariant to height and width
            shape = (c, 1, 1)

        # Note: we initialize mean, std, min, max to infinity. They should be overwritten
        # downstream by `stats` or `policy.load_state_dict`, as expected. During forward,
        # we assert they are not infinity anymore.

        buffer = {}
        if mode == "mean_std":
            buffer = {
                "mean": Tensor.ones(shape, dtype=dtypes.float, requires_grad=False) * float('inf'),
                "std": Tensor.ones(shape, dtype=dtypes.float, requires_grad=False) * float('inf'),
            }
        elif mode == "min_max":
            buffer = {
                "min": Tensor.ones(shape, dtype=dtypes.float, requires_grad=False) * float('inf'),
                "max": Tensor.ones(shape, dtype=dtypes.float, requires_grad=False) * float('inf'),
            }

        if stats is not None:
            # Note: The clone is needed to make sure that the logic in save_pretrained doesn't see duplicated
            # tensors anywhere (for example, when we use the same stats for normalization and
            # unnormalization). See the logic here
            # https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L97.
            if mode == "mean_std":
                buffer["mean"].assign(stats[key]["mean"].numpy())
                buffer["mean"].requires_grad = False
                buffer["std"].assign(stats[key]["std"].numpy())
                buffer["std"].requires_grad = False
            elif mode == "min_max":
                buffer["min"].assign(stats[key]["min"].numpy())
                buffer["min"].requires_grad = False
                buffer["max"].assign(stats[key]["max"].numpy())
                buffer["max"].requires_grad = False

        stats_buffers[key] = buffer
    return stats_buffers


def _no_stats_error_str(name: str) -> str:
    return (
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a "
        "pretrained model."
    )


class Normalize():
    """Normalizes data (e.g. "observation.image") for more stable and faster convergence during training."""

    def __init__(
        self,
        shapes: dict[str, list[int]],
        modes: dict[str, str],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            shapes (dict): A dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (dict): A dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (dict, optional): A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
        """
        self.shapes = shapes
        self.modes = modes
        self.stats = stats
        stats_buffers = create_stats_buffers(shapes, modes, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove Tensor.no_grad?
    # @Tensor.no_grad
    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        Tensor.no_grad = True
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, mode in self.modes.items():
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if mode == "mean_std":
                mean = buffer["mean"]
                std = buffer["std"]
                assert not (mean == float('inf')).any().numpy(), _no_stats_error_str("mean")
                assert not (std == float('inf')).any().numpy(), _no_stats_error_str("std")
                #print(f'mean: {mean.numpy()}, std: {std.numpy()}')
                #print(f'batch[{key}] before normalization: {batch[key].numpy()}')
                batch[key] = (batch[key] - mean) / (std + 1e-8)
                #print(f'batch[{key}] after normalization: {batch[key].numpy()}')
            elif mode == "min_max":
                min = buffer["min"]
                max = buffer["max"]
                assert not (min == float('inf')).any().numpy(), _no_stats_error_str("min")
                assert not (max == float('inf')).any().numpy(), _no_stats_error_str("max")
                # normalize to [0,1]
                #print(f'max: {max.numpy()}, min: {min.numpy()}')
                #print(f'batch[{key}] before normalization: {batch[key].numpy()}')
                batch[key] = (batch[key] - min) / (max - min + 1e-8)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
                #print(f'batch[{key}] after normalization: {batch[key].numpy()}')
            else:
                raise ValueError(mode)
        Tensor.no_grad = False
        return batch


class Unnormalize():
    """
    Similar to `Normalize` but unnormalizes output data (e.g. `{"action": torch.randn(b,c)}`) in their
    original range used by the environment.
    """

    def __init__(
        self,
        shapes: dict[str, list[int]],
        modes: dict[str, str],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            shapes (dict): A dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (dict): A dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (dict, optional): A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
        """
        self.shapes = shapes
        self.modes = modes
        self.stats = stats
        # `self.buffer_observation_state["mean"]` contains `torch.tensor(state_dim)`
        stats_buffers = create_stats_buffers(shapes, modes, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        Tensor.no_grad = True
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, mode in self.modes.items():
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if mode == "mean_std":
                mean = buffer["mean"]
                std = buffer["std"]
                assert not (mean == float('inf')).any().numpy(), _no_stats_error_str("mean")
                assert not (std == float('inf')).any().numpy(), _no_stats_error_str("std")
                batch[key] = batch[key] * std + mean
            elif mode == "min_max":
                min = buffer["min"]
                max = buffer["max"]
                assert not (min == float('inf')).any().numpy(), _no_stats_error_str("min")
                assert not (max == float('inf')).any().numpy(), _no_stats_error_str("max")
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (max - min) + min
            else:
                raise ValueError(mode)
        Tensor.no_grad = False
        return batch
