import random
from tinygrad.helpers import prod
from tinygrad.tensor import Tensor
from tinygrad import nn
from tinygrad.nn.optim import Adam
from tinygrad import dtypes
import math
import einops

from networks import DiffusionSinusoidalPosEmb, DiffusionConv1dBlock, DiffusionConditionalResidualBlock1d, SpatialSoftmax
from config import DiffusionConfig

class DiffusionConditionalResidualBlock1d():
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = [
            Tensor.mish,
            nn.Linear(cond_dim, cond_channels)
        ]

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def __call__(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = cond.sequential(self.cond_encoder).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        if self.in_channels != self.out_channels:
            out = out + self.residual_conv(x)
        return out

class DiffusionConditionalUnet1d():
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        self.config = config

        # Encoder for the diffusion timestep.
        print(f'DiffusionConditionalUnet1d: config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim: {(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim)}')
        self.diffusion_step_encoder = [
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            Tensor.mish,
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        ]

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim
        print(f'config.diffusion_step_embed_dim: {config.diffusion_step_embed_dim}')
        print(f'global_cond_dim: {global_cond_dim}')
        print(f'cond_dim: {cond_dim}')

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.output_shapes["action"][0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = []
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append([
                DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else None
            ])

        # Processing in the middle of the auto-encoder.
        self.mid_modules = [
            DiffusionConditionalResidualBlock1d(
                config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
            ),
            DiffusionConditionalResidualBlock1d(
                config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
            ),
        ]

        # Unet decoder.
        self.up_modules = []
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append([
                # dim_in * 2, because it takes the encoder's skip connection as well
                DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else None
            ])
        
        self.final_conv = [
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.output_shapes["action"][0], 1),
        ]

    def __call__(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t").cast(dtype=dtypes.float)

        timesteps_embed = timestep.sequential(self.diffusion_step_encoder)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        global_feature = Tensor.cat(*[timesteps_embed, global_cond], dim=-1) if global_cond is not None else timesteps_embed
        
        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            if downsample is not None:
                x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = x.cat(encoder_skip_features.pop(), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            if upsample is not None:
                x = upsample(x)

        x = x.sequential(self.final_conv)

        x = einops.rearrange(x, "b d t -> b t d")
        return x

import tinygrad.nn as nn
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import torch_load
from tinygrad.helpers import fetch, get_child

# allow monkeypatching in layer implementations
BatchNorm = nn.BatchNorm2d
Conv2d = nn.Conv2d
Linear = nn.Linear


class BasicBlock:
  expansion = 1

  def __init__(self, in_planes, planes, stride=1, groups=1, base_width=64, use_group_norm=False):
    assert groups == 1 and base_width == 64, "BasicBlock only supports groups=1 and base_width=64"
    self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.GroupNorm(num_groups=planes // 16, num_channels=planes) if use_group_norm else BatchNorm(planes)
    self.conv2 = Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = nn.GroupNorm(num_groups=planes // 16, num_channels=planes) if use_group_norm else BatchNorm(planes)
    self.downsample = []
    if stride != 1 or in_planes != self.expansion*planes:
      self.downsample = [
        Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        nn.GroupNorm(num_groups=self.expansion*planes // 16, num_channels=self.expansion*planes) if use_group_norm else BatchNorm(self.expansion*planes)
      ]

  def __call__(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out))
    out = out + x.sequential(self.downsample)
    out = out.relu()
    return out


class Bottleneck:
  # NOTE: stride_in_1x1=False, this is the v1.5 variant
  expansion = 4

  def __init__(self, in_planes, planes, stride=1, stride_in_1x1=False, groups=1, base_width=64, use_group_norm=False):
    width = int(planes * (base_width / 64.0)) * groups
    # NOTE: the original implementation places stride at the first convolution (self.conv1), control with stride_in_1x1
    self.conv1 = Conv2d(in_planes, width, kernel_size=1, stride=stride if stride_in_1x1 else 1, bias=False)
    self.bn1 = nn.GroupNorm(num_groups=width // 16, num_channels=width) if use_group_norm else BatchNorm(width)
    self.conv2 = Conv2d(width, width, kernel_size=3, padding=1, stride=1 if stride_in_1x1 else stride, groups=groups, bias=False)
    self.bn2 = nn.GroupNorm(num_groups=width // 16, num_channels=width) if use_group_norm else BatchNorm(width)
    self.conv3 = Conv2d(width, self.expansion*planes, kernel_size=1, bias=False)
    self.bn3 = nn.GroupNorm(num_groups=self.expansion*planes // 16, num_channels=self.expansion*planes) if use_group_norm else BatchNorm(self.expansion*planes)
    self.downsample = []
    if stride != 1 or in_planes != self.expansion*planes:
      self.downsample = [
        Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        nn.GroupNorm(num_groups=self.expansion*planes // 16, num_channels=self.expansion*planes) if use_group_norm else BatchNorm(self.expansion*planes)
      ]

  def __call__(self, x):
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out)).relu()
    out = self.bn3(self.conv3(out))
    out = out + x.sequential(self.downsample)
    out = out.relu()
    return out

class ResNet:
  def __init__(self, num, num_classes=None, groups=1, width_per_group=64, stride_in_1x1=False, use_group_norm=False):
    self.num = num
    self.block = {
      18: BasicBlock,
      34: BasicBlock,
      50: Bottleneck,
      101: Bottleneck,
      152: Bottleneck
    }[num]

    self.num_blocks = {
      18: [2,2,2,2],
      34: [3,4,6,3],
      50: [3,4,6,3],
      101: [3,4,23,3],
      152: [3,8,36,3]
    }[num]

    self.in_planes = 64

    self.groups = groups
    self.base_width = width_per_group
    self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3)
    self.bn1 = nn.GroupNorm(num_groups=64 // 16, num_channels=64) if use_group_norm else BatchNorm(64)
    self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1, stride_in_1x1=stride_in_1x1, use_group_norm=use_group_norm)
    self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2, stride_in_1x1=stride_in_1x1, use_group_norm=use_group_norm)
    self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2, stride_in_1x1=stride_in_1x1, use_group_norm=use_group_norm)
    self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2, stride_in_1x1=stride_in_1x1, use_group_norm=use_group_norm)
    self.fc = Linear(in_features = int(512 * self.block.expansion), out_features = num_classes) if num_classes is not None and use_group_norm is False else None

  def _make_layer(self, block, planes, num_blocks, stride, stride_in_1x1, use_group_norm):
    strides = [stride] + [1] * (num_blocks-1)
    layers = []
    for stride in strides:
      if block == Bottleneck:
        layers.append(block(self.in_planes, planes, stride, stride_in_1x1, self.groups, self.base_width, use_group_norm=use_group_norm))
      else:
        layers.append(block(self.in_planes, planes, stride, self.groups, self.base_width, use_group_norm=use_group_norm))
      self.in_planes = planes * block.expansion
    return layers

  def backbone(self, x:Tensor) -> Tensor:
    out = self.bn1(self.conv1(x)).relu()
    out = out.pad2d([1,1,1,1]).max_pool2d((3,3), 2)
    out = out.sequential(self.layer1)
    out = out.sequential(self.layer2)
    out = out.sequential(self.layer3)
    out = out.sequential(self.layer4)
    return out
    
  def __call__(self, x:Tensor) -> Tensor:
    is_feature_only = self.fc is None
    if is_feature_only: features = []
    out = self.bn1(self.conv1(x)).relu()
    out = out.pad2d([1,1,1,1]).max_pool2d((3,3), 2)
    out = out.sequential(self.layer1)
    if is_feature_only: features.append(out)
    out = out.sequential(self.layer2)
    if is_feature_only: features.append(out)
    out = out.sequential(self.layer3)
    if is_feature_only: features.append(out)
    out = out.sequential(self.layer4)
    if is_feature_only: features.append(out)
    if not is_feature_only:
      out = out.mean([2,3])
      out = self.fc(out.cast(dtypes.float32))
      return out
    return features

  def load_from_pretrained(self):
    model_urls = {
      (18, 1, 64): 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
      (34, 1, 64): 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
      (50, 1, 64): 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
      (50, 32, 4): 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
      (101, 1, 64): 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
      (152, 1, 64): 'https://download.pytorch.org/model|s/resnet152-b121ed2d.pth',
    }

    self.url = model_urls[(self.num, self.groups, self.base_width)]
    for k, dat in torch_load(fetch(self.url)).items():
      obj: Tensor = get_child(self, k)

      if 'fc.' in k and obj.shape != dat.shape:
        print("skipping fully connected layer")
        continue # Skip FC if transfer learning

      if 'bn' not in k and 'downsample' not in k: assert obj.shape == dat.shape, (k, obj.shape, dat.shape)
      obj.assign(dat.to(obj.device).reshape(obj.shape))

def generate_resnet_model(resnet_model, use_group_norm=False):
    return {
        'resnet18' : ResNet(18, num_classes=1000, use_group_norm=use_group_norm),
        'resnet34' : ResNet(34, num_classes=1000, use_group_norm=use_group_norm),
        'resnet50' : ResNet(50, num_classes=1000, use_group_norm=use_group_norm),
        'resnet101' : ResNet(101, num_classes=1000, use_group_norm=use_group_norm),
        'resnet152' : ResNet(152, num_classes=1000, use_group_norm=use_group_norm),
        'resnext50_32x4d' : ResNet(50, num_classes=1000, groups=32, width_per_group=4, use_group_norm=use_group_norm),
    }[resnet_model]

class DiffusionRgbEncoder():
    """Encoder an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: DiffusionConfig):
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            #self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            #if config.crop_is_random:
            #    self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            #else:
            #    self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = generate_resnet_model(config.vision_backbone, use_group_norm=config.use_group_norm)
        if config.use_group_norm and config.pretrained_backbone_weights:
            raise ValueError(
                "You can't replace BatchNorm in a pretrained model without ruining the weights!"
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.input_shapes` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.input_shapes`.
        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        # Note: we have a check in the config class to make sure all images have the same shape.
        image_key = image_keys[0]
        dummy_input_h_w = (
            config.crop_shape if config.crop_shape is not None else config.input_shapes[image_key][1:]
        )
        dummy_input = Tensor.zeros(1, config.input_shapes[image_key][0], *dummy_input_h_w)
        #prev_tensor_no_grad = Tensor.no_grad
        Tensor.no_grad = True
        dummy_feature_map = self.backbone.backbone(dummy_input)
        print(f'dummy_feature_map: {dummy_feature_map}')
        Tensor.no_grad = False
        feature_map_shape = tuple(dummy_feature_map.shape[1:])
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if False:# self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = self.pool(self.backbone.backbone(x)).flatten(start_dim=1)
        # Final linear layer with non-linearity.
        x = self.out(x).relu()
        return x
