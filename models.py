import random
from tinygrad.helpers import prod
from tinygrad.tensor import Tensor
from tinygrad import nn
from tinygrad.nn.optim import Adam
from tinygrad import dtypes
import math
import numpy as np

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
        x = x.transpose(1, 2).cast(dtype=dtypes.float)

        timesteps_embed = timestep.sequential(self.diffusion_step_encoder)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        global_feature = timesteps_embed.cat(global_cond, dim=-1) if global_cond is not None else timesteps_embed
        
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

        x = x.transpose(1, 2)
        return x

import tinygrad.nn as nn
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import torch_load
from tinygrad.helpers import fetch, get_child

# allow monkeypatching in layer implementations
BatchNorm = nn.BatchNorm2d
Conv2d = nn.Conv2d
Linear = nn.Linear

class ResNet18Backbone():
    def __init__(self, use_group_norm=False):
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3)
        self.bn1 = nn.GroupNorm(num_groups=4, num_channels=64) if use_group_norm else BatchNorm(64)

        self.layer1_0 = [
            # 0
            Conv2d(64, 64, kernel_size=3, stride=1, bias=False, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=64) if use_group_norm else BatchNorm(64),
            Tensor.relu,
            Conv2d(64, 64, kernel_size=3, stride=1, bias=False, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=64) if use_group_norm else BatchNorm(64),
        ]
        self.layer1_1 = [
            # 1
            Conv2d(64, 64, kernel_size=3, stride=1, bias=False, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=64) if use_group_norm else BatchNorm(64),
            Tensor.relu,
            Conv2d(64, 64, kernel_size=3, stride=1, bias=False, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=64) if use_group_norm else BatchNorm(64),
        ]
        self.layer2_0 = [
            # 0
            Conv2d(64, 128, kernel_size=3, stride=2, bias=False, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128) if use_group_norm else BatchNorm(128),
            Tensor.relu,
            Conv2d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128) if use_group_norm else BatchNorm(128)
        ]
        self.layer2_d = [
            # 0 downsample
            Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=128) if use_group_norm else BatchNorm(128)
        ]
        self.layer2_1 = [
            # 1
            Conv2d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128) if use_group_norm else BatchNorm(64),
            Tensor.relu,
            Conv2d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128) if use_group_norm else BatchNorm(128),
        ]
        self.layer3_0 = [
            # 0
            Conv2d(128, 256, kernel_size=3, stride=2, bias=False, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=256) if use_group_norm else BatchNorm(256),
            Tensor.relu,
            Conv2d(256, 256, kernel_size=3, stride=1, bias=False, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=256) if use_group_norm else BatchNorm(256)
        ]
        self.layer3_d = [
            # 0 downsample
            Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=256) if use_group_norm else BatchNorm(256)
        ]
        self.layer3_1 = [
            # 1
            Conv2d(256, 256, kernel_size=3, stride=1, bias=False, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=256) if use_group_norm else BatchNorm(256),
            Tensor.relu,
            Conv2d(256, 256, kernel_size=3, stride=1, bias=False, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=256) if use_group_norm else BatchNorm(256),
        ]
        self.layer4_0 = [
            # 0
            Conv2d(256, 512, kernel_size=3, stride=2, bias=False, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=512) if use_group_norm else BatchNorm(512),
            Tensor.relu,
            Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=512) if use_group_norm else BatchNorm(512)
        ]
        self.layer4_d = [
            # 0 downsample
            Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=512) if use_group_norm else BatchNorm(512)
        ]
        self.layer4_1 = [
            # 1
            Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=512) if use_group_norm else BatchNorm(512),
            Tensor.relu,
            Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=512) if use_group_norm else BatchNorm(512),
        ]

    
    def __call__(self, x:Tensor) -> Tensor:
        out = self.bn1(self.conv1(x)).relu()
        out = out.max_pool2d(kernel_size=(3,3), stride=2, padding=1, dilation=1)
        out = out.sequential(self.layer1_0)
        out = out.sequential(self.layer1_1)
        out = out.sequential(self.layer2_0) + out.sequential(self.layer2_d)
        out = out.sequential(self.layer2_1)
        out = out.sequential(self.layer3_0) + out.sequential(self.layer3_d)
        out = out.sequential(self.layer3_1)
        out = out.sequential(self.layer4_0) + out.sequential(self.layer4_d)
        out = out.sequential(self.layer4_1)
        return out
   
def center_crop_f(output_size: tuple, img: Tensor) -> Tensor:
        h, w = img.shape[-2:]
        th, tw = output_size
    
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
    
        return img[..., i:i+th, j:j+tw]

def random_crop_f(size:tuple, img: Tensor) -> Tensor:
    h, w = img.shape[-2:]
    th, tw = size
    
    if h < th or w < tw:
        raise ValueError("Requested crop size is larger than the image size")
    
    i = np.random.randint(0, h - th + 1)
    j = np.random.randint(0, w - tw + 1)
    
    return img[..., i:i+th, j:j+tw]

class DiffusionRgbEncoder():
    """Encoder an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: DiffusionConfig):
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = lambda x: center_crop_f(config.crop_shape, x)
            if config.crop_is_random:
                self.maybe_random_crop = lambda x: random_crop_f(config.crop_shape, x)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = ResNet18Backbone(use_group_norm=config.use_group_norm) 
        #self.backbone = generate_resnet_model(config.vision_backbone, use_group_norm=config.use_group_norm)
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
        dummy_input = Tensor.zeros(1, config.input_shapes[image_key][0], *dummy_input_h_w, requires_grad=False)
        #prev_tensor_no_grad = Tensor.no_grad
        Tensor.no_grad = True
        dummy_feature_map = self.backbone(dummy_input)
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
        if self.do_crop:
            if True:#self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = self.pool(self.backbone(x)).flatten(start_dim=1)
        # Final linear layer with non-linearity.
        x = self.out(x).relu()
        return x
