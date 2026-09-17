import collections
import itertools

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path)"""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


# adapted from timm (timm/models/layers/helpers.py)
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            assert len(x) == n
            return x
        return tuple(itertools.repeat(x, n))

    return parse


# adapted from timm (timm/models/layers/helpers.py)
def to_ntuple(x, n):
    return _ntuple(n=n)(x)


# from kappamodules.functional.pos_embed import interpolate_sincos
def interpolate_sincos(embed, seqlens, mode="bicubic"):
    assert embed.ndim - 2 == len(seqlens)
    embed = F.interpolate(
        einops.rearrange(embed, "1 ... dim -> 1 dim ..."),
        size=seqlens,
        mode=mode,
    )
    embed = einops.rearrange(embed, "1 dim ... -> 1 ... dim")
    return embed


# from kappamodules.vit import VitPatchEmbed
class VitPatchEmbed(nn.Module):
    def __init__(self, dim, num_channels, resolution, patch_size, stride=None, init_weights="xavier_uniform"):
        super().__init__()
        self.resolution = resolution
        self.init_weights = init_weights
        self.ndim = len(resolution)
        self.patch_size = to_ntuple(patch_size, n=self.ndim)
        if stride is None:
            self.stride = self.patch_size
        else:
            self.stride = to_ntuple(stride, n=self.ndim)
        for i in range(self.ndim):
            assert resolution[i] % self.patch_size[i] == 0, \
                f"resolution[{i}] % patch_size[{i}] != 0 (resolution={resolution} patch_size={patch_size})"
        self.seqlens = [resolution[i] // self.patch_size[i] for i in range(self.ndim)]
        if self.patch_size == self.stride:
            # use primitive type as np.prod gives np.int which is not compatible with all serialization/logging
            self.num_patches = int(np.prod(self.seqlens))
        else:
            if self.ndim == 1:
                conv_func = F.conv1d
            elif self.ndim == 2:
                conv_func = F.conv2d
            elif self.ndim == 3:
                conv_func = F.conv3d
            else:
                raise NotImplementedError
            self.num_patches = conv_func(
                input=torch.zeros(1, 1, *resolution),
                weight=torch.zeros(1, 1, *self.patch_size),
                stride=self.stride,
            ).numel()

        if self.ndim == 1:
            conv_ctor = nn.Conv1d
        elif self.ndim == 2:
            conv_ctor = nn.Conv2d
        elif self.ndim == 3:
            conv_ctor = nn.Conv3d
        else:
            raise NotImplementedError

        self.proj = conv_ctor(num_channels, dim, kernel_size=self.patch_size, stride=self.stride)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            # initialize as nn.Linear
            w = self.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.zeros_(self.proj.bias)
        else:
            raise NotImplementedError

    def forward(self, x):
        assert all(x.size(i + 2) % self.patch_size[i] == 0 for i in range(self.ndim)), \
            f"x.shape={x.shape} incompatible with patch_size={self.patch_size}"
        x = self.proj(x)
        _, _, H, W, L = x.shape
        x = einops.rearrange(x, "b c ... -> b ... c")
        return x, H, W, L


# from kappamodules.vit import VitPosEmbed2d
class VitPosEmbed2d(nn.Module):
    def __init__(self, seqlens, dim: int, allow_interpolation: bool = True):
        super().__init__()
        self.seqlens = seqlens
        self.dim = dim
        self.allow_interpolation = allow_interpolation
        self.embed = nn.Parameter(torch.zeros(1, *seqlens, dim))
        self.reset_parameters()

    @property
    def _expected_x_ndim(self):
        return len(self.seqlens) + 2

    def reset_parameters(self):
        nn.init.trunc_normal_(self.embed, std=.02)

    def forward(self, x):
        assert x.ndim == self._expected_x_ndim
        if x.shape[1:] != self.embed.shape[1:]:
            assert self.allow_interpolation
            embed = interpolate_sincos(embed=self.embed, seqlens=x.shape[1:-1])
        else:
            embed = self.embed
        return x + embed



