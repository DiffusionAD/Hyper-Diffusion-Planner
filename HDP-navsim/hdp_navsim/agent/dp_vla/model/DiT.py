# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.config import use_fused_attn
from timm.models.vision_transformer import Attention, Mlp
from torch.jit import Final


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CustomCrossAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.proj_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.proj_k(context).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.proj_v(context).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            if key_padding_mask is not None:
                attn_mask = key_padding_mask.view(B, 1, 1, M).to(dtype=torch.bool)
            else:
                attn_mask = None
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if key_padding_mask is not None:
                mask = key_padding_mask.view(B, 1, 1, M)
                attn = attn.masked_fill(mask == 0, float('-inf'))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=100):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(t.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, **block_kwargs)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.1)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CustomCrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, **block_kwargs)
        self.norm4 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 12 * hidden_size, bias=True)
        )

    def forward(self, x, t, c, c_mask):
        shift_msa, scale_msa, gate_msa, \
        shift_mca, scale_mca, gate_mca, \
        shift_mlp, scale_mlp, gate_mlp, \
        shift_cross_mlp, scale_cross_mlp, gate_cross_mlp = self.adaLN_modulation(t).chunk(12, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(modulate(self.norm3(x), shift_mca, scale_mca), c, c_mask)
        x = x + gate_cross_mlp.unsqueeze(1) * self.cross_mlp(modulate(self.norm4(x), shift_cross_mlp, scale_cross_mlp))
        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.proj(x)
