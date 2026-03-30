# --------------------------------------------------------
# References:
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

from math import pi

import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat


def rotate_half_of_features(x):
    """
    Rotate the second half of the feature dimension by 90 degrees:
    (x1, x2) -> (-x2, x1)
    This operation is the core of RoPE (Rotary Position Embedding).
    """
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x_rotated = torch.stack((-x2, x1), dim=-1)
    return rearrange(x_rotated, '... d r -> ... (d r)')


def broadcast_concatenate(tensors, dim=-1):
    """
    Concatenate multiple tensors with broadcastable shapes along a specified dimension.
    All tensors must have the same number of dimensions; non-concatenation dims
    may differ in size only if the difference can be broadcast (at most 2 distinct values).

    Args:
        tensors: List of tensors to concatenate
        dim:     Dimension along which to concatenate

    Returns:
        Concatenated tensor
    """
    num_tensors = len(tensors)
    shape_ndims = set(len(t.shape) for t in tensors)
    assert len(shape_ndims) == 1, 'All tensors must have the same number of dimensions'
    ndim = list(shape_ndims)[0]
    dim = (dim + ndim) if dim < 0 else dim
    per_dim_sizes = list(zip(*[list(t.shape) for t in tensors]))
    # Non-concatenation dimensions must be broadcastable (max 2 distinct sizes)
    non_concat_dims = [(i, sizes) for i, sizes in enumerate(per_dim_sizes) if i != dim]
    assert all(len(set(sizes)) <= 2 for _, sizes in non_concat_dims), \
        'Non-concatenation dimensions do not satisfy broadcast conditions'
    max_non_concat_dims = [(i, max(sizes)) for i, sizes in non_concat_dims]
    # Build expand shapes for each tensor
    target_dim_sizes = [(i, (size,) * num_tensors) for i, size in max_non_concat_dims]
    target_dim_sizes.insert(dim, (dim, per_dim_sizes[dim]))
    expand_shapes = list(zip(*[sizes for _, sizes in target_dim_sizes]))
    tensors = [t.expand(*shape) for t, shape in zip(tensors, expand_shapes)]
    return torch.cat(tensors, dim=dim)


class VisionRotaryEmbedding(nn.Module):
    """
    Vision Rotary Position Embedding (RoPE) - standard version.
    Computes 2D RoPE for image patch sequences by concatenating
    H-direction and W-direction frequencies.
    Supports resolution interpolation when ft_seq_len != pt_seq_len.
    """
    def __init__(
        self,
        dim,
        pt_seq_len,           # patch grid side length at pretraining resolution
        ft_seq_len=None,      # patch grid side length at fine-tuning resolution (None = same as pt)
        custom_freqs=None,    # custom frequency tensor (None = auto-generated)
        freqs_for='lang',     # frequency generation mode: 'lang' / 'pixel' / 'constant'
        theta=10000,          # RoPE base frequency
        max_freq=10,          # maximum frequency for 'pixel' mode
        num_freqs=1,          # number of frequencies for 'constant' mode
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'Unknown freqs_for type: {freqs_for}')

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        # Scale timesteps proportionally to support resolution interpolation
        time_steps = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        # Compute H- and W-direction frequency tensors
        freqs_h = torch.einsum('..., f -> ... f', time_steps, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)

        freqs_w = torch.einsum('..., f -> ... f', time_steps, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)

        # Broadcast-concatenate H and W frequencies into a 2D frequency map (H, W, D)
        freqs_2d = broadcast_concatenate(
            (freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1
        )

        self.register_buffer("freqs_cos", freqs_2d.cos())
        self.register_buffer("freqs_sin", freqs_2d.sin())

    def forward(self, t, start_index=0):
        """
        Apply rotary positional encoding to input features.
        t: (..., seq_len, head_dim)
        """
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], \
            f'Feature dim {t.shape[-1]} is insufficient to rotate all positions (need {rot_dim})'

        # Only rotate dimensions [start_index:end_index]; leave the rest unchanged
        t_left = t[..., :start_index]
        t_rot = t[..., start_index:end_index]
        t_right = t[..., end_index:]
        t_rot = (t_rot * self.freqs_cos) + (rotate_half_of_features(t_rot) * self.freqs_sin)
        return torch.cat((t_left, t_rot, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    """
    Vision Rotary Position Embedding (RoPE) - fast version.
    Pre-flattens cos/sin tables to (N, D) format for efficient element-wise multiplication
    without any dynamic reshaping. Supports prepending in-context (cls) tokens
    that should not receive rotational encoding.
    """
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
        num_cls_token=0  # number of in-context / cls tokens (no rotation applied to them)
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'Unknown freqs_for type: {freqs_for}')

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        time_steps = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_2d = torch.einsum('..., f -> ... f', time_steps, freqs)
        freqs_2d = repeat(freqs_2d, '... n -> ... (n r)', r=2)
        freqs_2d = broadcast_concatenate((freqs_2d[:, None, :], freqs_2d[None, :, :]), dim=-1)

        if num_cls_token > 0:
            # Flatten image token frequencies to (N_img, D)
            freqs_flat = freqs_2d.view(-1, freqs_2d.shape[-1])
            cos_img = freqs_flat.cos()
            sin_img = freqs_flat.sin()
            N_img, D = cos_img.shape

            # In-context token positions: cos=1, sin=0 (identity rotation = no rotation)
            cos_cls_pad = torch.ones(num_cls_token, D, dtype=cos_img.dtype, device=cos_img.device)
            sin_cls_pad = torch.zeros(num_cls_token, D, dtype=sin_img.dtype, device=sin_img.device)

            # Concatenate: first num_cls_token positions are unrotated, rest are image patches
            self.freqs_cos = torch.cat([cos_cls_pad, cos_img], dim=0).cuda()  # (N_cls + N_img, D)
            self.freqs_sin = torch.cat([sin_cls_pad, sin_img], dim=0).cuda()
        else:
            # No cls tokens: directly flatten to (N_img, D)
            self.freqs_cos = freqs_2d.cos().view(-1, freqs_2d.shape[-1]).cuda()
            self.freqs_sin = freqs_2d.sin().view(-1, freqs_2d.shape[-1]).cuda()

    def forward(self, t):
        """
        Apply rotary positional encoding via fast element-wise multiplication.
        t: (B, num_heads, seq_len, head_dim)
        """
        return t * self.freqs_cos + rotate_half_of_features(t) * self.freqs_sin


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    Equivalent to T5LayerNorm / LlamaRMSNorm.
    Simpler than LayerNorm (no mean subtraction).
    Used for QK normalization and feature normalization in Transformers.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # learnable scale
        self.variance_epsilon = eps                           # prevents division by zero

    def forward(self, hidden_states):
        """
        hidden_states: (..., hidden_size)
        Computes variance in float32 to avoid numerical instability at low precision.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # Normalize by root mean square
        rms_variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(rms_variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# ============================================================
# 2D sinusoidal positional embeddings
# ============================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sinusoidal positional embeddings by concatenating
    1D encodings along the H and W axes.

    Args:
        embed_dim: Embedding dimension (must be even)
        grid_size: Grid side length (square root of the number of patches)
        cls_token: If True, prepend a zero embedding for the cls token

    Returns:
        pos_embed: (grid_size*grid_size, embed_dim) or
                   (1 + grid_size*grid_size, embed_dim) when cls_token=True
    """
    # Build (H, W) grid coordinates
    grid_h = np.arange(grid_size, dtype=np.float64)
    grid_w = np.arange(grid_size, dtype=np.float64)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # (H, W)

    grid = np.stack([grid_h.ravel(), grid_w.ravel()])  # (2, H*W)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        # cls token uses a zero positional embedding
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Compute 2D sinusoidal positional embeddings from grid coordinates.
    Uses the first half of embed_dim for H-direction and the second half for W-direction.

    Args:
        embed_dim: Total embedding dimension (must be even)
        grid:      Shape (2, H*W); grid[0] = row indices, grid[1] = column indices

    Returns:
        Shape (H*W, embed_dim) positional embedding matrix
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"

    half_dim = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(half_dim, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(half_dim, grid[1])  # (H*W, D/2)

    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, positions):
    """
    Generate 1D sinusoidal positional embeddings for a sequence of positions.

    Args:
        embed_dim: Embedding dimension (must be even)
        positions: Shape (M,) array of position coordinates

    Returns:
        Shape (M, embed_dim) positional embedding matrix
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"

    # Frequency vector: omega_i = 1 / 10000^(2i/D)
    freq_indices = np.arange(embed_dim // 2, dtype=np.float64)
    freq_indices /= embed_dim / 2.
    omega = 1. / 10000 ** freq_indices  # (D/2,)

    positions = positions.reshape(-1)  # (M,)
    # Outer product gives (M, D/2) angle matrix
    angle_matrix = np.einsum('m,d->md', positions, omega)

    emb_sin = np.sin(angle_matrix)  # (M, D/2)
    emb_cos = np.cos(angle_matrix)  # (M, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return pos_embed
