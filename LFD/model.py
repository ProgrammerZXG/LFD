# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from util.model_util import get_2d_sincos_pos_embed, RMSNorm, VisionRotaryEmbeddingFast


def apply_adaptive_layer_norm_modulation(hidden_states, shift, scale):
    """
    Apply adaptive LayerNorm (adaLN) modulation to hidden states.
    Formula: output = hidden_states * (1 + scale) + shift
    """
    return hidden_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class IdentityRoPE(nn.Module):
    """
    Placeholder RoPE module that returns the input unchanged.
    Used for 3D LFD scenarios where RoPE is not yet implemented.
    """
    def forward(self, x):
        return x


class BottleneckPatchEmbed(nn.Module):
    """
    Image patch embedding layer with a bottleneck structure.
    First projects image patches to a low-dimensional bottleneck (pca_dim)
    using a large-kernel convolution, then expands to embed_dim via 1x1 conv.
    Output shape: (B, num_patches, embed_dim)
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        pca_dim=768,
        embed_dim=768,
        bias=True
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Step 1: Split image into patches and project to low-dim bottleneck (pca_dim)
        self.patch_proj = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        # Step 2: 1x1 conv expands bottleneck dim to final embed_dim
        self.expand_proj = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        # patch_proj -> expand_proj -> flatten patches -> transpose to (B, num_patches, embed_dim)
        x = self.expand_proj(self.patch_proj(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Encodes scalar diffusion timesteps into vector representations.
    Uses sinusoidal frequency encoding followed by a two-layer MLP.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.timestep_mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def build_sinusoidal_embedding(t, dim, max_period=10000):
        """
        Build sinusoidal position embeddings for a batch of timesteps.

        Args:
            t:          1D Tensor of shape (N,), one timestep per batch element (can be fractional)
            dim:        Output embedding dimension
            max_period: Controls the minimum frequency of the embeddings

        Returns:
            Tensor of shape (N, dim) with sinusoidal embeddings
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half_dim = dim // 2
        freq_bands = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        ).to(device=t.device)
        freq_inputs = t[:, None].float() * freq_bands[None]
        embedding = torch.cat([torch.cos(freq_inputs), torch.sin(freq_inputs)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """Encode timestep t: (B,) -> (B, hidden_size)"""
        freq_features = self.build_sinusoidal_embedding(t, self.frequency_embedding_size)
        timestep_emb = self.timestep_mlp(freq_features)
        return timestep_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Supports Classifier-Free Guidance (CFG) by replacing dropped labels
    with num_classes (the "null" class index).
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        # Table size is num_classes + 1; the last index is reserved for the null class (CFG)
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        """Map class labels (B,) to embeddings (B, hidden_size)"""
        embeddings = self.embedding_table(labels)
        return embeddings


def manual_scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    """
    Manual implementation of scaled dot-product attention.
    Forces float32 computation to avoid precision loss with bfloat16.
    query/key/value shape: (B, num_heads, seq_len, head_dim)
    """
    query_len, key_len = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    # Initialize attention bias to zero (no masking)
    attn_bias = torch.zeros(query.size(0), 1, query_len, key_len, dtype=query.dtype).cuda()

    # Force float32 dot-product to prevent precision overflow
    with torch.cuda.amp.autocast(enabled=False):
        attn_weights = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weights += attn_bias
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_weights = torch.dropout(attn_weights, dropout_p, train=True)
    return attn_weights @ value


class Attention(nn.Module):
    """
    Multi-head self-attention with QK normalization and RoPE positional encoding.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # QK normalization using RMSNorm for training stability
        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        """
        Args:
            x:    (B, N, C)
            rope: Rotary positional embedding module applied to Q and K
        """
        B, N, C = x.shape
        # Compute Q, K, V and reshape for multi-head format
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply rotary positional encoding
        q = rope(q)
        k = rope(k)

        # Scaled dot-product attention
        x = manual_scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        # Reshape and project output
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    Gating mechanism: output = W3(SiLU(W1(x)) * W2(x))
    Typically outperforms standard FFN with similar parameter count.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        # SwiGLU uses 2/3 of hidden_dim to match standard FFN parameter count
        hidden_dim = int(hidden_dim * 2 / 3)
        # w12 simultaneously produces gate and value feature maps
        self.gate_value_proj = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.output_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        """x: (B, N, dim) -> (B, N, dim)"""
        gate_and_value = self.gate_value_proj(x)
        gate, value = gate_and_value.chunk(2, dim=-1)
        # SwiGLU activation: SiLU(gate) * value
        gated_hidden = F.silu(gate) * value
        return self.output_proj(self.ffn_dropout(gated_hidden))


class FinalLayer(nn.Module):
    """
    Final output layer of LFD.
    Applies adaLN modulation to Transformer outputs, then projects
    each patch token back to pixel space.
    Output shape: (B, num_patches, patch_size^2 * out_channels)
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        # Linear layer: maps each patch token to patch_size^2 * out_channels pixel values
        self.pixel_proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # adaLN modulation: generates shift and scale from conditioning embedding
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, conditioning_emb):
        """
        Args:
            x:                (B, num_patches, hidden_size)
            conditioning_emb: (B, hidden_size), combined timestep + class embedding
        """
        shift, scale = self.adaLN_modulation(conditioning_emb).chunk(2, dim=1)
        x = apply_adaptive_layer_norm_modulation(self.norm_final(x), shift, scale)
        x = self.pixel_proj(x)
        return x


class LFDBlock(nn.Module):
    """
    LFD Transformer block.
    Contains adaLN-modulated multi-head self-attention + adaLN-modulated SwiGLU FFN.
    Both sub-layers use gated residual connections: x = x + gate * sublayer(x).
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm_attn = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.norm_ffn = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.ffn = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        # adaLN modulation: produces 6 vectors for attention (shift/scale/gate) + FFN (shift/scale/gate)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, conditioning_emb, rope=None):
        """
        Args:
            x:                (B, N, hidden_size)
            conditioning_emb: (B, hidden_size), combined timestep + class embedding
            rope:             Rotary positional embedding module
        """
        # Decompose conditioning into 6 modulation parameters
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = \
            self.adaLN_modulation(conditioning_emb).chunk(6, dim=-1)

        # Attention branch with gated residual
        x = x + gate_attn.unsqueeze(1) * self.attn(
            apply_adaptive_layer_norm_modulation(self.norm_attn(x), shift_attn, scale_attn),
            rope=rope
        )
        # FFN branch with gated residual
        x = x + gate_ffn.unsqueeze(1) * self.ffn(
            apply_adaptive_layer_norm_modulation(self.norm_ffn(x), shift_ffn, scale_ffn)
        )
        return x


class LFD(nn.Module):
    """
    LFD main model (Latent-Free Diffusion) for structural modeling.
    Conditioned on fault (Fault) and horizon (Horizon) maps to predict RGT.

    Key design choices:
    - Input-level injection: fault + horizon tokens are added directly to patch tokens
    - Per-layer residual injection: each Transformer block injects fault/horizon
      via learnable alpha-scaled linear projections
    - In-context tokens: learnable class tokens inserted at a specified depth
      for enhanced global conditioning
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len       # number of in-context tokens to insert
        self.in_context_start = in_context_start   # block index where in-context tokens are inserted
        self.num_classes = num_classes

        # ---- Conditioning embedders ----
        self.timestep_embedder = TimestepEmbedder(hidden_size)
        self.class_label_embedder = LabelEmbedder(num_classes, hidden_size)

        # ---- Patch embedders ----
        # Noisy target image patch embedding
        self.x_embedder = BottleneckPatchEmbed(
            input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True
        )
        # Fault condition patch embedding (single channel)
        self.fault_embedder = BottleneckPatchEmbed(
            input_size, patch_size, 1, bottleneck_dim, hidden_size, bias=True
        )
        # Horizon condition patch embedding (single channel)
        self.horizon_embedder = BottleneckPatchEmbed(
            input_size, patch_size, 1, bottleneck_dim, hidden_size, bias=True
        )

        # ---- Fixed sinusoidal positional embedding ----
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # ---- Learnable positional embedding for in-context tokens ----
        if self.in_context_len > 0:
            self.in_context_pos_embed = nn.Parameter(
                torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True
            )
            torch.nn.init.normal_(self.in_context_pos_embed, std=.02)

        # ---- Rotary positional embeddings (RoPE) ----
        half_head_dim = hidden_size // num_heads // 2
        patch_grid_size = input_size // patch_size  # number of patches per row/col
        # RoPE without in-context tokens (used for early blocks)
        self.rope_without_context = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=patch_grid_size,
            num_cls_token=0
        )
        # RoPE with in-context tokens (used after in-context tokens are inserted)
        self.rope_with_context = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=patch_grid_size,
            num_cls_token=self.in_context_len
        )

        # ---- Transformer blocks ----
        self.blocks = nn.ModuleList([
            LFDBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                # Enable dropout only in the middle quarter of blocks
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0
            )
            for i in range(depth)
        ])

        # ---- Per-layer horizon / fault residual injection ----
        # Each block has its own linear projection to inject condition tokens
        self.horizon_residual_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=True)
            for _ in range(depth)
        ])
        self.fault_residual_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=True)
            for _ in range(depth)
        ])

        # Learnable injection strength coefficients (initialized to 0)
        self.horizon_inject_alpha = nn.Parameter(torch.zeros(depth))
        self.fault_inject_alpha = nn.Parameter(torch.zeros(depth))

        # ---- Final output layer ----
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize all model weights."""
        def _xavier_linear_init(module):
            """Xavier uniform init for Linear layers; zero-init bias."""
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_xavier_linear_init)

        # Initialize fixed sinusoidal positional embedding (frozen)
        sincos_pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(sincos_pos_embed).float().unsqueeze(0))

        # Initialize patch embedder convolution weights with Xavier uniform
        for embedder in [self.x_embedder, self.fault_embedder, self.horizon_embedder]:
            w1 = embedder.patch_proj.weight.data
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            w2 = embedder.expand_proj.weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(embedder.expand_proj.bias, 0)

        # Initialize label embedding table and timestep MLP with small std
        nn.init.normal_(self.class_label_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.timestep_mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_embedder.timestep_mlp[2].weight, std=0.02)

        # Zero-init all adaLN final linear layers (identity transform at init)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-init final layer adaLN and pixel projection
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.pixel_proj.weight, 0)
        nn.init.constant_(self.final_layer.pixel_proj.bias, 0)

        # Zero-init horizon/fault residual injection layers (no injection at start of training)
        for layer in self.horizon_residual_layers:
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)
        for layer in self.fault_residual_layers:
            nn.init.constant_(layer.weight, 0)
            nn.init.constant_(layer.bias, 0)

    def unpatchify(self, patch_tokens, patch_size):
        """
        Reconstruct image tensor from a sequence of patch tokens.
        patch_tokens: (B, num_patches, patch_size^2 * C) -> imgs: (B, C, H, W)
        """
        out_channels = self.out_channels
        grid_h = grid_w = int(patch_tokens.shape[1] ** 0.5)
        assert grid_h * grid_w == patch_tokens.shape[1]

        # Reshape to (B, grid_h, grid_w, patch_h, patch_w, C)
        x = patch_tokens.reshape(
            shape=(patch_tokens.shape[0], grid_h, grid_w, patch_size, patch_size, out_channels)
        )
        # Rearrange dimensions and merge patch dims to reconstruct image
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], out_channels, grid_h * patch_size, grid_h * patch_size))
        return imgs

    def forward(self, noisy_input, timesteps, class_labels, structure_cond):
        """
        LFD forward pass.

        Args:
            noisy_input:    (B, C, H, W), noise-corrupted input image (RGT)
            timesteps:      (B,), current diffusion timesteps
            class_labels:   (B,), class labels for CFG
            structure_cond: (B, 2, H, W), geological structure conditions where:
                            structure_cond[:, 0:1] = fault map
                            structure_cond[:, 1:2] = horizon map

        Returns:
            predicted_clean: (B, C, H, W), model prediction of the clean image
        """
        # ---- Timestep + class conditioning ----
        timestep_emb = self.timestep_embedder(timesteps)        # (B, D)
        class_emb = self.class_label_embedder(class_labels)     # (B, D)
        conditioning_emb = timestep_emb + class_emb             # (B, D), fused conditioning

        # ---- Patch embedding ----
        img_tokens = self.x_embedder(noisy_input)                              # (B, T, D)
        fault_tokens = self.fault_embedder(structure_cond[:, 0:1, :, :])      # (B, T, D)
        horizon_tokens = self.horizon_embedder(structure_cond[:, 1:2, :, :])  # (B, T, D)

        # Input-level condition injection: add fault/horizon tokens to image tokens
        img_tokens = img_tokens + fault_tokens + horizon_tokens

        # Add fixed positional embedding
        img_tokens = img_tokens + self.pos_embed

        # ---- Prepare padded condition tokens for in-context phase ----
        if self.in_context_len > 0:
            # Zero-pad to match sequence length after in-context token insertion
            horizon_context_pad = torch.zeros(
                horizon_tokens.size(0), self.in_context_len, horizon_tokens.size(-1),
                device=horizon_tokens.device, dtype=horizon_tokens.dtype
            )
            # Concatenate: (B, in_context_len + T, D)
            horizon_tokens_with_context_pad = torch.cat([horizon_context_pad, horizon_tokens], dim=1)

            fault_context_pad = torch.zeros(
                fault_tokens.size(0), self.in_context_len, fault_tokens.size(-1),
                device=fault_tokens.device, dtype=fault_tokens.dtype
            )
            fault_tokens_with_context_pad = torch.cat([fault_context_pad, fault_tokens], dim=1)

        # ---- Transformer block loop ----
        for block_idx, block in enumerate(self.blocks):
            # Insert in-context tokens at the designated depth (only once)
            if self.in_context_len > 0 and block_idx == self.in_context_start:
                # Broadcast class embedding to in-context token sequence
                context_tokens = class_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                context_tokens = context_tokens + self.in_context_pos_embed
                # Prepend in-context tokens: (B, in_context_len + T, D)
                img_tokens = torch.cat([context_tokens, img_tokens], dim=1)

            # ---- Select condition tokens for current phase ----
            if self.in_context_len > 0 and block_idx >= self.in_context_start:
                # In-context phase: use padded version to match sequence length
                horizon_inject_tokens = horizon_tokens_with_context_pad
                fault_inject_tokens = fault_tokens_with_context_pad
            else:
                # Pre-context phase: use original patch tokens
                horizon_inject_tokens = horizon_tokens
                fault_inject_tokens = fault_tokens

            # Per-layer residual injection (learnable alpha controls injection strength)
            img_tokens = (
                img_tokens
                + self.horizon_inject_alpha[block_idx] * self.horizon_residual_layers[block_idx](horizon_inject_tokens)
                + self.fault_inject_alpha[block_idx] * self.fault_residual_layers[block_idx](fault_inject_tokens)
            )

            # Select the appropriate RoPE variant
            current_rope = (
                self.rope_without_context if block_idx < self.in_context_start
                else self.rope_with_context
            )
            img_tokens = block(img_tokens, conditioning_emb, current_rope)

        # ---- Remove in-context tokens to restore pure patch token sequence ----
        if self.in_context_len > 0:
            img_tokens = img_tokens[:, self.in_context_len:]

        # ---- Final output layer: project back to image space ----
        img_tokens = self.final_layer(img_tokens, conditioning_emb)
        predicted_clean = self.unpatchify(img_tokens, self.patch_size)

        return predicted_clean


def build_lfd_base_32(**kwargs):
    """Build LFD-B/32 model (Base size, patch_size=32)."""
    return LFD(
        depth=12, hidden_size=768, num_heads=12,
        bottleneck_dim=128, in_context_len=32, in_context_start=4,
        patch_size=32, **kwargs
    )


LFD_models = {
    'LFD-B/32': build_lfd_base_32,
}
