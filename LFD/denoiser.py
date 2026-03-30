import torch
import torch.nn as nn
import torch.nn.functional as F
from model import LFD_models


# ============================================================
# Utility functions: pretrained weight adaptation
# ============================================================

def strip_module_prefix(state_dict):
    """
    Remove 'module.' or 'net.' prefixes from state_dict keys.
    Ensures compatibility with checkpoints saved through DistributedDataParallel
    or DenoiserFH wrappers.
    """
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key.replace("module.", "")
        if new_key.startswith("net."):
            new_key = new_key.replace("net.", "")
        cleaned_state_dict[new_key] = value
    return cleaned_state_dict


def adapt_input_channel_weight(conv_weight, target_in_channels: int):
    """
    Adapt a convolution weight tensor to a different number of input channels.
    conv_weight shape: [out_c, in_c, kH, kW]

    Adaptation rules:
    - in_c == target:  return as-is
    - target == 1:     average across original channels (common 3->1 approach)
    - target < in_c:   truncate to first target channels
    - target > in_c:   repeat and rescale weights
    """
    out_c, in_c, kh, kw = conv_weight.shape
    if in_c == target_in_channels:
        return conv_weight

    if target_in_channels == 1:
        # Multi-channel -> single-channel: average over input channel dimension
        return conv_weight.mean(dim=1, keepdim=True)

    if target_in_channels < in_c:
        # Truncate to first target_in_channels channels
        return conv_weight[:, :target_in_channels, :, :]

    # target_in_channels > in_c: repeat and rescale to preserve output magnitude
    repeat_times = (target_in_channels + in_c - 1) // in_c
    expanded_weight = conv_weight.repeat(1, repeat_times, 1, 1)[:, :target_in_channels, :, :]
    expanded_weight = expanded_weight * (in_c / target_in_channels)
    return expanded_weight


def adapt_final_linear_weight(weight, bias, patch_size: int, target_out_channels: int):
    """
    Adapt the final output linear layer weight from (patch^2 * old_C, hidden)
    to (patch^2 * target_C, hidden).

    weight shape: [patch^2 * old_C, hidden_dim]
    bias   shape: [patch^2 * old_C]

    Adaptation rules mirror those in adapt_input_channel_weight.
    """
    patch_area = patch_size * patch_size
    old_out_dim = weight.shape[0]
    old_channels = old_out_dim // patch_area
    assert old_out_dim % patch_area == 0, \
        f"final_layer output dim {old_out_dim} is not divisible by patch^2={patch_area}"

    if old_channels == target_out_channels:
        return weight, bias

    # Reshape to [old_C, patch^2, hidden_dim] for channel-wise operations
    weight_reshaped = weight.view(old_channels, patch_area, -1)

    if target_out_channels == 1:
        adapted_weight = weight_reshaped.mean(dim=0, keepdim=True)
    elif target_out_channels < old_channels:
        adapted_weight = weight_reshaped[:target_out_channels]
    else:
        repeat_times = (target_out_channels + old_channels - 1) // old_channels
        adapted_weight = weight_reshaped.repeat(repeat_times, 1, 1)[:target_out_channels]
        adapted_weight = adapted_weight * (old_channels / target_out_channels)

    # Restore to [target_C * patch^2, hidden_dim]
    adapted_weight = adapted_weight.reshape(target_out_channels * patch_area, -1)

    adapted_bias = None
    if bias is not None:
        bias_reshaped = bias.view(old_channels, patch_area)
        if target_out_channels == 1:
            adapted_bias = bias_reshaped.mean(dim=0, keepdim=True)
        elif target_out_channels < old_channels:
            adapted_bias = bias_reshaped[:target_out_channels]
        else:
            repeat_times = (target_out_channels + old_channels - 1) // old_channels
            adapted_bias = bias_reshaped.repeat(repeat_times, 1)[:target_out_channels]
            adapted_bias = adapted_bias * (old_channels / target_out_channels)
        adapted_bias = adapted_bias.reshape(target_out_channels * patch_area)

    return adapted_weight, adapted_bias


# ============================================================
# Main denoiser model
# ============================================================

class DenoiserFH(nn.Module):
    """
    LFD Denoiser conditioned on Fault + Horizon (FH) structure maps.

    Wraps the LFD backbone and provides:
    - Label and structure dropout for Classifier-Free Guidance (CFG) training
    - Flow Matching noise injection and velocity field loss
    - Horizon constraint loss (sparse RGT supervision at labeled horizon positions)
    - Thin-plate bending energy regularization loss
    - ODE sampling (Euler / Heun)
    - EMA parameter tracking and updates
    """

    def __init__(self, args):
        super().__init__()

        # ---- Build LFD backbone ----
        self.net = LFD_models[args.model](
            input_size=args.img_size,
            in_channels=args.in_channels,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )

        # ---- Load pretrained weights if specified ----
        if args.pretrained_base:
            raw_ckpt = torch.load(args.pretrained_base, map_location="cpu", weights_only=False)
            pretrained_sd = raw_ckpt["model"] if "model" in raw_ckpt else raw_ckpt
            pretrained_sd = strip_module_prefix(pretrained_sd)

            # 1) Skip class embedding (size mismatch is expected)
            pretrained_sd.pop("y_embedder.embedding_table.weight", None)

            # 2) Adapt input convolution channel count (x_embedder.patch_proj.weight)
            input_proj_key = "x_embedder.patch_proj.weight"
            if input_proj_key in pretrained_sd:
                pretrained_sd[input_proj_key] = adapt_input_channel_weight(
                    pretrained_sd[input_proj_key], args.in_channels
                )

            # 3) Adapt output linear channel count (final_layer.pixel_proj)
            out_weight_key = "final_layer.pixel_proj.weight"
            out_bias_key = "final_layer.pixel_proj.bias"
            if out_weight_key in pretrained_sd:
                out_weight = pretrained_sd[out_weight_key]
                out_bias = pretrained_sd.get(out_bias_key, None)
                adapted_w, adapted_b = adapt_final_linear_weight(
                    out_weight, out_bias,
                    patch_size=self.net.patch_size,
                    target_out_channels=args.in_channels
                )
                pretrained_sd[out_weight_key] = adapted_w
                if adapted_b is not None and out_bias_key in pretrained_sd:
                    pretrained_sd[out_bias_key] = adapted_b

            # 4) Shape filter: skip any remaining mismatched parameters
            current_model_sd = self.net.state_dict()
            filtered_sd = {}
            shape_mismatch_keys = []
            for key, value in pretrained_sd.items():
                if key in current_model_sd and current_model_sd[key].shape == value.shape:
                    filtered_sd[key] = value
                elif key in current_model_sd:
                    shape_mismatch_keys.append((key, tuple(value.shape), tuple(current_model_sd[key].shape)))

            missing_keys, unexpected_keys = self.net.load_state_dict(filtered_sd, strict=False)
            print(
                f"[pretrain] loaded={len(filtered_sd)} "
                f"shape_mismatch_skipped={len(shape_mismatch_keys)} "
                f"missing={len(missing_keys)} "
                f"unexpected={len(unexpected_keys)}"
            )
            if 0 < len(shape_mismatch_keys) < 30:
                print("Shape-mismatched keys:", shape_mismatch_keys)

        # ---- Model hyperparameters ----
        self.in_channels = args.in_channels
        self.img_size = args.img_size
        self.num_classes = args.class_num

        # ---- Training dropout probabilities ----
        self.label_drop_prob = args.label_drop_prob   # CFG label dropout probability
        self.structure_drop_prob = 0.1                # geological structure condition dropout

        # ---- Flow Matching noise schedule parameters ----
        self.noise_mean = args.P_mean    # logit-normal distribution mean for timestep sampling
        self.noise_std = args.P_std      # logit-normal distribution std
        self.time_eps = args.t_eps       # lower bound for timestep (avoids division by zero)
        self.noise_scale = args.noise_scale  # initial noise scaling factor

        # ---- EMA parameters (exponential moving average, used for higher-quality sampling) ----
        self.ema_decay1 = args.ema_decay1    # first EMA decay rate (used by default at sampling)
        self.ema_decay2 = args.ema_decay2    # second EMA decay rate (backup)
        self.ema_params1 = None
        self.ema_params2 = None

        # ---- ODE sampling configuration ----
        self.sampling_method = args.sampling_method          # 'euler' or 'heun'
        self.num_sampling_steps = args.num_sampling_steps    # number of ODE integration steps
        self.cfg_scale = args.cfg                            # Classifier-Free Guidance scale
        self.cfg_interval = (args.interval_min, args.interval_max)  # timestep range for CFG

    # ----------------------------------------------------------
    # Training-time dropout helpers
    # ----------------------------------------------------------

    def randomly_drop_class_labels(self, class_labels):
        """
        Replace class labels with the null class (num_classes) at probability
        label_drop_prob, enabling Classifier-Free Guidance (CFG) training.

        Args:
            class_labels: (B,) integer label tensor

        Returns:
            (B,) tensor with randomly dropped labels
        """
        drop_mask = torch.rand(class_labels.shape[0], device=class_labels.device) < self.label_drop_prob
        dropped_labels = torch.where(
            drop_mask,
            torch.full_like(class_labels, self.num_classes),
            class_labels
        )
        return dropped_labels

    def randomly_drop_structure_condition(self, structure):
        """
        Replace the entire geological structure condition with -1 (null condition)
        at probability structure_drop_prob during training (for CFG).
        No-op at inference time.

        Args:
            structure: (B, C, H, W) condition tensor

        Returns:
            (B, C, H, W) tensor with randomly zeroed-out conditions
        """
        if not self.training:
            return structure

        batch_size = structure.shape[0]
        # Shape (B, 1, 1, 1) mask that broadcasts over spatial dimensions
        drop_mask = (
            torch.rand(batch_size, device=structure.device) < self.structure_drop_prob
        ).view(batch_size, 1, 1, 1)

        # Fill dropped conditions with -1 (consistent with normalized background value)
        null_structure = torch.full_like(structure, -1.0)
        return torch.where(drop_mask, null_structure, structure)

    def sample_flow_timestep(self, batch_size: int, device=None):
        """
        Sample flow matching timesteps t in (0, 1) from a logit-normal distribution.
        Formula: t = sigmoid(z), z ~ N(noise_mean, noise_std^2)

        Args:
            batch_size: number of timesteps to sample

        Returns:
            Tensor of shape (batch_size,)
        """
        z = torch.randn(batch_size, device=device) * self.noise_std + self.noise_mean
        return torch.sigmoid(z)

    # ----------------------------------------------------------
    # Training forward pass
    # ----------------------------------------------------------

    def forward(self, clean_target, class_labels, structure_cond):
        """
        Training forward pass: inject noise -> network prediction -> compute losses.

        Args:
            clean_target:   (B, C, H, W), clean target image (e.g., RGT)
            class_labels:   (B,), class labels
            structure_cond: (B, C_cond, H, W), geological structure conditions (fault + horizon)

        Returns:
            total_loss:    scalar, weighted sum of all losses
            velocity_loss: scalar, flow matching velocity field loss
            horizon_loss:  scalar, horizon constraint loss
            bending_loss:  scalar, bending energy regularization loss
        """
        # Apply training-time dropout to labels and structure conditions (CFG)
        labels_for_forward = self.randomly_drop_class_labels(class_labels) if self.training else class_labels
        structure_for_forward = self.randomly_drop_structure_condition(structure_cond) if self.training else structure_cond

        # Sample timestep t and expand to (B, 1, 1, 1) for broadcasting
        t = self.sample_flow_timestep(clean_target.size(0), device=clean_target.device)
        t = t.view(-1, *([1] * (clean_target.ndim - 1)))

        # Flow Matching forward process: z_t = t * x_0 + (1 - t) * noise
        noise = torch.randn_like(clean_target) * self.noise_scale
        noisy_input = t * clean_target + (1 - t) * noise

        # Network prediction
        predicted_clean = self.net(noisy_input, t.flatten(), labels_for_forward, structure_for_forward)

        # ---- Velocity Field Loss ----
        # True velocity:  v = (x_0 - z_t) / (1 - t)
        # Pred velocity:  v_pred = (x_pred - z_t) / (1 - t)
        true_velocity = (clean_target - noisy_input) / (1 - t).clamp_min(self.time_eps)
        pred_velocity = (predicted_clean - noisy_input) / (1 - t).clamp_min(self.time_eps)
        velocity_loss = ((true_velocity - pred_velocity) ** 2).mean(dim=(1, 2, 3)).mean()

        # ---- Horizon Constraint Loss ----
        # Uses channel index 1 from structure_cond as the horizon supervision signal
        horizon_loss = self.compute_horizon_loss(
            predicted_clean, structure_for_forward[:, 1:2, :, :]
        )

        # ---- Bending Energy Regularization Loss ----
        # Uses channel index 0 from structure_cond as the fault mask
        bending_loss = self.compute_bending_energy_loss(
            predicted_clean, structure_for_forward[:, 0:1, :, :], bg=-1.0
        )

        # ---- Weighted total loss ----
        total_loss = velocity_loss + 10.0 * horizon_loss + 0.1 * bending_loss

        return total_loss, velocity_loss, horizon_loss, bending_loss

    # ----------------------------------------------------------
    # ODE Sampling (inference)
    # ----------------------------------------------------------

    @torch.no_grad()
    def generate(self, class_labels, structure_cond):
        """
        Conditional generation: integrate ODE from random noise to a clean image.

        Args:
            class_labels:   (B,)
            structure_cond: (B, C_cond, H, W)

        Returns:
            generated_images: (B, C, H, W)
        """
        device = class_labels.device
        batch_size = class_labels.size(0)

        # Initial noise z_0 ~ N(0, noise_scale^2)
        z = self.noise_scale * torch.randn(
            batch_size, self.in_channels, self.img_size, self.img_size, device=device
        )

        # Uniform timestep sequence [0, 1], expanded to (num_steps+1, B, C, H, W)
        timestep_seq = torch.linspace(0.0, 1.0, self.num_sampling_steps + 1, device=device)
        timestep_seq = timestep_seq.view(-1, *([1] * z.ndim)).expand(-1, batch_size, -1, -1, -1)

        # Select ODE integration method
        if self.sampling_method == "euler":
            ode_step_fn = self._euler_step
        elif self.sampling_method == "heun":
            ode_step_fn = self._heun_step
        else:
            raise NotImplementedError(f"Unsupported sampling method: {self.sampling_method}")

        # ODE integration (last step always uses Euler to avoid double inference cost of Heun)
        for step_idx in range(self.num_sampling_steps - 1):
            t_current = timestep_seq[step_idx]
            t_next = timestep_seq[step_idx + 1]
            z = ode_step_fn(z, t_current, t_next, class_labels, structure_cond)

        # Final step: force Euler
        z = self._euler_step(z, timestep_seq[-2], timestep_seq[-1], class_labels, structure_cond)
        return z

    @torch.no_grad()
    def _compute_cfg_velocity(self, z, t, class_labels, structure_cond):
        """
        Compute CFG-guided velocity field estimate.
        Runs both conditional and unconditional forward passes and interpolates.

        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)

        Args:
            z:              (B, C, H, W), current noisy state
            t:              (B, 1, 1, 1), current timestep
            class_labels:   (B,)
            structure_cond: (B, C_cond, H, W)

        Returns:
            cfg_velocity: (B, C, H, W)
        """
        # Conditional prediction
        cond_pred = self.net(z, t.flatten(), class_labels, structure_cond)
        v_cond = (cond_pred - z) / (1.0 - t).clamp_min(self.time_eps)

        # Unconditional prediction (null class + all-(-1) structure)
        null_structure = torch.full_like(structure_cond, -1.0)
        uncond_pred = self.net(
            z, t.flatten(),
            torch.full_like(class_labels, self.num_classes),
            null_structure
        )
        v_uncond = (uncond_pred - z) / (1.0 - t).clamp_min(self.time_eps)

        # Apply CFG only within the specified timestep interval
        cfg_low, cfg_high = self.cfg_interval
        in_cfg_interval = (t < cfg_high) & ((cfg_low == 0) | (t > cfg_low))
        # Use cfg_scale within interval, 1.0 outside (pure conditional)
        effective_cfg_scale = torch.where(in_cfg_interval, self.cfg_scale, 1.0)

        return v_uncond + effective_cfg_scale * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t_current, t_next, class_labels, structure_cond):
        """
        Euler ODE integration step:
        z_{t+1} = z_t + (t_{t+1} - t_t) * v(z_t, t_t)
        """
        velocity = self._compute_cfg_velocity(z, t_current, class_labels, structure_cond)
        z_next = z + (t_next - t_current) * velocity
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t_current, t_next, class_labels, structure_cond):
        """
        Heun (2nd-order Runge-Kutta) ODE integration step:
        1. Predict z_{t+1}^* via Euler step
        2. Evaluate velocity at the predicted point
        3. Integrate using the average of both velocity estimates (trapezoidal rule)
        """
        # First velocity estimate at current point
        v_at_current = self._compute_cfg_velocity(z, t_current, class_labels, structure_cond)
        z_euler_pred = z + (t_next - t_current) * v_at_current

        # Second velocity estimate at predicted point
        v_at_next = self._compute_cfg_velocity(z_euler_pred, t_next, class_labels, structure_cond)

        # Trapezoidal rule: average both velocity estimates
        v_average = 0.5 * (v_at_current + v_at_next)
        z_next = z + (t_next - t_current) * v_average
        return z_next

    # ----------------------------------------------------------
    # EMA updates
    # ----------------------------------------------------------

    @torch.no_grad()
    def update_ema(self):
        """
        Update both EMA parameter copies.
        Formula: ema_param = decay * ema_param + (1 - decay) * current_param
        """
        current_params = list(self.parameters())
        for ema_param, current_param in zip(self.ema_params1, current_params):
            ema_param.detach().mul_(self.ema_decay1).add_(current_param, alpha=1 - self.ema_decay1)
        for ema_param, current_param in zip(self.ema_params2, current_params):
            ema_param.detach().mul_(self.ema_decay2).add_(current_param, alpha=1 - self.ema_decay2)

    # ----------------------------------------------------------
    # Loss functions
    # ----------------------------------------------------------

    def compute_horizon_loss(self, predicted, horizon_target, bg_threshold=-0.99, eps=1e-6):
        """
        Masked MSE loss at horizon annotation positions (Horizon Loss).
        Only computes loss where horizon_target > bg_threshold.
        Loss is normalized by the number of valid horizon points to handle
        sparse annotations gracefully.

        Args:
            predicted:      (B, C, H, W), model-predicted denoised image
            horizon_target: (B, 1, H, W) or (B, H, W), horizon target values
                            (background values < bg_threshold)
            bg_threshold:   threshold for distinguishing valid horizon pixels from background
            eps:            small constant to prevent division by zero

        Returns:
            mean_horizon_loss: scalar
        """
        assert predicted.dim() == 4
        B, C, H, W = predicted.shape

        # Ensure horizon_target is (B, H, W)
        if horizon_target.dim() == 4:
            horizon_target = horizon_target[:, 0]

        # Valid horizon pixel mask
        valid_horizon_mask = (horizon_target > bg_threshold)  # (B, H, W)

        # Build target tensor matching predicted shape
        if C == 1:
            target_expanded = horizon_target.unsqueeze(1)                         # (B, 1, H, W)
        else:
            target_expanded = horizon_target.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, H, W)

        # Expand mask to channel dimension
        mask_float = valid_horizon_mask.unsqueeze(1).to(predicted.dtype)          # (B, 1, H, W)
        if C > 1:
            mask_float = mask_float.expand(-1, C, -1, -1)                        # (B, C, H, W)

        # Compute per-sample MSE at valid positions, normalized by number of valid points
        squared_error = (predicted - target_expanded) ** 2
        loss_numerator = (squared_error * mask_float).sum(dim=(1, 2, 3))          # (B,)
        loss_denominator = mask_float.sum(dim=(1, 2, 3)).clamp_min(1.0)           # (B,)
        per_sample_loss = loss_numerator / (loss_denominator + eps)               # (B,)

        return per_sample_loss.mean()

    def compute_bending_energy_loss(self, predicted, fault_mask, bg=-1.0, eps=1e-6, dx=1.0, dy=1.0):
        """
        Thin-Plate Bending Energy regularization loss.
        Minimizes second-order derivatives of the predicted RGT in non-fault regions,
        encouraging smooth and continuous stratigraphy.
        Fault pixels and cross-fault finite differences are masked out.

        Bending energy: E = integral( u_xx^2 + 2*u_xy^2 + u_yy^2 ) dOmega

        Args:
            predicted:  (B, C, H, W), predicted RGT image
            fault_mask: (B, 1, H, W), fault mask (-1 for background, +1 for fault)
            bg:         background value used to identify non-fault regions
            eps:        small constant to prevent division by zero
            dx, dy:     spatial grid spacing (default 1.0)

        Returns:
            mean_bending_energy: scalar
        """
        x = predicted
        B, C, H, W = x.shape

        # Non-fault region mask (valid = 1 means non-fault, participates in loss)
        is_fault = (fault_mask > bg)             # (B, 1, H, W), True = fault pixel
        valid_mask = (~is_fault).float()         # (B, 1, H, W), 1.0 = non-fault pixel
        valid_mask_expanded = valid_mask.expand(-1, C, -1, -1)  # (B, C, H, W)

        # ---- u_xx: horizontal second derivative (central diff, masked at fault crossings) ----
        # All three neighbors must be non-fault for the derivative to be valid
        valid_xx = (
            valid_mask[:, :, :, :-2] * valid_mask[:, :, :, 1:-1] * valid_mask[:, :, :, 2:]
        )  # (B, 1, H, W-2)
        valid_xx = valid_xx.expand(-1, C, -1, -1)  # (B, C, H, W-2)

        u_xx = x.new_zeros(B, C, H, W)
        u_xx[:, :, :, 1:-1] = valid_xx * (
            x[:, :, :, 2:] - 2 * x[:, :, :, 1:-1] + x[:, :, :, :-2]
        ) / (dx * dx)

        # ---- u_yy: vertical second derivative (central diff, masked at fault crossings) ----
        valid_yy = (
            valid_mask[:, :, :-2, :] * valid_mask[:, :, 1:-1, :] * valid_mask[:, :, 2:, :]
        )  # (B, 1, H-2, W)
        valid_yy = valid_yy.expand(-1, C, -1, -1)  # (B, C, H-2, W)

        u_yy = x.new_zeros(B, C, H, W)
        u_yy[:, :, 1:-1, :] = valid_yy * (
            x[:, :, 2:, :] - 2 * x[:, :, 1:-1, :] + x[:, :, :-2, :]
        ) / (dy * dy)

        # ---- u_xy: mixed partial derivative (2x2 stencil, masked at fault crossings) ----
        valid_xy = (
            valid_mask[:, :, :-1, :-1] * valid_mask[:, :, :-1, 1:]
            * valid_mask[:, :, 1:, :-1] * valid_mask[:, :, 1:, 1:]
        )  # (B, 1, H-1, W-1)
        valid_xy = valid_xy.expand(-1, C, -1, -1)  # (B, C, H-1, W-1)

        u_xy = x.new_zeros(B, C, H, W)
        u_xy[:, :, :H-1, :W-1] = valid_xy * (
            x[:, :, 1:, 1:] - x[:, :, 1:, :-1] - x[:, :, :-1, 1:] + x[:, :, :-1, :-1]
        ) / (dx * dy)

        # Accumulate bending energy only in non-fault regions
        bending_energy = (u_xx ** 2 + 2.0 * (u_xy ** 2) + u_yy ** 2) * valid_mask_expanded
        return bending_energy.sum() / (valid_mask_expanded.sum() + eps)
