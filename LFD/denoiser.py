import torch
import torch.nn as nn
import torch.nn.functional as F
from model import LFD_models

def _strip_prefix(sd):
    out = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2.replace("module.", "")
        if k2.startswith("net."):
            k2 = k2.replace("net.", "")
        out[k2] = v
    return out

def _adapt_in_chans(weight, in_chans: int):
    """
    weight: [out_c, in_c, k, k]
    """
    out_c, in_c, kh, kw = weight.shape
    if in_c == in_chans:
        return weight

    if in_chans == 1:
        # 3->1: Common practice: mean (or sum) over input channels
        return weight.mean(dim=1, keepdim=True)

    if in_chans < in_c:
        return weight[:, :in_chans, :, :]

    # in_chans > in_c: repeat + scale
    repeat = (in_chans + in_c - 1) // in_c
    w = weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
    w = w * (in_c / in_chans)
    return w

def _adapt_final_linear(weight, bias, patch_size: int, out_chans: int):
    """
    Convert final_layer.linear from (p^2 * C_pre) -> (p^2 * out_chans)
    weight: [p^2*C_pre, H]
    bias  : [p^2*C_pre]
    """
    p2 = patch_size * patch_size
    old_out = weight.shape[0]
    old_c = old_out // p2
    assert old_out % p2 == 0, f"final_layer out dim {old_out} not divisible by p^2={p2}"

    if old_c == out_chans:
        return weight, bias

    # [C_pre, p^2, H]
    W = weight.view(old_c, p2, -1)

    if out_chans == 1:
        Wn = W.mean(dim=0, keepdim=True)          # [1, p^2, H]
    elif out_chans < old_c:
        Wn = W[:out_chans]
    else:
        repeat = (out_chans + old_c - 1) // old_c
        Wn = W.repeat(repeat, 1, 1)[:out_chans]
        Wn = Wn * (old_c / out_chans)

    Wn = Wn.reshape(out_chans * p2, -1)

    bn = None
    if bias is not None:
        b = bias.view(old_c, p2)
        if out_chans == 1:
            bn = b.mean(dim=0, keepdim=True)
        elif out_chans < old_c:
            bn = b[:out_chans]
        else:
            repeat = (out_chans + old_c - 1) // old_c
            bn = b.repeat(repeat, 1)[:out_chans]
            bn = bn * (old_c / out_chans)
        bn = bn.reshape(out_chans * p2)

    return Wn, bn

class DenoiserFH(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = LFD_models[args.model](
            input_size=args.img_size,
            in_channels=args.in_channels,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )


        # ---- load pretrained JiT ----
        if args.pretrained_base:
            ckpt = torch.load(args.pretrained_base, map_location="cpu", weights_only=False)
            sd = ckpt["model"] if "model" in ckpt else ckpt
            sd = _strip_prefix(sd)

            # 1) Class embedding mismatch: do not load
            sd.pop("y_embedder.embedding_table.weight", None)

            # 2) Input channel adaptation: x_embedder.proj1.weight
            k = "x_embedder.proj1.weight"
            if k in sd:
                sd[k] = _adapt_in_chans(sd[k], args.in_channels)

            # 3) Output channel adaptation: final_layer.linear.(weight/bias)
            w_key = "final_layer.linear.weight"
            b_key = "final_layer.linear.bias"
            if w_key in sd:
                W = sd[w_key]
                b = sd.get(b_key, None)
                Wn, bn = _adapt_final_linear(W, b, patch_size=self.net.patch_size, out_chans=args.in_channels)
                sd[w_key] = Wn
                if bn is not None and b_key in sd:
                    sd[b_key] = bn

            # 4) Safest: do "shape filtering" again to avoid other mismatches
            model_sd = self.net.state_dict()
            load_sd = {}
            skipped = []
            for k, v in sd.items():
                if k in model_sd and model_sd[k].shape == v.shape:
                    load_sd[k] = v
                elif k in model_sd:
                    skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

            missing, unexpected = self.net.load_state_dict(load_sd, strict=False)
            print(f"[pretrain] loaded={len(load_sd)} skipped={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}")
            if len(skipped) and len(skipped) < 30:
                print("skipped:", skipped)

        self.in_channels = args.in_channels
        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.label_drop_prob = args.label_drop_prob
        self.struct_drop_prob = 0.1
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None
        
        self.method = args.sampling_method
        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def drop_structure(self, structure):
        """
        structure: [B, C, H, W]
        Replace the structure of some samples with -1 with probability self.struct_drop_prob
        """
        if not self.training:
            return structure
        
        B = structure.shape[0]
        # Generate mask: [B, 1, 1, 1]
        # Places where drop_mask is True need to be dropped
        drop_mask = torch.rand(B, device=structure.device) < self.struct_drop_prob
        drop_mask = drop_mask.view(B, 1, 1, 1)
        
        # Construct empty structure (assuming -1 is your null value, consistent with structure0 in generate)
        null_structure = torch.full_like(structure, -1.0)
        
        # Replace
        out = torch.where(drop_mask, null_structure, structure)
        return out

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x, labels, structure):
        labels_dropped = self.drop_labels(labels) if self.training else labels
        structure_dropped = self.drop_structure(structure) if self.training else structure

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z = t * x + (1 - t) * e
        
        # Forward pass
        x_pred = self.net(z, t.flatten(), labels_dropped, structure_dropped)
        
        # Calculate v_pred (for L2 loss)
        v = (x - z) / (1 - t).clamp_min(self.t_eps)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        # --- Loss Calculation ---
        
        # 1. Pixel/Velocity Loss
        loss_v = (v - v_pred) ** 2
        loss_v = loss_v.mean(dim=(1, 2, 3)).mean()

        # 2. Horizon Loss (assuming structure channel 1 is horizon)
        loss_h = self.horizon_loss_exact(x_pred, structure_dropped[:, 1:2, :, :])

        # 3. Bending energy Loss
        loss_b = self.bending_energy_loss(x_pred, structure_dropped[:, 0:1, :, :], bg=-1.0)
        
        # Total Loss weights
        loss = loss_v + 10 * loss_h + 0.1*loss_b

        return loss, loss_v, loss_h, loss_b
    
    @torch.no_grad()
    def generate(self, labels, structure):
        device = labels.device
        bsz = labels.size(0)
        z = self.noise_scale * torch.randn(bsz, self.in_channels, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=device).view(-1, *([1] * z.ndim)).expand(-1, bsz, -1, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            z = stepper(z, t, t_next, labels, structure)
        # last step euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels, structure)
        return z

    @torch.no_grad()
    def _forward_sample(self, z, t, labels, structure):
        # conditional
        x_cond = self.net(z, t.flatten(), labels, structure)
        v_cond = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        structure0 = torch.full_like(structure, -1.0)
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes), structure0)
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        low, high = self.cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_interval * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, z, t, t_next, labels, structure):
        v_pred = self._forward_sample(z, t, labels, structure)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def _heun_step(self, z, t, t_next, labels, structure):
        v_pred_t = self._forward_sample(z, t, labels, structure)

        z_next_euler = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(z_next_euler, t_next, labels, structure)
        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        z_next = z + (t_next - t) * v_pred
        return z_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)

    def horizon_loss_exact(self, output, horizons, bg_thresh=-0.99, eps=1e-6):
        """
        Masked MSE on horizon points:
        Make output as close as possible to the values given in horizons at the horizon positions.
        Normalize by number of horizon points (average per point), more stable.

        output   : [B, C, H, W]  (x_pred)
        horizons : [B, 1, H, W] or [B, H, W]
                Target values at horizon positions, background < bg_thresh
        """
        assert output.dim() == 4
        B, C, H, W = output.shape

        # [B, H, W]
        if horizons.dim() == 4:
            horizons = horizons[:, 0]

        # Valid mask (horizon points)
        mask = (horizons > bg_thresh)  # [B, H, W]

        # Construct target based on channel count
        if C == 1:
            target = horizons.unsqueeze(1)  # [B,1,H,W]
        else:
            target = horizons.unsqueeze(1).expand(-1, C, -1, -1)  # [B,C,H,W]

        # Expand mask to channel dimension
        mask_f = mask.unsqueeze(1).to(output.dtype)  # [B,1,H,W]
        if C > 1:
            mask_f = mask_f.expand(-1, C, -1, -1)    # [B,C,H,W]

        # masked MSE: sum / (#horizon points)
        sq = (output - target) ** 2
        num = (sq * mask_f).sum(dim=(1, 2, 3))                 # [B]
        den = mask_f.sum(dim=(1, 2, 3)).clamp_min(1.0)         # [B]  Prevent division by zero
        loss_per = num / (den + eps)                           # [B]

        # If some samples have no horizon points: its num=0, den=1 -> loss=0, won't explode
        return loss_per.mean()

    def bending_energy_loss(self, output, fault, bg=-1.0, eps=1e-6, dx=1.0, dy=1.0):
        """
        output: [B,C,H,W]
        fault : [B,1,H,W]  -1 background, +1 fault
        thin-plate energy on non-fault ROI, and DO NOT take stencils across faults.
        """
        x = output
        B, C, H, W = x.shape

        # fault = +1 means fault, -1 means background
        fault_mask = (fault > bg)          # [B,1,H,W]
        valid = (~fault_mask).float()     # 1 on non-fault
        valid_c = valid.expand(-1, C, -1, -1)

        # -------- u_xx (3-point stencil) --------
        v_xx = valid[:, :, :, :-2] * valid[:, :, :, 1:-1] * valid[:, :, :, 2:]   # [B,1,H,W-2]
        v_xx = v_xx.expand(-1, C, -1, -1)                                        # [B,C,H,W-2]
        u_xx = x.new_zeros(B, C, H, W)
        u_xx[:, :, :, 1:-1] = v_xx * (x[:, :, :, 2:] - 2*x[:, :, :, 1:-1] + x[:, :, :, :-2]) / (dx*dx)

        # -------- u_yy (3-point stencil) --------
        v_yy = valid[:, :, :-2, :] * valid[:, :, 1:-1, :] * valid[:, :, 2:, :]   # [B,1,H-2,W]
        v_yy = v_yy.expand(-1, C, -1, -1)                                        # [B,C,H-2,W]
        u_yy = x.new_zeros(B, C, H, W)
        u_yy[:, :, 1:-1, :] = v_yy * (x[:, :, 2:, :] - 2*x[:, :, 1:-1, :] + x[:, :, :-2, :]) / (dy*dy)

        # -------- u_xy (2x2 stencil) --------
        v_xy = (
            valid[:, :, :-1, :-1] * valid[:, :, :-1, 1:] *
            valid[:, :,  1:, :-1] * valid[:, :,  1:, 1:]
        )  # [B,1,H-1,W-1]
        v_xy = v_xy.expand(-1, C, -1, -1)  # [B,C,H-1,W-1]

        u_xy = x.new_zeros(B, C, H, W)
        u_xy[:, :, :H-1, :W-1] = v_xy * (
            x[:, :, 1:, 1:] - x[:, :, 1:, :-1] - x[:, :, :-1, 1:] + x[:, :, :-1, :-1]
        ) / (dx*dy)

        # thin-plate energy only on non-fault
        energy = (u_xx**2 + 2.0*(u_xy**2) + u_yy**2) * valid_c
        return energy.sum() / (valid_c.sum() + eps)
