# -*- coding: utf-8 -*-

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from denoiser import DenoiserFH as Denoiser
from typing import Optional
import math
import scipy.ndimage
from engine import getStrataColors,getFxColor

def replace_fault_values_with_nearest(pred_np: np.ndarray,
                                      fault_np: np.ndarray,
                                      thr: float = -0.999,
                                      dilate: int = 0) -> np.ndarray:
    """
    Replace values in pred at fault locations with the nearest neighbor (nearest non-fault pixel).
    pred_np : [B, C, H, W]
    fault_np: [H, W]  (your fx channel, background around -1, fault +1)
    thr     : Fault threshold (consistent with your visualization)
    dilate  : Optional, dilate the fault mask by a few pixels to avoid edge residuals (0 means no dilation)
    """
    assert pred_np.ndim == 4, f"pred_np should be [B,C,H,W], got {pred_np.shape}"
    assert fault_np.ndim == 2, f"fault_np should be [H,W], got {fault_np.shape}"

    # True = fault region
    fault_mask = (fault_np > thr)
    if dilate > 0:
        fault_mask = scipy.ndimage.binary_dilation(fault_mask, iterations=dilate)

    # If no fault pixels, return directly
    if not np.any(fault_mask):
        return pred_np

    # distance_transform_edt: computes distance from "non-zero" elements to the nearest "zero" element
    # So here let fault_mask=1 (fault), non-fault=0, return coordinates of the nearest 0 (non-fault) for each pixel
    _, (iy, ix) = scipy.ndimage.distance_transform_edt(
        fault_mask.astype(np.uint8),
        return_indices=True
    )

    out = pred_np.copy()
    B, C, H, W = out.shape
    for b in range(B):
        for c in range(C):
            img = out[b, c]
            # Replace fault locations with nearest non-fault pixel values
            img[fault_mask] = img[iy[fault_mask], ix[fault_mask]]
            out[b, c] = img
    return out

def smooth_prediction(pred_np, sigma=(0.5, 1.0)):
    """
    sigma: (sigma_y, sigma_x)
    """
    for b in range(pred_np.shape[0]):
        for c in range(pred_np.shape[1]):
            pred_np[b, c] = scipy.ndimage.gaussian_filter(
                pred_np[b, c],
                sigma=sigma,     # anisotropic!
                mode="nearest"
            )
    return pred_np

def normalize_cond_args(raw_cond):
    """
    Normalize --cond argument to list[str], compatible with:
        --cond fx hrz
        --cond "fx,hrz"
        CONDITION="fx","hrz"; --cond ${CONDITION}
    """
    if isinstance(raw_cond, list):
        if len(raw_cond) == 1:
            s = raw_cond[0]
        else:
            return raw_cond
    else:
        s = raw_cond

    s = s.strip()

    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]

    s = s.replace('"', '').replace("'", '')

    tokens = []
    for tok in s.replace(',', ' ').split():
        tok = tok.strip()
        if tok:
            tokens.append(tok)

    return tokens

def normalize_array(array: np.ndarray, key: Optional[str] = None) -> np.ndarray:
    """
    - fx/fault -> non-zero becomes 1, others -1
    - rgt      -> linearly normalize to [-1,1]
    - hrz/horiz:
        * Assume same horizon uses same label
        * Sort by average depth, map shallow->deep to [-1,1]
        * Same horizon gets same value
    Other keys are kept as is
    """
    array = array.astype(np.float32, copy=False)

    # 1) Fault
    if key in ["fx", "fault"]:
        out = np.full_like(array, -1.0, dtype=np.float32)
        out[array != 0] = 1.0
        return out

    # 2) RGT: Standard min-max normalization
    if key == "rgt":
        min_val = float(array.min())
        max_val = float(array.max())
        if max_val - min_val > 1e-6:
            array = (array - min_val) / (max_val - min_val) * 2.0 - 1.0
        else:
            array = np.full_like(array, -1.0, dtype=np.float32)
        return array

    # 3) HRZ: Same value for same horizon, deeper is closer to 1
    if key in ("hrz", "horiz"):
        depth_axis = 0  # If array is [H,W], depth is H axis

        labels = np.unique(array)
        labels = labels[labels != 0]
        if labels.size == 0:
            return np.full_like(array, -1.0, dtype=np.float32)

        depth_indices = np.indices(array.shape, dtype=np.int32)[depth_axis].astype(np.float32)

        label_depths = []
        for lab in labels:
            z_mean = depth_indices[array == lab].mean()
            # z_med = np.median(depth_indices[array == lab])
            label_depths.append((lab, z_mean))
        print(f"label_depths: {label_depths}")
        # Sorting is not strictly necessary but harmless
        label_depths.sort(key=lambda x: x[1])

        depth_len = array.shape[depth_axis]
        z_max = float(depth_len - 1)  # e.g. 512 -> 511

        mapping = {}
        for lab, z_mean in label_depths:
            v = -1.0 + 2.0 * (float(z_mean) / max(z_max, 1.0))
            mapping[lab] = np.float32(np.clip(v, -1.0, 1.0))

        out = np.full_like(array, -1.0, dtype=np.float32)
        for lab, val in mapping.items():
            out[array == lab] = val

        return out

    # 4) Default: no change
    return array

def load_cond_from_npz(path: str, cond_keys: list[str]) -> torch.Tensor:
    """
    Load multiple cond channels from a single npz file and normalize.
    Return size: [1, C_cond, H, W]
    """
    data = np.load(path)
    cond_list = []
    for k in cond_keys:
        if k not in data:
            raise KeyError(f"Key '{k}' not found in npz file: {path}")
        arr = data[k]                        # [H, W]
        print(arr.shape)
        arr = normalize_array(arr, key=k)
        t = torch.from_numpy(arr).float().unsqueeze(0)   # [1, H, W]
        cond_list.append(t)

    cond = torch.cat(cond_list, dim=0)      # [C, H, W]
    cond = cond.unsqueeze(0)                # [1, C, H, W]
    cond = torch.nn.functional.interpolate(
        cond, size=(512, 512), mode="nearest"
    )
    return cond


def load_cond_from_npy(path: str) -> torch.Tensor:
    """
    Load cond from npy file:
    - If [H,W] -> Treat as single channel, become [1,1,H,W]
    - If [C,H,W] -> Become [1,C,H,W]
    """
    arr = np.load(path).astype(np.float32, copy=False)

    if arr.ndim == 2:
        # [H,W] -> [1,1,H,W]
        arr = arr[None, None, :, :]
    elif arr.ndim == 3:
        # [C,H,W] -> [1,C,H,W]
        arr = arr[None, :, :, :]
    else:
        raise ValueError(
            f"Unsupported npy shape {arr.shape}, expect [H,W] or [C,H,W]."
        )

    cond = torch.from_numpy(arr).float()
    return cond


def build_model(args, device: torch.device) -> torch.nn.Module:
    """
    Build DenoiserFH and load checkpoint (prefer EMA1 weights).
    """
    model = Denoiser(args).to(device)

    ckpt_path = args.ckpt
    if os.path.isdir(ckpt_path):
        ckpt_path = os.path.join(ckpt_path, "checkpoint-last.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_ema1" in ckpt:
        state_dict = ckpt["model_ema1"]
        print("[INFO] Using EMA1 weights.")
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        print("[INFO] Using 'model' weights (no EMA found).")
    else:
        # Direct state_dict
        state_dict = ckpt
        print("[WARN] Checkpoint format not standard, use as plain state_dict.")

    # Strict loading
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] Missing keys in state_dict:", len(missing))
    if unexpected:
        print("[WARN] Unexpected keys in state_dict:", len(unexpected))
    print(f"[INFO] LFD Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    model.eval()
    return model


# -------------------------
# Main
# -------------------------
def get_args():
    parser = argparse.ArgumentParser("Single-file inference for LFD")

    # Required
    parser.add_argument("--input", type=str,
                        default="xxx.npz",)
    parser.add_argument("--ckpt", type=str,
                        default="./result/lfd_rgt_10abshrzloss_0.1bending")

    # Optional output
    parser.add_argument("--out", type=str, default="./output/xxx",)

    # Model architecture and sampling hyperparameters (default consistent with training script)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples to generate with the same condition"
    )
    parser.add_argument("--grid_cols", type=int, default=5,
                        help="Number of columns in the summary grid")
    parser.add_argument("--model", type=str, default="LFD-B/32")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--class_num", type=int, default=1)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--cond_in_ch", type=int, default=2)

    parser.add_argument("--P_mean", type=float, default=-1.0)
    parser.add_argument("--P_std", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--t_eps", type=float, default=5e-2)
    parser.add_argument("--label_drop_prob", type=float, default=0.0)

    parser.add_argument("--ema_decay1", type=float, default=0.9999)
    parser.add_argument("--ema_decay2", type=float, default=0.9996)

    parser.add_argument("--sampling_method", type=str, default="heun")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--interval_min", type=float, default=0.1)
    parser.add_argument("--interval_max", type=float, default=1.0)

    # Condition keys (meaningful for npz only)
    parser.add_argument("--cond", nargs="+", default=["fault", "horiz"],
                        help="Condition keys in npz file, e.g., fx hrz")
    
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for generation")

    # 设备
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'")
    parser.add_argument('--pretrained_base', default='', type=str,
                        help='path to jit_all checkpoint-last.pth')
    args = parser.parse_args()
    args.cond = normalize_cond_args(args.cond)
    return args


def main():
    args = get_args()

    # 设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    model = build_model(args, device)

    labels = torch.zeros(1, dtype=torch.long, device=device)
    input_path = args.input
    ext = Path(input_path).suffix.lower()
    if ext == ".npz":
        print(f"[INFO] Loading npz: {input_path} with cond keys = {args.cond}")
        cond = load_cond_from_npz(input_path, args.cond)
    elif ext == ".npy":
        print(f"[INFO] Loading npy: {input_path}")
        cond = load_cond_from_npy(input_path)
    else:
        raise ValueError(f"Unsupported input extension '{ext}', only .npz/.npy are supported.")

    cond = cond.to(device, non_blocking=True)

    base_out = Path(args.out)
    out_dir =  os.path.join(base_out, args.ckpt.split("/")[-1] , "cfg_"+str(args.cfg))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output dir: {out_dir}")

    print(f"[INFO] Start generation: {args.num_samples} samples")

    fx_np = cond[0, 0].detach().cpu().numpy()
    hrz_np = cond[0, 1].detach().cpu().numpy()

    print(f"min/max fx: {fx_np.min()}/{fx_np.max()}")
    print(f"min/max hrz: {hrz_np.min()}/{hrz_np.max()}")

    pred_imgs = []  # store all pred images for summary grid

    hrzmap = getStrataColors()
    hrzmap.set_bad('#f0f5f9')

    fxmap = getFxColor()
    fxmap.set_bad('#f0f5f9')


    hrz_thick = scipy.ndimage.maximum_filter(hrz_np, size=2)
    fx_thick = scipy.ndimage.maximum_filter(fx_np, size=2)
    fx_masked = np.ma.masked_where(fx_thick <= -0.999, fx_thick)  
    hrz_masked = np.ma.masked_where(hrz_thick <= -0.999, hrz_thick)

    total_gen_time = 0.0
    batch_size = args.batch_size
    num_batches = math.ceil(args.num_samples / batch_size)

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, args.num_samples)
            current_bs = end_idx - start_idx
            
            print(f"[INFO] Sampling batch {batch_idx+1}/{num_batches} (samples {start_idx} to {end_idx-1})")
            
            # Prepare batch inputs
            batch_labels = torch.zeros(current_bs, dtype=torch.long, device=device)
            batch_cond = cond.repeat(current_bs, 1, 1, 1)

            t0 = time.time()
            preds = model.generate(batch_labels, batch_cond)      # [B, C, H, W]
            dt = time.time() - t0
            total_gen_time += dt
            print(f"[INFO] Batch generation time: {dt:.4f}s")
            
            preds_np = preds.detach().cpu().numpy()

            preds_np = smooth_prediction(preds_np,sigma=1)

            preds_np = replace_fault_values_with_nearest(
                preds_np,
                fault_np=fx_thick,
                thr=-0.999,
                dilate=1
            )
            
            for b_local in range(current_bs):
                i = start_idx + b_local
                pred_np = preds_np[b_local:b_local+1] # [1, C, H, W]
                
                npy_path = os.path.join(out_dir, f"sample_{i:03d}.npy")
                np.save(npy_path, pred_np)
                pred_img = pred_np[0, 0]
                pred_imgs.append(pred_img)

                plt.figure(figsize=(4, 4))
                plt.imshow(fx_masked, cmap=fxmap, vmin=-1.0, vmax=1.0)
                plt.axis("off")
                plt.savefig(os.path.join(out_dir, f"sample_{i:03d}_fx.png"), dpi=300, bbox_inches="tight", pad_inches=0)
                plt.close()

                plt.figure(figsize=(4, 4))
                plt.imshow(hrz_masked, cmap=hrzmap, vmin=-1.0, vmax=1.0)
                plt.axis("off")
                plt.savefig(os.path.join(out_dir, f"sample_{i:03d}_hrz.png"), dpi=300, bbox_inches="tight", pad_inches=0)
                plt.close()

                black_cmap = ListedColormap(['black'])
                black_cmap.set_bad(alpha=0) 

                plt.figure(figsize=(4, 4))
                plt.imshow(pred_img, cmap=getStrataColors(), vmin=-1.0, vmax=1.0)
                plt.imshow(np.ma.masked_where(hrz_np<= -0.999, hrz_np), cmap=black_cmap, vmin=-1.0, vmax=1.0, interpolation='nearest')
                plt.axis("off")
                
                png_path = os.path.join(out_dir, f"sample_{i:03d}_pred_overlay.png")
                plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
                plt.close()

    print(f"[INFO] Average generation time: {total_gen_time / args.num_samples:.4f}s")

    n = len(pred_imgs)
    cols = max(1, int(getattr(args, "grid_cols", 5)))
    rows = math.ceil(n / cols)

    hrzmap = getStrataColors()
    fxmap = getFxColor()
    stratamap = getStrataColors()

    # same masking threshold
    thr = -0.999
    fx_m_grid  = np.ma.masked_where(fx_np  <= thr, fx_np)
    hrz_m_grid = np.ma.masked_where(hrz_np <= thr, hrz_np)

    fxmap.set_bad((0, 0, 0, 0))      # transparent
    hrzmap.set_bad((0, 0, 0, 0))     # transparent

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes = np.array(axes).reshape(-1)  # flatten, works for 1 row/col too

    for j in range(rows * cols):
        ax = axes[j]
        ax.axis("off")
        if j < n:
            # pred image: use the same strata colormap (instead of tab20)
            ax.imshow(pred_imgs[j], cmap=stratamap, vmin=-1.0, vmax=1.0)

            # overlays: use the same masked + colormap logic
            ax.imshow(fx_m_grid,  cmap=fxmap,  vmin=-1.0, vmax=1.0)
            ax.imshow(hrz_m_grid, cmap=hrzmap, vmin=-1.0, vmax=1.0)

            ax.text(
                0.02, 0.06, f"#{j:03d}",
                transform=ax.transAxes,
                fontsize=10,
                color="white",
                ha="left", va="bottom",
                bbox=dict(facecolor="black", alpha=0.35, pad=1, edgecolor="none")
            )

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0.02)

    grid_path = os.path.join(out_dir, "pred_grid.png")
    plt.savefig(grid_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[INFO] Summary grid saved to: {grid_path}")

if __name__ == "__main__":
    main()