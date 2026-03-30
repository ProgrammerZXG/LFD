# -*- coding: utf-8 -*-
"""
Single-file inference script: generate RGT predictions from a trained LFD model
given geological condition inputs (fault + horizon).
Supports input formats: NPZ (multi-channel conditions) and NPY (pre-processed tensors).
"""

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
from engine import get_stratigraphy_colormap, get_fault_colormap


# ============================================================
# Image post-processing utilities
# ============================================================

def fill_fault_pixels_with_nearest_neighbor(
    pred_np: np.ndarray,
    fault_np: np.ndarray,
    fault_threshold: float = -0.999,
    dilate_iters: int = 0
) -> np.ndarray:
    """
    Replace prediction values at fault pixel locations with the nearest
    non-fault neighbor value. Fault-region predictions are typically unreliable;
    nearest-neighbor filling produces a smooth transition across faults.

    Args:
        pred_np:         Predicted image, shape (B, C, H, W)
        fault_np:        Fault map, shape (H, W); background ~= -1, fault ~= +1
        fault_threshold: Pixels above this value are treated as fault
        dilate_iters:    Optional dilation iterations on fault mask to remove
                         edge artifacts (0 = no dilation)

    Returns:
        Repaired prediction array, shape (B, C, H, W)
    """
    assert pred_np.ndim == 4, f"pred_np must be [B,C,H,W], got {pred_np.shape}"
    assert fault_np.ndim == 2, f"fault_np must be [H,W], got {fault_np.shape}"

    fault_region_mask = (fault_np > fault_threshold)
    if dilate_iters > 0:
        fault_region_mask = scipy.ndimage.binary_dilation(fault_region_mask, iterations=dilate_iters)

    if not np.any(fault_region_mask):
        return pred_np

    # Euclidean distance transform: for each fault pixel find the nearest non-fault pixel index
    _, (nearest_row_idx, nearest_col_idx) = scipy.ndimage.distance_transform_edt(
        fault_region_mask.astype(np.uint8),
        return_indices=True
    )

    output = pred_np.copy()
    B, C, H, W = output.shape
    for b in range(B):
        for c in range(C):
            channel_img = output[b, c]
            channel_img[fault_region_mask] = channel_img[
                nearest_row_idx[fault_region_mask],
                nearest_col_idx[fault_region_mask]
            ]
            output[b, c] = channel_img
    return output


def apply_gaussian_smoothing(pred_np, sigma=(0.5, 1.0)):
    """
    Apply anisotropic Gaussian smoothing to reduce high-frequency noise.

    Args:
        pred_np: Prediction array, shape (B, C, H, W)
        sigma:   Anisotropic smoothing parameters (sigma_y, sigma_x)

    Returns:
        Smoothed prediction array, shape (B, C, H, W)
    """
    for b in range(pred_np.shape[0]):
        for c in range(pred_np.shape[1]):
            pred_np[b, c] = scipy.ndimage.gaussian_filter(
                pred_np[b, c],
                sigma=sigma,
                mode="nearest"
            )
    return pred_np


# ============================================================
# Argument parsing
# ============================================================

def parse_condition_keys(raw_cond_arg):
    """
    Parse the --cond argument into a list of key strings.
    Supports multiple input formats (space-separated, comma-separated, bracketed).
    """
    if isinstance(raw_cond_arg, list):
        if len(raw_cond_arg) == 1:
            single_str = raw_cond_arg[0]
        else:
            return raw_cond_arg
    else:
        single_str = raw_cond_arg

    single_str = single_str.strip()
    if single_str.startswith('[') and single_str.endswith(']'):
        single_str = single_str[1:-1]
    single_str = single_str.replace('"', '').replace("'", '')

    parsed_keys = []
    for key_token in single_str.replace(',', ' ').split():
        key_token = key_token.strip()
        if key_token:
            parsed_keys.append(key_token)
    return parsed_keys


# ============================================================
# Data normalization
# ============================================================

def normalize_geologic_array(array: np.ndarray, key: Optional[str] = None) -> np.ndarray:
    """
    Normalize a geological array to [-1, 1] using key-specific strategies:

    - 'fx' / 'fault':  binarize: non-zero -> +1, zero -> -1
    - 'rgt':           linear min-max stretch to [-1, 1]
    - 'hrz' / 'horiz': sort horizons by mean depth, map shallowest -> -1, deepest -> +1;
                        all pixels belonging to the same horizon receive the same value

    Any other key: return as-is (no normalization).

    Args:
        array: Input geological array, shape (H, W)
        key:   Geological data type key

    Returns:
        Normalized float32 array
    """
    array = array.astype(np.float32, copy=False)

    # ---- 1) Fault: binarization ----
    if key in ["fx", "fault"]:
        normalized = np.full_like(array, -1.0, dtype=np.float32)
        normalized[array != 0] = 1.0
        return normalized

    # ---- 2) RGT: linear min-max normalization ----
    if key == "rgt":
        min_val = float(array.min())
        max_val = float(array.max())
        if max_val - min_val > 1e-6:
            array = (array - min_val) / (max_val - min_val) * 2.0 - 1.0
        else:
            array = np.full_like(array, -1.0, dtype=np.float32)
        return array

    # ---- 3) Horizon: depth-sorted uniform mapping ----
    if key in ("hrz", "horiz"):
        depth_axis = 0  # H axis corresponds to depth direction

        horizon_labels = np.unique(array)
        horizon_labels = horizon_labels[horizon_labels != 0]  # remove background (label=0)
        if horizon_labels.size == 0:
            return np.full_like(array, -1.0, dtype=np.float32)

        # Compute the mean depth (row index) for each horizon label
        depth_index_map = np.indices(array.shape, dtype=np.int32)[depth_axis].astype(np.float32)

        label_depth_pairs = []
        for label in horizon_labels:
            avg_depth = depth_index_map[array == label].mean()
            label_depth_pairs.append((label, avg_depth))

        print(f"Horizon depth info: {label_depth_pairs}")
        label_depth_pairs.sort(key=lambda x: x[1])  # sort by increasing depth

        max_depth_value = float(array.shape[depth_axis] - 1)  # e.g., 512 -> 511

        # Map each horizon's mean depth linearly to [-1, 1]
        label_to_normalized_value = {}
        for label, avg_depth in label_depth_pairs:
            normalized_val = -1.0 + 2.0 * (float(avg_depth) / max(max_depth_value, 1.0))
            label_to_normalized_value[label] = np.float32(np.clip(normalized_val, -1.0, 1.0))

        # Build output: valid horizon positions get their normalized value, rest -> -1
        normalized_array = np.full_like(array, -1.0, dtype=np.float32)
        for label, normalized_val in label_to_normalized_value.items():
            normalized_array[array == label] = normalized_val

        return normalized_array

    # ---- 4) Default: return unchanged ----
    return array


# ============================================================
# Data loading functions
# ============================================================

def load_conditions_from_npz(npz_path: str, condition_keys: list[str]) -> torch.Tensor:
    """
    Load multiple condition channels from a single NPZ file, normalize each,
    and concatenate into a single tensor. Interpolates to 512x512 to match model input.

    Args:
        npz_path:       Path to the NPZ file
        condition_keys: List of keys to load (e.g., ['fault', 'horiz'])

    Returns:
        Condition tensor, shape (1, C_cond, 512, 512)
    """
    npz_data = np.load(npz_path)
    channel_tensors = []
    for key in condition_keys:
        if key not in npz_data:
            raise KeyError(f"Key '{key}' not found in NPZ file: {npz_path}")
        raw_array = npz_data[key]     # (H, W)
        print(f"  Loaded '{key}', shape: {raw_array.shape}")
        normalized = normalize_geologic_array(raw_array, key=key)
        channel_tensor = torch.from_numpy(normalized).float().unsqueeze(0)  # (1, H, W)
        channel_tensors.append(channel_tensor)

    condition_tensor = torch.cat(channel_tensors, dim=0)      # (C_cond, H, W)
    condition_tensor = condition_tensor.unsqueeze(0)           # (1, C_cond, H, W)

    # Resize to model input resolution via nearest-neighbor interpolation
    condition_tensor = torch.nn.functional.interpolate(
        condition_tensor, size=(512, 512), mode="nearest"
    )
    return condition_tensor


def load_conditions_from_npy(npy_path: str) -> torch.Tensor:
    """
    Load a pre-processed condition tensor from an NPY file.

    Supported shapes:
    - (H, W)    -> expanded to (1, 1, H, W) as a single-channel input
    - (C, H, W) -> expanded to (1, C, H, W) as a multi-channel input

    Args:
        npy_path: Path to the NPY file

    Returns:
        Condition tensor, shape (1, C, H, W)
    """
    raw_array = np.load(npy_path).astype(np.float32, copy=False)

    if raw_array.ndim == 2:
        # (H, W) -> (1, 1, H, W)
        raw_array = raw_array[None, None, :, :]
    elif raw_array.ndim == 3:
        # (C, H, W) -> (1, C, H, W)
        raw_array = raw_array[None, :, :, :]
    else:
        raise ValueError(
            f"Unsupported NPY shape {raw_array.shape}; expected (H,W) or (C,H,W)."
        )

    return torch.from_numpy(raw_array).float()


# ============================================================
# Model construction
# ============================================================

def build_and_load_model(args, device: torch.device) -> torch.nn.Module:
    """
    Instantiate DenoiserFH and load checkpoint weights (EMA1 preferred).

    Args:
        args:   Command-line arguments (model architecture + checkpoint path)
        device: Target device

    Returns:
        Model in eval mode with weights loaded
    """
    model = Denoiser(args).to(device)

    # If a directory is given, look for checkpoint-last.pth inside it
    ckpt_file_path = args.ckpt
    if os.path.isdir(ckpt_file_path):
        ckpt_file_path = os.path.join(ckpt_file_path, "checkpoint-last.pth")

    if not os.path.exists(ckpt_file_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file_path}")

    print(f"[INFO] Loading checkpoint from: {ckpt_file_path}")
    checkpoint = torch.load(ckpt_file_path, map_location=device, weights_only=False)

    # Priority: EMA1 > model > raw state_dict
    if isinstance(checkpoint, dict) and "model_ema1" in checkpoint:
        state_dict = checkpoint["model_ema1"]
        print("[INFO] Using EMA1 weights (recommended for highest generation quality).")
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        print("[INFO] Using 'model' weights (no EMA found).")
    else:
        state_dict = checkpoint
        print("[WARN] Non-standard checkpoint format; treating as plain state_dict.")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[WARN] {len(missing_keys)} missing keys in state_dict")
    if unexpected_keys:
        print(f"[WARN] {len(unexpected_keys)} unexpected keys in state_dict")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params / 1e6:.2f} M")
    model.eval()
    return model


# ============================================================
# Inference entry point
# ============================================================

def build_inference_arg_parser():
    """Build the argument parser for the inference script."""
    parser = argparse.ArgumentParser("Single-file inference for LFD")

    # ---- Required arguments ----
    parser.add_argument("--input", type=str, default="xxx.npz",
                        help="Path to input file (.npz or .npy)")
    parser.add_argument("--ckpt", type=str,
                        default="./result/lfd_rgt_10abshrzloss_0.1bending",
                        help="Checkpoint path (directory or .pth file)")

    # ---- Output arguments ----
    parser.add_argument("--out", type=str, default="./output/xxx",
                        help="Base path for the output directory")

    # ---- Sampling arguments ----
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples to generate for the same condition")
    parser.add_argument("--grid_cols", type=int, default=5,
                        help="Number of columns in the summary grid image")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Batch size for generation")

    # ---- Model architecture (must match training configuration) ----
    parser.add_argument("--model", type=str, default="LFD-B/32")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--class_num", type=int, default=1)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--cond_in_ch", type=int, default=2)

    # ---- Flow Matching parameters ----
    parser.add_argument("--P_mean", type=float, default=-1.0)
    parser.add_argument("--P_std", type=float, default=0.8)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--t_eps", type=float, default=5e-2)
    parser.add_argument("--label_drop_prob", type=float, default=0.0)
    parser.add_argument("--ema_decay1", type=float, default=0.9999)
    parser.add_argument("--ema_decay2", type=float, default=0.9996)

    # ---- ODE sampling parameters ----
    parser.add_argument("--sampling_method", type=str, default="heun",
                        help="ODE integration method: 'euler' or 'heun'")
    parser.add_argument("--num_sampling_steps", type=int, default=50,
                        help="Number of ODE integration steps")
    parser.add_argument("--cfg", type=float, default=3.0,
                        help="Classifier-Free Guidance scale")
    parser.add_argument("--interval_min", type=float, default=0.1,
                        help="Lower bound of CFG timestep interval")
    parser.add_argument("--interval_max", type=float, default=1.0,
                        help="Upper bound of CFG timestep interval")

    # ---- Condition keys ----
    parser.add_argument("--cond", nargs="+", default=["fault", "horiz"],
                        help="NPZ keys for condition channels, e.g., fx hrz")

    # ---- Device ----
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device: 'cuda' or 'cpu'")
    parser.add_argument('--pretrained_base', default='', type=str,
                        help='Path to pretrained checkpoint for transfer learning init')

    args = parser.parse_args()
    args.cond = parse_condition_keys(args.cond)
    return args


def main():
    """
    Inference main function:
    1. Load model and condition inputs
    2. Generate RGT predictions in batches
    3. Post-process (Gaussian smoothing + fault region repair)
    4. Save individual sample images and a summary grid image
    """
    args = build_inference_arg_parser()

    # ---- Device initialization ----
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # ---- Load model ----
    model = build_and_load_model(args, device)

    # Class label fixed to 0 (single-class task)
    base_class_labels = torch.zeros(1, dtype=torch.long, device=device)

    # ---- Load condition input ----
    input_file_path = args.input
    input_extension = Path(input_file_path).suffix.lower()

    if input_extension == ".npz":
        print(f"[INFO] Loading NPZ: {input_file_path}, condition keys = {args.cond}")
        condition_tensor = load_conditions_from_npz(input_file_path, args.cond)
    elif input_extension == ".npy":
        print(f"[INFO] Loading NPY: {input_file_path}")
        condition_tensor = load_conditions_from_npy(input_file_path)
    else:
        raise ValueError(f"Unsupported input extension '{input_extension}'; only .npz/.npy are supported.")

    condition_tensor = condition_tensor.to(device, non_blocking=True)

    # ---- Configure output directory ----
    output_base_dir = Path(args.out)
    output_dir = os.path.join(
        output_base_dir,
        args.ckpt.split("/")[-1],
        "cfg_" + str(args.cfg)
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Starting generation: {args.num_samples} samples")

    # ---- Extract condition channels for post-processing and visualization ----
    fault_channel_np = condition_tensor[0, 0].detach().cpu().numpy()    # (H, W)
    horizon_channel_np = condition_tensor[0, 1].detach().cpu().numpy()  # (H, W)

    print(f"fault  channel min/max: {fault_channel_np.min():.3f}/{fault_channel_np.max():.3f}")
    print(f"horizon channel min/max: {horizon_channel_np.min():.3f}/{horizon_channel_np.max():.3f}")

    all_pred_images = []  # collect all prediction images for summary grid

    # ---- Prepare colormaps ----
    horizon_cmap = get_stratigraphy_colormap()
    horizon_cmap.set_bad('#f0f5f9')
    fault_cmap = get_fault_colormap()
    fault_cmap.set_bad('#f0f5f9')

    # Apply slight dilation to condition channels for better visualization
    horizon_channel_thick = scipy.ndimage.maximum_filter(horizon_channel_np, size=2)
    fault_channel_thick = scipy.ndimage.maximum_filter(fault_channel_np, size=2)

    # Masked arrays: background pixels are hidden in visualization
    fault_channel_masked = np.ma.masked_where(fault_channel_thick <= -0.999, fault_channel_thick)
    horizon_channel_masked = np.ma.masked_where(horizon_channel_thick <= -0.999, horizon_channel_thick)

    # ---- Batched generation ----
    total_generation_time = 0.0
    num_batches = math.ceil(args.num_samples / args.batch_size)

    with torch.no_grad():
        for batch_idx in range(num_batches):
            batch_start_idx = batch_idx * args.batch_size
            batch_end_idx = min((batch_idx + 1) * args.batch_size, args.num_samples)
            current_batch_size = batch_end_idx - batch_start_idx

            print(f"[INFO] Generating batch {batch_idx + 1}/{num_batches} "
                  f"(samples {batch_start_idx} to {batch_end_idx - 1})")

            # Build batch inputs
            batch_labels = torch.zeros(current_batch_size, dtype=torch.long, device=device)
            batch_conditions = condition_tensor.repeat(current_batch_size, 1, 1, 1)

            # Timed generation
            gen_start = time.time()
            generated_batch = model.generate(batch_labels, batch_conditions)  # (B, C, H, W)
            gen_elapsed = time.time() - gen_start
            total_generation_time += gen_elapsed
            print(f"[INFO] Batch generation time: {gen_elapsed:.4f}s")

            # ---- Post-processing: smooth + repair fault regions ----
            pred_np = generated_batch.detach().cpu().numpy()
            pred_np = apply_gaussian_smoothing(pred_np, sigma=1)
            pred_np = fill_fault_pixels_with_nearest_neighbor(
                pred_np,
                fault_np=fault_channel_thick,
                fault_threshold=-0.999,
                dilate_iters=1
            )

            # ---- Save individual sample results ----
            for sample_local_idx in range(current_batch_size):
                global_sample_idx = batch_start_idx + sample_local_idx
                single_pred = pred_np[sample_local_idx:sample_local_idx + 1]  # (1, C, H, W)

                # Save raw prediction as NPY (full precision)
                npy_save_path = os.path.join(output_dir, f"sample_{global_sample_idx:03d}.npy")
                np.save(npy_save_path, single_pred)

                pred_2d = single_pred[0, 0]  # (H, W)
                all_pred_images.append(pred_2d)

                # ---- Save fault condition visualization ----
                plt.figure(figsize=(4, 4))
                plt.imshow(fault_channel_masked, cmap=fault_cmap, vmin=-1.0, vmax=1.0)
                plt.axis("off")
                plt.savefig(
                    os.path.join(output_dir, f"sample_{global_sample_idx:03d}_fx.png"),
                    dpi=300, bbox_inches="tight", pad_inches=0
                )
                plt.close()

                # ---- Save horizon condition visualization ----
                plt.figure(figsize=(4, 4))
                plt.imshow(horizon_channel_masked, cmap=horizon_cmap, vmin=-1.0, vmax=1.0)
                plt.axis("off")
                plt.savefig(
                    os.path.join(output_dir, f"sample_{global_sample_idx:03d}_hrz.png"),
                    dpi=300, bbox_inches="tight", pad_inches=0
                )
                plt.close()

                # ---- Save prediction + horizon overlay visualization ----
                # Black colormap for horizon overlay (background = transparent)
                black_cmap = ListedColormap(['black'])
                black_cmap.set_bad(alpha=0)

                plt.figure(figsize=(4, 4))
                # Bottom layer: predicted RGT with stratigraphy colormap
                plt.imshow(pred_2d, cmap=get_stratigraphy_colormap(), vmin=-1.0, vmax=1.0)
                # Overlay: horizon annotations in black
                plt.imshow(
                    np.ma.masked_where(horizon_channel_np <= -0.999, horizon_channel_np),
                    cmap=black_cmap, vmin=-1.0, vmax=1.0, interpolation='nearest'
                )
                plt.axis("off")
                overlay_save_path = os.path.join(
                    output_dir, f"sample_{global_sample_idx:03d}_pred_overlay.png"
                )
                plt.savefig(overlay_save_path, dpi=300, bbox_inches="tight", pad_inches=0)
                plt.close()

    print(f"[INFO] Average generation time per sample: {total_generation_time / args.num_samples:.4f}s")

    # ============================================================
    # Generate summary grid image
    # ============================================================
    num_generated = len(all_pred_images)
    num_grid_cols = max(1, int(getattr(args, "grid_cols", 5)))
    num_grid_rows = math.ceil(num_generated / num_grid_cols)

    # Colormaps for grid
    grid_horizon_cmap = get_stratigraphy_colormap()
    grid_fault_cmap = get_fault_colormap()
    grid_strata_cmap = get_stratigraphy_colormap()

    # Set background to transparent for overlay display
    bg_mask_threshold = -0.999
    fault_masked_for_grid = np.ma.masked_where(fault_channel_np <= bg_mask_threshold, fault_channel_np)
    horizon_masked_for_grid = np.ma.masked_where(horizon_channel_np <= bg_mask_threshold, horizon_channel_np)
    grid_fault_cmap.set_bad((0, 0, 0, 0))
    grid_horizon_cmap.set_bad((0, 0, 0, 0))

    fig, axes = plt.subplots(num_grid_rows, num_grid_cols,
                             figsize=(num_grid_cols * 3.0, num_grid_rows * 3.0))
    axes = np.array(axes).reshape(-1)  # flatten to 1D for uniform indexing

    for cell_idx in range(num_grid_rows * num_grid_cols):
        ax = axes[cell_idx]
        ax.axis("off")
        if cell_idx < num_generated:
            # Bottom layer: predicted RGT
            ax.imshow(all_pred_images[cell_idx], cmap=grid_strata_cmap, vmin=-1.0, vmax=1.0)
            # Overlay: fault + horizon
            ax.imshow(fault_masked_for_grid, cmap=grid_fault_cmap, vmin=-1.0, vmax=1.0)
            ax.imshow(horizon_masked_for_grid, cmap=grid_horizon_cmap, vmin=-1.0, vmax=1.0)
            # Sample index label in top-left corner
            ax.text(
                0.02, 0.06, f"#{cell_idx:03d}",
                transform=ax.transAxes,
                fontsize=10, color="white",
                ha="left", va="bottom",
                bbox=dict(facecolor="black", alpha=0.35, pad=1, edgecolor="none")
            )

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0.02)

    grid_save_path = os.path.join(output_dir, "pred_grid.png")
    plt.savefig(grid_save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[INFO] Summary grid saved to: {grid_save_path}")


if __name__ == "__main__":
    main()
