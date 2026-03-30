import math
import sys
import os
import shutil
import copy
import time
import contextlib

import torch
import numpy as np

import util.misc as misc
import util.lr_sched as lr_sched
import copy
import matplotlib
matplotlib.use("Agg")  # Required for server / no-display environments
import matplotlib.pyplot as plt
import util.misc as misc
from matplotlib.colors import ListedColormap
import scipy.ndimage


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

    # Euclidean distance transform: for each fault pixel, find the nearest non-fault pixel index
    _, (nearest_row_indices, nearest_col_indices) = scipy.ndimage.distance_transform_edt(
        fault_region_mask.astype(np.uint8),
        return_indices=True
    )

    output = pred_np.copy()
    batch_size, num_channels, H, W = output.shape
    for b in range(batch_size):
        for c in range(num_channels):
            channel_img = output[b, c]
            channel_img[fault_region_mask] = channel_img[
                nearest_row_indices[fault_region_mask],
                nearest_col_indices[fault_region_mask]
            ]
            output[b, c] = channel_img
    return output


def apply_gaussian_smoothing(pred_np, sigma=(0.5, 1.0)):
    """
    Apply anisotropic Gaussian smoothing to prediction arrays to reduce noise.

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
# Visualization color maps
# ============================================================

def get_fault_colormap():
    """
    Return a single-color colormap (dark red) for fault visualization.
    """
    return ListedColormap(["#a6172d"])


def get_stratigraphy_colormap(alpha=1, fill_except_min=False, reverse=False):
    """
    Return a multi-color colormap (256 bins) for stratigraphy visualization.
    Every 8 consecutive values share one color; 32 base colors * 8 = 256 entries.

    Args:
        alpha:           Global transparency (0~1)
        fill_except_min: If True, set the minimum-value bin (index 0) to transparent
        reverse:         If True, reverse the color order

    Returns:
        matplotlib ListedColormap
    """
    rgba_table = np.full((256, 4), 0, dtype=np.float32)

    base_colors = np.array([
        [1.0, 0.0, 0.0, alpha], [1.0, 0.5019608, 0.0, alpha], [1.0, 1.0, 0.0, alpha],
        [0.0, 1, 0.0, alpha], [0.0, 0.5019608, 0.0, alpha], [0.0, 0.2509804, 0.0, alpha],
        [0, 1.0, 1.0, alpha], [0.0, 0.5019608, 1.0, alpha], [0.0, 0.0, 1.0, alpha],
        [0.0, 0.0, 0.627451, alpha], [0.0, 0.5019608, 0.7529412, alpha], [1.0, 0.5019608, 0.5019608, alpha],
        [0.5019608, 0.5019608, 1.0, alpha], [0.5019608, 0.0, 1.0, alpha], [0.5019608, 0, 0.5019608, alpha],
        [1.0, 0.5019608, 1.0, alpha], [1.0, 0.0, 1.0, alpha], [0.5019608, 0.2509804, 0, alpha],
        # Original gray 1 (will be overridden)
        [0.5019608, 0.5019608, 0.5019608, alpha],
        # Original gray 2 (will be overridden)
        [0.7529412, 0.7529412, 0.7529412, alpha],
        [0.2509804, 0, 0.2509804, alpha],
        [0.90588236, 0.7294118, 0.19607843, alpha], [0.44313726, 0.58431375, 0.58431375, alpha],
        [0.5254902, 0.42352942, 0.4862745, alpha],
        [0.7176471, 0.54509807, 0.44313726, alpha], [0.5019608, 0.5019608, 0, alpha],
        [0.7529412, 0.7294118, 0.8784314, alpha],
        [0.61960787, 0.85882354, 0.9882353, alpha], [0.7372549, 0.25882354, 0.24705882, alpha],
        [0.8862745, 0.8509804, 0.627451, alpha],
        [0.60784316, 0.9411765, 0.7490196, alpha], [0.62352943, 0.79607844, 0.105882354, alpha]
    ], dtype=np.float32)

    # Override selected gray entries with custom accent colors
    color_gold = np.array([0xFD / 255.0, 0xC2 / 255.0, 0x3E / 255.0], dtype=np.float32)
    color_sky_blue = np.array([0x6A / 255.0, 0xAF / 255.0, 0xE6 / 255.0], dtype=np.float32)
    color_teal = np.array([0x67 / 255.0, 0xD5 / 255.0, 0xB5 / 255.0], dtype=np.float32)

    base_colors[2, :3] = color_gold
    base_colors[18, :3] = color_sky_blue
    base_colors[19, :3] = color_teal

    # Fill each base color into 8 consecutive bins
    for color_idx in range(32):
        rgba_table[color_idx * 8:(color_idx + 1) * 8] = base_colors[color_idx]

    if fill_except_min:
        rgba_table[0, -1] = 0  # Set minimum bin to transparent
    if reverse:
        rgba_table = np.flip(rgba_table, axis=0)

    return ListedColormap(rgba_table)


# ============================================================
# Training function
# ============================================================

def train_one_epoch(
    model,
    model_without_ddp,
    data_loader,
    optimizer,
    device,
    epoch,
    log_writer=None,
    args=None
):
    """
    Run one training epoch.

    Args:
        model:             DDP-wrapped model (used for forward pass)
        model_without_ddp: Unwrapped model (used for EMA updates and checkpointing)
        data_loader:       Training data loader; each batch yields (x, labels, cond)
        optimizer:         AdamW optimizer
        device:            Training device
        epoch:             Current epoch index
        log_writer:        TensorBoard SummaryWriter (None = no logging)
        args:              Training configuration arguments
    """
    model.train(True)

    # Metric logger tracks loss, lr, and individual loss components
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_v', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_h', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_b', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    epoch_header = 'Epoch: [{}]'.format(epoch)
    log_print_freq = 20  # print training log every N steps

    optimizer.zero_grad()

    if log_writer is not None:
        print('TensorBoard log dir: {}'.format(log_writer.log_dir))

    for batch_step, (images, class_labels, condition_maps) in enumerate(
        metric_logger.log_every(data_loader, log_print_freq, epoch_header)
    ):
        # Update learning rate per iteration (not per epoch) for smoother scheduling
        lr_sched.adjust_learning_rate(
            optimizer,
            batch_step / len(data_loader) + epoch,
            args
        )

        # Move data to GPU
        images = images.to(device)
        class_labels = class_labels.to(device, non_blocking=True)
        condition_maps = condition_maps.to(device, non_blocking=True)

        # Forward pass with bfloat16 automatic mixed precision
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            total_loss, velocity_loss, horizon_loss, bending_loss = model(
                images, class_labels, condition_maps
            )

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Backpropagation and optimizer step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        # Update EMA parameters
        model_without_ddp.update_ema()

        # Update metric logger
        metric_logger.update(loss=loss_value)
        current_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=current_lr)
        metric_logger.update(loss_v=velocity_loss.item())
        metric_logger.update(loss_h=horizon_loss.item())
        metric_logger.update(loss_b=bending_loss.item())

        # All-reduce loss for global mean (used for TensorBoard logging)
        loss_global_mean = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as x-axis in TensorBoard to smooth out per-epoch curves
            epoch_1000x = int((batch_step / len(data_loader) + epoch) * 1000)
            if batch_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_global_mean, epoch_1000x)
                log_writer.add_scalar('lr', current_lr, epoch_1000x)
                log_writer.add_scalar('loss_v', velocity_loss.item(), epoch_1000x)
                log_writer.add_scalar('loss_h', horizon_loss.item(), epoch_1000x)
                log_writer.add_scalar('loss_b', bending_loss.item(), epoch_1000x)


# ============================================================
# Evaluation function (conditional generation)
# ============================================================

def evaluate_conditional_generation(
    model_without_ddp,
    args,
    epoch,
    val_loader,
    log_writer=None
):
    """
    Conditional generation evaluation: generate RGT predictions conditioned on
    validation-set fault and horizon maps, then save visualization results.

    Args:
        model_without_ddp: Unwrapped model
        args:              Evaluation configuration arguments
        epoch:             Current epoch (used for output directory naming)
        val_loader:        Validation data loader; each batch yields (x, labels, cond)
                           - x:     (B, C_x, H, W), target image (not used for generation)
                           - labels:(B,), class labels
                           - cond:  (B, C_cond, H, W), condition maps; C_cond = len(args.cond)
        log_writer:        TensorBoard SummaryWriter (optional)
    """
    device = torch.device(args.device)
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()

    max_num_images = args.num_images  # maximum number of images to generate

    # Parse condition key list
    if isinstance(args.cond, (list, tuple)):
        cond_key_list = list(args.cond)
    else:
        cond_key_list = [args.cond]
    cond_tag_str = "+".join(cond_key_list)

    # Build output directory path
    save_dir = os.path.join(
        "ssd/tmp",
        args.output_dir,
        "condVAL-steps{}-max{}-res{}_{}".format(
            model_without_ddp.num_sampling_steps,
            max_num_images, args.img_size, cond_tag_str
        )
    )
    print("Saving results to:", save_dir)
    if misc.get_rank() == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # ---- Switch to EMA1 parameters (higher generation quality) ----
    original_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for param_idx, (param_name, _) in enumerate(model_without_ddp.named_parameters()):
        assert param_name in ema_state_dict
        ema_state_dict[param_name] = model_without_ddp.ema_params1[param_idx]
    print("Switched to EMA1 parameters")
    model_without_ddp.load_state_dict(ema_state_dict)

    def get_condition_colormap(cond_name: str):
        """Return the appropriate colormap for a given condition type."""
        colormap_mapping = {
            "fx": get_fault_colormap(),
            "sx": "gray",
            "rgt": get_stratigraphy_colormap(),
            "imp": "jet",
            "hrz": get_stratigraphy_colormap(),
        }
        return colormap_mapping.get(cond_name, "gray")

    # ---- Timing statistics ----
    generation_time_total = 0.0    # cumulative model generation time
    saving_time_total = 0.0        # cumulative image save / plotting time
    num_generated_images = 0       # number of images generated on this rank

    def sync_cuda():
        """Synchronize CUDA operations for accurate timing."""
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Autocast context for mixed-precision inference
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    total_images_so_far = 0  # global image counter across batches

    for batch_idx, (target_images, class_labels, condition_maps) in enumerate(val_loader):
        if total_images_so_far >= max_num_images:
            break

        # Move to GPU
        target_images = target_images.to(device, non_blocking=True)
        condition_maps = condition_maps.to(device, non_blocking=True)
        class_labels = class_labels.to(device, non_blocking=True)

        current_batch_size = condition_maps.size(0)
        # Trim batch if we would exceed the max image count
        if total_images_so_far + current_batch_size > max_num_images:
            keep_count = max_num_images - total_images_so_far
            condition_maps = condition_maps[:keep_count]
            class_labels = class_labels[:keep_count]
            target_images = target_images[:keep_count]
            current_batch_size = keep_count

        # ---- Model generation (timed) ----
        sync_cuda()
        gen_start_time = time.perf_counter()
        with autocast_ctx:
            generated_images = model_without_ddp.generate(class_labels, condition_maps)
        sync_cuda()
        gen_end_time = time.perf_counter()

        generation_time_total += (gen_end_time - gen_start_time)
        num_generated_images += current_batch_size

        # Move to CPU (not counted in generation time)
        generated_images = generated_images.detach().cpu()
        cond_cpu = condition_maps.detach().cpu()

        B, C_cond, H, W = cond_cpu.shape
        assert C_cond == len(cond_key_list), \
            f"Condition channels ({C_cond}) != len(cond_keys) ({len(cond_key_list)})"

        # Layout: one row per condition channel + one row for prediction
        num_plot_rows = C_cond + 1

        # ---- Plotting and saving (timed) ----
        plot_start_time = time.perf_counter()

        for sample_idx in range(current_batch_size):
            global_img_id = total_images_so_far + sample_idx

            # ---- Prediction post-processing ----
            pred_4d = generated_images[sample_idx].unsqueeze(0).float().numpy()
            pred_4d = apply_gaussian_smoothing(pred_4d, sigma=1)
            # Fill fault-region pixels using nearest-neighbor interpolation
            pred_4d = fill_fault_pixels_with_nearest_neighbor(
                pred_4d,
                fault_np=cond_cpu[sample_idx, 0].numpy(),
                fault_threshold=-0.999,
                dilate_iters=1
            )
            pred_2d = pred_4d[0, 0]  # (H, W)

            prediction_colormap = get_stratigraphy_colormap()

            # ---- Plot: condition rows + prediction row ----
            plt.figure(figsize=(4, 4 * num_plot_rows))

            for cond_channel_idx, cond_name in enumerate(cond_key_list):
                cond_channel_np = cond_cpu[sample_idx, cond_channel_idx].numpy()
                cmap = plt.get_cmap(get_condition_colormap(cond_name)).copy()
                cmap.set_bad('#f0f5f9')  # NaN / masked values shown as light gray

                # Apply slight dilation to the second condition channel (usually horizon)
                # to improve visualization of thin horizon lines
                if cond_channel_idx == 1:
                    import scipy.ndimage
                    cond_channel_np = scipy.ndimage.maximum_filter(cond_channel_np, size=3)
                    cmap = plt.get_cmap(get_condition_colormap(cond_name)).copy()
                    cmap.set_bad('#f0f5f9')

                plt.subplot(num_plot_rows, 1, cond_channel_idx + 1)
                # Mask out background values (< -0.95) from color mapping
                plt.imshow(
                    np.ma.masked_where(cond_channel_np < -0.95, cond_channel_np),
                    cmap=cmap, vmin=-1.0, vmax=1.0
                )
                plt.axis("off")

            # Final row: prediction result
            plt.subplot(num_plot_rows, 1, num_plot_rows)
            plt.imshow(pred_2d, cmap=prediction_colormap, vmin=-1.0, vmax=1.0)
            plt.axis("off")

            plt.tight_layout(pad=0.1)

            # Filename: rank + global image id + condition type tag
            output_filename = (
                f"rank{local_rank}_img{str(global_img_id).zfill(6)}_"
                f"cond-{'-'.join(cond_key_list)}.png"
            )
            plt.savefig(os.path.join(save_dir, output_filename), dpi=150)
            plt.close()

        plot_end_time = time.perf_counter()
        saving_time_total += (plot_end_time - plot_start_time)

        total_images_so_far += current_batch_size

    # ---- Distributed synchronization barrier ----
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # ---- Aggregate timing statistics across all ranks and print per-image averages ----
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        timing_tensor = torch.tensor(
            [generation_time_total, saving_time_total, float(num_generated_images)],
            device=device, dtype=torch.float64
        )
        torch.distributed.all_reduce(timing_tensor, op=torch.distributed.ReduceOp.SUM)
        gen_time_all, save_time_all, total_img_count = timing_tensor.tolist()
    else:
        gen_time_all = generation_time_total
        save_time_all = saving_time_total
        total_img_count = float(num_generated_images)

    if misc.get_rank() == 0 and total_img_count > 0:
        avg_gen_ms = 1000.0 * gen_time_all / total_img_count
        avg_save_ms = 1000.0 * save_time_all / total_img_count
        avg_e2e_ms = avg_gen_ms + avg_save_ms

        print(f"[Timing] total images = {int(total_img_count)}")
        print(f"[Timing] avg generate() per image = {avg_gen_ms:.3f} ms")
        print(f"[Timing] avg save/plot per image  = {avg_save_ms:.3f} ms")
        print(f"[Timing] avg end-to-end per image = {avg_e2e_ms:.3f} ms")

    # ---- Switch back to original (non-EMA) parameters ----
    print("Switched back to original parameters")
    model_without_ddp.load_state_dict(original_state_dict)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


# ============================================================
# Backward-compatible function aliases
# ============================================================

# Map old names -> new names so that existing callers (e.g. main.py) continue to work
train_one_epoch_fh = train_one_epoch
evaluate_fh = evaluate_conditional_generation
