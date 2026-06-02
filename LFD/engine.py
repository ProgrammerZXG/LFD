import math
import sys
import os
import shutil
import copy
import time
import contextlib

import torch
import numpy as np
# import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import copy
import matplotlib
matplotlib.use("Agg")  # important on servers / no-display environments
import matplotlib.pyplot as plt
import util.misc as misc
from matplotlib.colors import ListedColormap
import scipy.ndimage

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

def getFxColor():
    return ListedColormap(["#a6172d"]) 

def getStrataColors(alpha=1, FillExceptMin=False, reverse=False):
    rgba = np.full((256,4), 0, dtype=np.float32)

    strata = np.array([
        [1.0,0.0,0.0,alpha],[1.0,0.5019608,0.0,alpha],[1.0,1.0,0.0,alpha],
        [0.0,1,0.0,alpha],[0.0,0.5019608,0.0,alpha],[0.0,0.2509804,0.0,alpha],
        [0,1.0,1.0,alpha],[0.0,0.5019608,1.0,alpha],[0.0,0.0,1.0,alpha],
        [0.0,0.0,0.627451,alpha],[0.0,0.5019608,0.7529412,alpha],[1.0,0.5019608,0.5019608,alpha],
        [0.5019608,0.5019608,1.0,alpha],[0.5019608,0.0,1.0,alpha],[0.5019608,0,0.5019608,alpha],
        [1.0,0.5019608,1.0,alpha],[1.0,0.0,1.0,alpha],[0.5019608,0.2509804,0,alpha],
        # Original gray 1
        [0.5019608,0.5019608,0.5019608,alpha],
        # Original gray 2
        [0.7529412,0.7529412,0.7529412,alpha],
        [0.2509804,0,0.2509804,alpha],
        [0.90588236,0.7294118,0.19607843,alpha],[0.44313726,0.58431375,0.58431375,alpha],[0.5254902,0.42352942,0.4862745,alpha],
        [0.7176471,0.54509807,0.44313726,alpha],[0.5019608,0.5019608,0,alpha],[0.7529412,0.7294118,0.8784314,alpha],
        [0.61960787,0.85882354,0.9882353,alpha],[0.7372549,0.25882354,0.24705882,alpha],[0.8862745,0.8509804,0.627451,alpha],
        [0.60784316,0.9411765,0.7490196,alpha],[0.62352943,0.79607844,0.105882354,alpha]
    ], dtype=np.float32)

    # --- Replace gray with specified colors ---
    c0 = np.array([0xFD/255.0, 0xC2/255.0, 0x3E/255.0], dtype=np.float32)  # #6A6A6A
    c1 = np.array([0x6A/255.0, 0xAF/255.0, 0xE6/255.0], dtype=np.float32)  # #6AAFE6
    c2 = np.array([0x67/255.0, 0xD5/255.0, 0xB5/255.0], dtype=np.float32)  # #519D9E
    strata[2, :3] = c0
    strata[18, :3] = c1
    strata[19, :3] = c2
    # --------------------------------

    for i in range(32):
        rgba[i*8:(i+1)*8] = strata[i]

    if FillExceptMin:
        rgba[0, -1] = 0
    if reverse:
        rgba = np.flip(rgba, axis=0)

    return ListedColormap(rgba)

def train_one_epoch_fh(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_v', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_h', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_b', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels, cond) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = x.to(device)
        labels = labels.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss, loss_v, loss_h, loss_b = model(x, labels, cond)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(loss_v=loss_v.item())
        metric_logger.update(loss_h=loss_h.item())
        metric_logger.update(loss_b=loss_b.item())

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
                log_writer.add_scalar('loss_v', loss_v.item(), epoch_1000x)
                log_writer.add_scalar('loss_h', loss_h.item(), epoch_1000x)
                log_writer.add_scalar('loss_b', loss_b.item(), epoch_1000x)

def evaluate_fh(model_without_ddp, args, epoch, val_loader, log_writer=None):
    """
    Conditional generation evaluation with multi-channel conditions.

    val_loader: yields (x, labels, cond)
        - x:     [B, C_x, H, W]  (target, e.g., rgt, not used in pure generation)
        - labels:[B]
        - cond:  [B, C_cond, H, W], C_cond = len(args.cond)
    """
    device = torch.device(args.device)
    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()

    max_images = args.num_images

    # normalize cond keys: ensure list[str]
    if isinstance(args.cond, (list, tuple)):
        cond_keys = list(args.cond)
    else:
        cond_keys = [args.cond]
    cond_tag = "+".join(cond_keys)

    # save folder
    save_folder = os.path.join(
        "ssd/tmp",
        args.output_dir,
        "condVAL-steps{}-max{}-res{}_{}".format(
            model_without_ddp.steps,
            max_images, args.img_size, cond_tag
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # ---- switch to EMA params (ema1) ----
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    def get_cond_cmap(cond_name: str) -> str:
        # mapping for condition types
        if cond_name == "fx":
            return getFxColor()
        elif cond_name == "sx":
            return "gray"
        elif cond_name == "rgt":
            return getStrataColors()
        elif cond_name == "imp":
            return "jet"
        elif cond_name == "hrz":
            return getStrataColors()
        else:
            return "gray"

    # ==========================
    # Timing stats (per-rank)
    # ==========================
    gen_time_s = 0.0     # only model.generate()
    save_time_s = 0.0    # matplotlib + savefig
    num_imgs = 0         # images generated on this rank

    def _sync_cuda():
        if device.type == "cuda":
            torch.cuda.synchronize()

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    img_global = 0
    c = 0

    for batch_idx, (x, labels, cond) in enumerate(val_loader):
        c += 1
        if img_global >= max_images:
            break

        x = x.to(device, non_blocking=True)         # [B, C_x, H, W]
        cond = cond.to(device, non_blocking=True)   # [B, C_cond, H, W]
        labels = labels.to(device, non_blocking=True)

        bsz = cond.size(0)
        if img_global + bsz > max_images:
            keep = max_images - img_global
            cond = cond[:keep]
            labels = labels[:keep]
            x = x[:keep]
            bsz = keep

        # ---- generate with condition (timed) ----
        _sync_cuda()
        t0 = time.perf_counter()
        with autocast_ctx:
            sampled_images = model_without_ddp.generate(labels, cond)
        _sync_cuda()
        t1 = time.perf_counter()

        gen_time_s += (t1 - t0)
        num_imgs += bsz

        # move to cpu (not counted as "generate()" time above)
        sampled_images = sampled_images.detach().cpu()
        cond_cpu = cond.detach().cpu()

        B, C_cond, H, W = cond_cpu.shape
        assert C_cond == len(cond_keys), \
            f"cond channels ({C_cond}) != len(cond_keys) ({len(cond_keys)})"

        n_cols = C_cond + 1  # e.g. fx, hrz, pred

        # ---- plotting/saving time (timed) ----
        t2 = time.perf_counter()

        for b_id in range(bsz):
            img_id = img_global + b_id

            # ---- Prediction Processing ----
            # Extract sample as [1, C, H, W] numpy array
            img_4d = sampled_images[b_id].unsqueeze(0).float().numpy()
            img_4d = smooth_prediction(img_4d, sigma=1)
            # Use FAULT channel (index 0) from cond_cpu
            img_4d = replace_fault_values_with_nearest(
                img_4d,
                fault_np=cond_cpu[b_id, 0].numpy(), 
                thr=-0.999,
                dilate=1
            )
            # Flatten to [H, W]
            if img_4d.shape[1] == 1:
                img_np = img_4d[0, 0]
            else:
                img_np = img_4d[0, 0]

            pred_cmap = getStrataColors()

            # ---- figure: top..bottom = fx, hrz, pred ----
            # n_cols here is actually n_rows for us
            n_rows = n_cols 
            plt.figure(figsize=(4, 4 * n_rows))

            # plot each condition channel
            for c_idx, cond_name in enumerate(cond_keys):
                cond_tensor = cond_cpu[b_id, c_idx]   # [H, W]
                cond_np = cond_tensor.numpy()
                cmap = plt.get_cmap(get_cond_cmap(cond_name)).copy()
                cmap.set_bad('#f0f5f9')

                # If it's the second channel (c_idx == 1), apply dilation
                if c_idx == 1:
                    import scipy.ndimage
                    # 1. Backup original data for color picking
                    original_cond = cond_np.copy()
                    
                    # 2. Compute dilation mask (dilate only foreground)
                    # Assume > -0.9 is foreground
                    mask = (cond_np > -0.95)
                    dilated_mask = scipy.ndimage.binary_dilation(mask, iterations=2)
                    
                    # 3. Construct new data for display
                    # Background (-1) remains unchanged
                    new_cond = np.full_like(cond_np, -1.0)

                    cond_np = scipy.ndimage.maximum_filter(cond_np, size=3) # size=3 is equivalent to dilation of 1 unit
                    
                    # Reset cmap, ensuring background is gray
                    cmap = plt.get_cmap(get_cond_cmap(cond_name)).copy()
                    cmap.set_bad('#f0f5f9')

                plt.subplot(n_rows, 1, c_idx + 1)
                plt.imshow(np.ma.masked_where(cond_np < -0.95, cond_np), cmap=cmap, vmin=-1.0, vmax=1.0)
                plt.axis("off")
                # plt.title(f"cond: {cond_name}", fontsize=10)

            # last row: prediction
            plt.subplot(n_rows, 1, n_rows)
            plt.imshow(img_np, cmap=pred_cmap, vmin=-1.0, vmax=1.0)
            plt.axis("off")
            # plt.title("pred", fontsize=10)

            plt.tight_layout(pad=0.1)

            fname = (
                f"rank{local_rank}_img{str(img_id).zfill(6)}_"
                f"cond-{'-'.join(cond_keys)}.png"
            )
            plt.savefig(os.path.join(save_folder, fname), dpi=150)
            plt.close()

        t3 = time.perf_counter()
        save_time_s += (t3 - t2)

        img_global += bsz

    # barrier (only if DDP initialized)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # ---- Aggregate time and image count across ranks, and print average per image ----
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = torch.tensor([gen_time_s, save_time_s, float(num_imgs)],
                         device=device, dtype=torch.float64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        gen_time_all, save_time_all, num_imgs_all = t.tolist()
    else:
        gen_time_all, save_time_all, num_imgs_all = gen_time_s, save_time_s, float(num_imgs)

    if misc.get_rank() == 0 and num_imgs_all > 0:
        avg_gen_ms = 1000.0 * gen_time_all / num_imgs_all
        avg_save_ms = 1000.0 * save_time_all / num_imgs_all
        avg_e2e_ms = avg_gen_ms + avg_save_ms

        print(f"[Timing] images total = {int(num_imgs_all)}")
        print(f"[Timing] avg generate() per image = {avg_gen_ms:.3f} ms")
        print(f"[Timing] avg save/plot per image  = {avg_save_ms:.3f} ms")
        print(f"[Timing] avg end-to-end per image = {avg_e2e_ms:.3f} ms")

    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
