import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.misc as misc

import copy
from engine import train_one_epoch_fh, evaluate_fh

from denoiser import DenoiserFH as Denoiser
from torch.utils.data import Dataset
from typing import List, Union
import glob


def parse_condition_keys(raw_cond_arg):
    """
    Parse the --cond command-line argument into a list of key strings.

    Supports the following input formats:
        --cond fx hrz           -> ['fx', 'hrz']  (multiple separate args)
        --cond "fx,hrz"         -> ['fx', 'hrz']  (comma-separated single string)
        CONDITION="fx","hrz"    -> ['fx', 'hrz']  (shell variable expansion)
        --cond "[fx, hrz]"      -> ['fx', 'hrz']  (bracket-wrapped format)

    Args:
        raw_cond_arg: Raw argument received by nargs='+' (list or str)

    Returns:
        List[str] of condition key strings
    """
    if isinstance(raw_cond_arg, list):
        if len(raw_cond_arg) == 1:
            # Single string element; may be "fx,hrz" or "[fx,hrz]" format
            single_str = raw_cond_arg[0]
        else:
            # Already a multi-element list; return as-is
            return raw_cond_arg
    else:
        single_str = raw_cond_arg

    single_str = single_str.strip()

    # Strip optional square brackets
    if single_str.startswith('[') and single_str.endswith(']'):
        single_str = single_str[1:-1]

    # Remove quotation marks
    single_str = single_str.replace('"', '').replace("'", '')

    # Normalize separators and split
    parsed_keys = []
    for key_token in single_str.replace(',', ' ').split():
        key_token = key_token.strip()
        if key_token:
            parsed_keys.append(key_token)

    return parsed_keys


class GeoSeismicDataset(Dataset):
    """
    Geological seismic dataset that loads training/validation samples from NPZ files.

    Each NPZ file contains multiple geological arrays, such as:
        - 'rgt' : Relative Geologic Time (training target)
        - 'fx'  : Fault map (condition input)
        - 'hrz' : Horizon picks (condition input)
        - 'sx'  : Seismic amplitude (optional condition)
        - 'imp' : Acoustic impedance (optional condition)

    All arrays are normalized to [-1, 1]. Optional random horizontal flip augmentation.
    """

    def __init__(
        self,
        data_root: str,
        target_key: str,
        condition_keys: Union[str, List[str]],
        use_random_crop: bool = False,
        use_random_flip: bool = False,
        normalize: bool = True,
        split: str = 'train',
        num_channels: int = 1
    ):
        """
        Args:
            data_root:       Dataset root directory (expects train/ and valid/ subdirectories)
            target_key:      NPZ key for the prediction target (e.g., 'rgt')
            condition_keys:  NPZ key or list of keys for condition inputs (e.g., ['fx', 'hrz'])
            use_random_crop: Whether to apply random cropping (reserved; not yet implemented)
            use_random_flip: Whether to apply random horizontal flip augmentation
            normalize:       Whether to normalize data to [-1, 1]
            split:           Data split, either 'train' or 'valid'
            num_channels:    Number of input channels (1 = single-channel grayscale)
        """
        self.target_key = target_key
        self.condition_keys = condition_keys
        self.use_random_crop = use_random_crop
        self.use_random_flip = use_random_flip
        self.normalize = normalize
        self.split = split
        self.num_channels = num_channels

        # All supported geological data keys
        self.supported_data_keys = ['sx', 'rgt', 'fx', 'imp', 'hrz']

        # Collect all NPZ file paths
        self.npz_file_paths = self._collect_npz_files(data_root, split)

        if len(self.npz_file_paths) == 0:
            raise ValueError(f"No NPZ files found in {data_root}/{split}")

        print(f"[GeoSeismicDataset] Loaded {len(self.npz_file_paths)} {split} samples")
        print(f"[GeoSeismicDataset] Number of channels: {self.num_channels}")
        print(f"[GeoSeismicDataset] Target key: {self.target_key}")
        print(f"[GeoSeismicDataset] Condition key(s): {self.condition_keys}")

    def _collect_npz_files(self, data_root: str, split: str) -> List[str]:
        """
        Collect all NPZ file paths under the specified split directory.
        Supports both .npz and .NPZ file extensions.
        """
        split_dir = Path(data_root) / split
        file_paths = sorted(glob.glob(str(split_dir / "*.npz")))
        file_paths += sorted(glob.glob(str(split_dir / "*.NPZ")))
        return file_paths

    def __len__(self) -> int:
        return len(self.npz_file_paths)

    def _load_and_normalize_array(self, npz_path: str, key: str) -> torch.Tensor:
        """
        Load a single array from an NPZ file, optionally normalize it,
        and return as a (1, H, W) float Tensor.
        """
        npz_data = np.load(npz_path)
        array = npz_data[key]

        if self.normalize:
            array = self._normalize_to_minus_one_one(array, key=key)

        tensor = torch.from_numpy(array).float().unsqueeze(0)  # (1, H, W)
        return tensor

    def _load_target_with_channel_expand(self, npz_path: str, key: str):
        """
        Load the target array and optionally expand to num_channels.

        Returns:
            tensor: (num_channels, H, W)
            label:  class label tensor (fixed to 0 for single-class tasks)
        """
        tensor = self._load_and_normalize_array(npz_path, key)  # (1, H, W)

        if self.num_channels > 1:
            tensor = tensor.repeat(self.num_channels, 1, 1)

        label = torch.tensor(0).long()  # Single-class task; label always 0
        return tensor, label

    def _normalize_to_minus_one_one(self, array: np.ndarray, key=None) -> np.ndarray:
        """
        Normalize an array to [-1, 1] using key-specific strategies.

        - 'fx' (fault):           non-zero -> +1, zero -> -1 (binarization)
        - 'rgt' (relative time):  min-max linear stretch to [-1, 1]
        - 'hrz' (horizon):        min-max linear stretch to [-1, 1]
        """
        if key == "fx":
            # Fault binarization: fault -> +1, background -> -1
            normalized = np.full_like(array, -1.0, dtype=np.float32)
            normalized[array != 0] = 1.0
            return normalized
        elif key in ("rgt", "hrz"):
            # Linear min-max stretch to [-1, 1]
            min_val = array.min()
            max_val = array.max()
            if max_val - min_val > 1e-6:
                array = (array - min_val) / (max_val - min_val) * 2 - 1
            else:
                array = np.full_like(array, -1.0, dtype=np.float32)
            return array

    def _load_horizon_as_sparse_rgt(self, npz_path: str) -> torch.Tensor:
        """
        Build a sparse RGT horizon map from 'rgt' and 'hrz' arrays:
        - Pixels with valid horizon labels receive the corresponding RGT value
        - All other pixels are set to -1 (background / no constraint)

        This representation converts horizon annotations into sparse RGT supervision
        compatible with the horizon_loss (bg_threshold = -1.0).

        Returns:
            Tensor of shape (1, H, W)
        """
        npz_data = np.load(npz_path)
        rgt_array = npz_data["rgt"].astype(np.float32)   # (H, W)
        hrz_array = npz_data["hrz"].astype(np.float32)   # (H, W), horizon id map

        # Normalize RGT consistent with the training target
        if self.normalize:
            rgt_array = self._normalize_to_minus_one_one(rgt_array, key="rgt")

        # Initialize sparse map to -1 (no constraint everywhere)
        sparse_rgt_map = np.full_like(rgt_array, -1.0, dtype=np.float32)

        # horizon id > 0 indicates a valid horizon annotation
        horizon_valid_mask = hrz_array > 0
        # Fill valid horizon positions with the corresponding RGT value
        sparse_rgt_map[horizon_valid_mask] = rgt_array[horizon_valid_mask]

        return torch.from_numpy(sparse_rgt_map).float().unsqueeze(0)  # (1, H, W)

    def __getitem__(self, idx):
        """
        Returns a single sample:
            target_tensor:    (num_channels, H, W), normalized target image
            class_label:      scalar long tensor, class label
            condition_tensor: (C_cond, H, W), condition maps; C_cond = len(condition_keys)
        """
        npz_path = self.npz_file_paths[idx]

        # ---- Load target ----
        target_tensor, class_label = self._load_target_with_channel_expand(
            npz_path, self.target_key
        )

        # ---- Load conditions ----
        if isinstance(self.condition_keys, (list, tuple)):
            # Multiple conditions: load individually and concatenate along channel dim
            single_condition_list = []
            for cond_key in self.condition_keys:
                if cond_key == "hrz":
                    # Special handling: convert horizon picks to sparse RGT map
                    cond_tensor = self._load_horizon_as_sparse_rgt(npz_path)  # (1, H, W)
                else:
                    cond_tensor = self._load_and_normalize_array(npz_path, cond_key)  # (1, H, W)
                single_condition_list.append(cond_tensor)
            condition_tensor = torch.cat(single_condition_list, dim=0)  # (C_cond, H, W)
        else:
            # Single condition
            if self.condition_keys == "hrz":
                condition_tensor = self._load_horizon_as_sparse_rgt(npz_path)
            else:
                condition_tensor = self._load_and_normalize_array(npz_path, self.condition_keys)
                if self.num_channels > 1:
                    condition_tensor = condition_tensor.repeat(self.num_channels, 1, 1)

        # ---- Random horizontal flip augmentation (50% probability) ----
        if self.use_random_flip:
            if torch.rand(1) < 0.5:
                target_tensor = torch.flip(target_tensor, dims=[2])
                condition_tensor = torch.flip(condition_tensor, dims=[2])

        return target_tensor, class_label, condition_tensor


def build_argument_parser():
    """
    Build the command-line argument parser covering model architecture,
    training hyperparameters, ODE sampling, dataset, checkpointing,
    and distributed training settings.
    """
    parser = argparse.ArgumentParser('LFD', add_help=False)

    # ---- Model architecture ----
    parser.add_argument('--model', default='LFD-B/32', type=str, metavar='MODEL',
                        help='Model variant name (e.g., LFD-B/32)')
    parser.add_argument('--img_size', default=512, type=int,
                        help='Input image size (square)')
    parser.add_argument('--attn_dropout', type=float, default=0.0,
                        help='Attention weight dropout probability')
    parser.add_argument('--proj_dropout', type=float, default=0.0,
                        help='Projection layer dropout probability')

    # ---- Training hyperparameters ----
    parser.add_argument('--epochs', default=1200, type=int,
                        help='Total number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='Number of epochs for linear LR warmup')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Per-GPU batch size (effective batch = batch_size * num_GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Absolute learning rate (derived from blr if not set)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='Base learning rate: absolute_lr = blr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cosine annealing schedule')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='LR schedule strategy: "constant" or "cosine"')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay coefficient (bias and norm layers are exempt)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='First EMA decay rate (used by default during sampling)')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='Second EMA decay rate (backup)')
    parser.add_argument('--P_mean', default=-0.8, type=float,
                        help='Mean of logit-normal timestep sampling distribution')
    parser.add_argument('--P_std', default=0.8, type=float,
                        help='Std of logit-normal timestep sampling distribution')
    parser.add_argument('--noise_scale', default=1.0, type=float,
                        help='Initial noise scaling factor')
    parser.add_argument('--t_eps', default=5e-2, type=float,
                        help='Timestep lower bound (prevents division by zero)')
    parser.add_argument('--label_drop_prob', default=0.1, type=float,
                        help='CFG label dropout probability during training')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed (each process adds its rank for diversity)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch index (used when resuming training)')
    parser.add_argument('--num_workers', default=12, type=int,
                        help='Number of DataLoader worker processes')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # ---- ODE sampling parameters ----
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE integration method: "euler" or "heun"')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Number of ODE integration steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-Free Guidance scale')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='Lower bound of the timestep interval for CFG')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='Upper bound of the timestep interval for CFG')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Maximum number of images to generate during evaluation')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Online evaluation frequency (every N epochs)')
    parser.add_argument('--online_eval', action='store_true',
                        help='Enable online evaluation interleaved with training')
    parser.add_argument('--evaluate_gen', action='store_true',
                        help='Run generation-only evaluation (no training); requires --resume')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Batch size for generation during evaluation')

    # ---- Dataset ----
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Dataset root directory (expects train/ and valid/ subdirectories)')
    parser.add_argument('--class_num', default=1, type=int,
                        help='Number of classes (typically 1 for geological tasks)')

    # ---- Checkpointing ----
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Output directory for checkpoints and TensorBoard logs')
    parser.add_argument('--resume', default='',
                        help='Directory containing checkpoint-last.pth to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (epochs) for saving checkpoint-last.pth (rolling)')
    parser.add_argument('--log_freq', default=100, type=int,
                        help='TensorBoard logging frequency (every N steps)')
    parser.add_argument('--device', default='cuda',
                        help='Training / testing device: "cuda" or "cpu"')

    # ---- Distributed training ----
    parser.add_argument('--world_size', default=1, type=int,
                        help='Total number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Local rank of the current process')
    parser.add_argument('--dist_on_itp', action='store_true',
                        help='Use ITP cluster distributed mode (OMPI env vars)')
    parser.add_argument('--dist_url', default='env://',
                        help='init_method URL for distributed training')

    # ---- Model input / condition settings ----
    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of input channels for the target image')
    parser.add_argument('--cond_in_ch', default=1, type=int,
                        help='Number of channels per condition input (ControlNet branch)')
    parser.add_argument('--pretrained_base', default='', type=str,
                        help='Path to a pretrained checkpoint for transfer learning')
    parser.add_argument('--cond', nargs='+', default=['fx', 'hrz'],
                        help='Condition key list, e.g., fx hrz')
    parser.add_argument('--target', default='rgt', type=str,
                        help='NPZ key of the training target (e.g., rgt)')

    return parser


def main(args):
    """
    Main training / evaluation function.

    Workflow:
    1. Initialize distributed environment
    2. Build training dataset and DataLoader
    3. Build DenoiserFH model and wrap with DDP
    4. Configure optimizer and learning rate schedule
    5. Optionally resume from checkpoint or initialize EMA from scratch
    6. Run training loop with periodic checkpointing and optional online evaluation
    """
    torch.set_float32_matmul_precision('high')

    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Each process uses a distinct random seed (seed + rank)
    per_process_seed = args.seed + misc.get_rank()
    torch.manual_seed(per_process_seed)
    np.random.seed(per_process_seed)

    cudnn.benchmark = True  # Enable cuDNN auto-tuner for fixed input sizes

    num_distributed_tasks = misc.get_world_size()
    current_process_rank = misc.get_rank()

    # ---- Initialize TensorBoard (main process only) ----
    if current_process_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        tensorboard_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        tensorboard_writer = None

    # ---- Build training dataset ----
    train_dataset = GeoSeismicDataset(
        data_root=args.data_path,
        target_key=args.target,
        condition_keys=args.cond,
        use_random_crop=False,
        use_random_flip=True,    # Enable random horizontal flip augmentation
        normalize=True,
        split='train',
        num_channels=1
    )

    # Distributed sampler: each process receives a distinct data subset
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=num_distributed_tasks,
        rank=current_process_rank,
        shuffle=True
    )
    print("Train sampler:", train_sampler)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    # ---- torch.compile configuration ----
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # ---- Build denoiser model ----
    denoiser_model = Denoiser(args)
    print("Model:", denoiser_model)
    num_trainable_params = sum(p.numel() for p in denoiser_model.parameters() if p.requires_grad)
    print("Trainable parameters: {:.6f}M".format(num_trainable_params / 1e6))

    denoiser_model.to(device)

    # ---- Compute effective learning rate ----
    effective_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:
        # Linear scaling rule: absolute_lr = blr * total_batch_size / 256
        args.lr = args.blr * effective_batch_size / 256

    print("Base lr (blr): {:.2e}".format(args.lr * 256 / effective_batch_size))
    print("Actual lr:     {:.2e}".format(args.lr))
    print("Effective batch size: %d" % effective_batch_size)

    # ---- Wrap with DDP ----
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        denoiser_model, device_ids=[args.gpu]
    )
    model_core = ddp_model.module  # Unwrapped model (for EMA updates, saving, etc.)

    # ---- Optimizer (bias and norm layers excluded from weight decay) ----
    param_groups = misc.add_weight_decay(model_core, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # ---- Load checkpoint or initialize EMA from scratch ----
    checkpoint_file = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_file and os.path.exists(checkpoint_file):
        # Resume: load model weights, optimizer state, and EMA parameters
        saved_checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        model_core.load_state_dict(saved_checkpoint['model'])

        ema_state1 = saved_checkpoint['model_ema1']
        ema_state2 = saved_checkpoint['model_ema2']
        # Load EMA parameters to GPU
        model_core.ema_params1 = [ema_state1[name].cuda() for name, _ in model_core.named_parameters()]
        model_core.ema_params2 = [ema_state2[name].cuda() for name, _ in model_core.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in saved_checkpoint and 'epoch' in saved_checkpoint:
            optimizer.load_state_dict(saved_checkpoint['optimizer'])
            args.start_epoch = saved_checkpoint['epoch'] + 1
            print("Loaded optimizer state; resuming from epoch {}".format(args.start_epoch))
        del saved_checkpoint
    else:
        # Train from scratch: initialize EMA from current parameters
        model_core.ema_params1 = copy.deepcopy(list(model_core.parameters()))
        model_core.ema_params2 = copy.deepcopy(list(model_core.parameters()))
        print("Training from scratch")

    # ---- Generation-only evaluation mode ----
    if args.evaluate_gen:
        print("Evaluating checkpoint at epoch {}".format(args.start_epoch))

        # Use the validation set as condition input
        val_dataset = GeoSeismicDataset(
            data_root=args.data_path,
            target_key=args.target,
            condition_keys=args.cond,
            use_random_crop=False,
            use_random_flip=False,
            normalize=True,
            split='valid',  # Must match the validation subdirectory name in your dataset
            num_channels=1,
        )

        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset,
            num_replicas=num_distributed_tasks,
            rank=current_process_rank,
            shuffle=True
        )
        print("Val sampler:", val_sampler)

        val_data_loader = torch.utils.data.DataLoader(
            val_dataset,
            sampler=val_sampler,
            batch_size=args.gen_bsz,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        # Evaluate with a fixed random seed for reproducibility
        with torch.random.fork_rng():
            torch.manual_seed(per_process_seed)
            with torch.no_grad():
                evaluate_fh(
                    model_core, args, epoch=0,
                    val_loader=val_data_loader,
                    log_writer=tensorboard_writer
                )
        return

    # ---- Main training loop ----
    print(f"Start training for {args.epochs} epochs")
    training_start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        # Set different sampler seed each epoch for data order diversity
        if args.distributed:
            train_data_loader.sampler.set_epoch(epoch)

        # Train for one epoch
        train_one_epoch_fh(
            ddp_model, model_core, train_data_loader,
            optimizer, device, epoch,
            log_writer=tensorboard_writer, args=args
        )

        # ---- Periodic checkpoint (checkpoint-last.pth, rolling overwrite) ----
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_core,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        # ---- Numbered checkpoint every 100 epochs (for rollback) ----
        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_core,
                optimizer=optimizer,
                epoch=epoch
            )

        # ---- Online evaluation (triggered at eval_freq intervals) ----
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate_fh(
                    model_core, args, epoch,
                    batch_size=args.gen_bsz,
                    log_writer=tensorboard_writer
                )
            torch.cuda.empty_cache()

        # Flush TensorBoard on main process
        if misc.is_main_process() and tensorboard_writer is not None:
            tensorboard_writer.flush()

    total_training_time = time.time() - training_start_time
    total_training_time_str = str(datetime.timedelta(seconds=int(total_training_time)))
    print('Total training time:', total_training_time_str)


if __name__ == '__main__':
    args = build_argument_parser().parse_args()
    # Normalize --cond argument to a standard List[str]
    args.cond = parse_condition_keys(args.cond)
    print(f"Parsed condition keys: {args.cond}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
