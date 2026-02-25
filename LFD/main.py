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

def normalize_cond_args(raw_cond):
    """
    Normalize --cond argument to a list of keys.
    Supports usages like:
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

class GeoDataset(Dataset):

    def __init__(
        self,
        data_path: str,
        target_key: str,
        cond_key: Union[str, List[str]], 
        random_crop: bool = False,
        random_flip: bool = False,
        normalize: bool = True,
        split: str = 'train',
        num_channels: int = 1
    ):

        self.target_key = target_key
        self.cond_key = cond_key
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.normalize = normalize
        self.split = split
        self.num_channels = num_channels 

        self.valid_keys = ['sx', 'rgt', 'fx', 'imp', 'hrz']

        self.npz_files = self._collect_files(data_path, split)
        
        if len(self.npz_files) == 0:
            raise ValueError(f"No NPZ files found in {data_path}")
        
        print(f"[GeoDataset] Loaded {len(self.npz_files)} {split} samples")
        print(f"[GeoDataset] Number of channels: {self.num_channels}")
        print(f"[GeoDataset] Target key: {self.target_key}")
        print(f"[GeoDataset] Condition key(s): {self.cond_key}")

    def _collect_files(self, data_path: str, split: str) -> List[str]:
        data_path = Path(data_path)
        split_dir = data_path / split

        npz_files = sorted(glob.glob(str(split_dir / "*.npz")))
        npz_files += sorted(glob.glob(str(split_dir / "*.NPZ")))

        # If not found in split_dir, do not fallback to root directory (root directory does not have npz either)
        return npz_files
    
    def __len__(self) -> int:
        return len(self.npz_files)

    def _load_single(self, npz_path: str, key: str):
        data = np.load(npz_path)
        array = data[key]

        if self.normalize:
            array = self._normalize(array, key=key)

        tensor = torch.from_numpy(array).float().unsqueeze(0)  # [1, H, W]
        return tensor

    def _load_data(self, npz_path: str, key: str):
        tensor = self._load_single(npz_path, key)  # [1,H,W]

        if self.num_channels > 1:
            tensor = tensor.repeat(self.num_channels, 1, 1)

        # label = torch.tensor(self.valid_keys.index(key)).long()
        label = torch.tensor(0).long()
        return tensor, label
    
    def _normalize(self, array: np.ndarray, key=None) -> np.ndarray:
        """Normalize to [-1, 1]"""
        
        if key == "fx":
            normalized = np.full_like(array, -1.0, dtype=np.float32)
            normalized[array!=0] = 1.0
            return normalized
        elif key == "rgt" or key == "hrz": 
            min_val = array.min()
            max_val = array.max()
            if max_val - min_val > 1e-6:
                array = (array - min_val) / (max_val - min_val) * 2 - 1
            else:
                array = np.full_like(array, -1.0, dtype=np.float32)
            return array

    def _load_hrz_as_rgt(self, npz_path: str):
        """
        Construct a "sparse RGT horizon map":
        - rgt: Full field RGT
        - hrz: Horizon pick mask / id
        Assign RGT values at locations where hrz has values;
        Background is -1 (convenient for use with bg_thresh=-1.0).
        Returns tensor: [1, H, W]
        """
        data = np.load(npz_path)
        rgt = data["rgt"].astype(np.float32)   # [H, W]
        hrz = data["hrz"].astype(np.float32)   # [H, W]

        # First normalize rgt (consistent with target)
        if self.normalize:
            rgt = self._normalize(rgt, key="rgt")

        # Fill background with -1.0, so horizon_loss can filter using bg_thresh=-1.0
        hrz_rgt = np.full_like(rgt, -1.0, dtype=np.float32)

        # Here the mask can be adjusted according to your hrz definition:
        # If hrz==0 means background, >0 is horizon; use >0
        mask = hrz > 0

        # Use RGT values at horizon locations
        hrz_rgt[mask] = rgt[mask]

        tensor = torch.from_numpy(hrz_rgt).float().unsqueeze(0)  # [1, H, W]
        return tensor
    
    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]

        # ---- target ----
        x, label = self._load_data(npz_path, self.target_key)  # [C_x, H, W]

        # ---- condition ----
        if isinstance(self.cond_key, (list, tuple)):
            cond_list = []
            for ck in self.cond_key:
                if ck == "hrz":
                    # Special handling: take rgt value at horizon location, -1 elsewhere
                    c = self._load_hrz_as_rgt(npz_path)   # [1, H, W]
                else:
                    c = self._load_single(npz_path, ck)   # [1, H, W]
                cond_list.append(c)
            cond = torch.cat(cond_list, dim=0)            # [C_cond, H, W]
        else:
            if self.cond_key == "hrz":
                cond = self._load_hrz_as_rgt(npz_path)
            else:
                cond = self._load_single(npz_path, self.cond_key)  # [1, H, W]
                if self.num_channels > 1:
                    cond = cond.repeat(self.num_channels, 1, 1)

        if self.random_flip:
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=[2]) 
                cond = torch.flip(cond, dims=[2])

        return x, label, cond

def get_args_parser():
    parser = argparse.ArgumentParser('LFD', add_help=False)

    # architecture
    parser.add_argument('--model', default='LFD-B/32', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=512, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # training
    parser.add_argument('--epochs', default=1200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1, type=int)

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')
    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of input channels')
    parser.add_argument('--cond_in_ch', default=1, type=int,
                        help='Number of condition channels for ControlNet branch')
    parser.add_argument('--pretrained_base', default='', type=str,
                        help='path to checkpoint-last.pth')
    parser.add_argument('--cond', nargs='+', default=['fx','hrz'],
                        help='Condition keys for ControlNet branch, e.g., fx hrz')
    parser.add_argument('--target', default='rgt', type=str,
                        help='Condition key for ControlNet branch')
    return parser


def main(args):
    torch.set_float32_matmul_precision('high')

    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # # Data augmentation transforms
    # transform_train = transforms.Compose([
    #     transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.PILToTensor()
    # ])

    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    # print(dataset_train)

    dataset_train = GeoDataset(
        data_path=args.data_path,
        target_key=args.target,
        cond_key=args.cond,
        random_crop=False,  # Enable random crop as needed
        random_flip=True,  # Enable random flip as needed
        normalize=True,  # Enable normalization
        split='train',  # Load training set
        num_channels=1  # Single channel data
    )

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    print("Base lr: {:.2e}".format(args.lr * 256 / eff_batch_size))
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # Resume from checkpoint if provided
    checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth") if args.resume else None
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        print("Resumed checkpoint from", args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        print("Training from scratch")

    # Evaluate generation (use validation fx as condition)
    if args.evaluate_gen:
        print("Evaluating checkpoint at {} epoch".format(args.start_epoch))

        # —— Use validation set as condition —— 
        dataset_val = GeoDataset(
            data_path=args.data_path,
            target_key=args.target,   # 'rgt'
            cond_key=args.cond,       # 'fx'
            random_crop=False,
            random_flip=False,
            normalize=True,
            split='valid',            # ★ Change to your validation set subdirectory name: valid / val / test
            num_channels=1,
        )

        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_val =", sampler_val)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.gen_bsz,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        with torch.random.fork_rng():
            torch.manual_seed(seed)
            with torch.no_grad():
                # New version evaluate_cond accepts a loader
                evaluate_fh(model_without_ddp, args, epoch=0,
                              val_loader=data_loader_val,
                              log_writer=log_writer)
        return

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch_fh(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if epoch % 100 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate_fh(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    args.cond = normalize_cond_args(args.cond)
    print(f"Parsed cond keys: {args.cond}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
