import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import copy

import torch
import torch.distributed as dist


# ============================================================
# Metric tracking utilities
# ============================================================

class SmoothedValue(object):
    """
    Tracks a series of scalar values and provides smoothed statistics
    over a sliding window (median, mean, global average, etc.).
    Commonly used to display smoothed loss and learning rate during training.
    """

    def __init__(self, window_size=20, fmt=None):
        """
        Args:
            window_size: Size of the sliding window (used for median and window average).
            fmt:         Format string supporting {median}, {avg}, {global_avg}, {max}, {value}.
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # sliding window buffer
        self.total = 0.0   # cumulative sum over all updates
        self.count = 0     # cumulative number of samples
        self.fmt = fmt

    def update(self, value, n=1):
        """Add a new value; n specifies the number of samples this value represents."""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Synchronize count and total across all distributed processes
        (used for global statistics in distributed training).
        Note: the sliding window deque is NOT synchronized — window stats remain local.
        """
        if not is_dist_avail_and_initialized():
            return
        sync_tensor = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(sync_tensor)
        sync_tensor = sync_tensor.tolist()
        self.count = int(sync_tensor[0])
        self.total = sync_tensor[1]

    @property
    def median(self):
        """Median of values in the current sliding window."""
        window_tensor = torch.tensor(list(self.deque))
        if window_tensor.numel() == 0:
            return 0.0
        return window_tensor.median().item()

    @property
    def avg(self):
        """Arithmetic mean of values in the current sliding window."""
        window_tensor = torch.tensor(list(self.deque), dtype=torch.float32)
        if window_tensor.numel() == 0:
            return 0.0
        return window_tensor.mean().item()

    @property
    def global_avg(self):
        """Global average over all values seen since initialization."""
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def max(self):
        """Maximum value in the current sliding window."""
        return max(self.deque) if len(self.deque) > 0 else 0.0

    @property
    def value(self):
        """Most recently added value."""
        return self.deque[-1] if len(self.deque) > 0 else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


class MetricLogger(object):
    """
    Training metric logger.
    Manages multiple SmoothedValue trackers, supports formatted printing
    and distributed synchronization.
    """

    def __init__(self, delimiter="\t"):
        """
        Args:
            delimiter: Separator used when printing multiple metrics side by side.
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        Update one or more metric values.
        Accepts both torch.Tensor and Python numeric types.
        """
        for metric_name, metric_value in kwargs.items():
            if metric_value is None:
                continue
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()
            assert isinstance(metric_value, (float, int))
            self.meters[metric_name].update(metric_value)

    def __getattr__(self, attr):
        """Allow attribute-style access to meters, e.g. logger.loss."""
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        """Format all tracked metrics as a single string."""
        metric_strings = []
        for name, meter in self.meters.items():
            metric_strings.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(metric_strings)

    def synchronize_between_processes(self):
        """Synchronize all meters across distributed processes."""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """Manually register a new metric tracker."""
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Iterator wrapper that periodically prints training progress
        (including ETA, elapsed time, and GPU memory usage).

        Args:
            iterable:   DataLoader or any iterable to wrap.
            print_freq: Number of steps between each log print.
            header:     Log line prefix (e.g., 'Epoch: [5]').

        Yields:
            Each element from iterable (transparent pass-through).
        """
        iteration_idx = 0
        if not header:
            header = ''
        loop_start_time = time.time()
        step_end_time = time.time()

        # Track per-step time and data loading time
        iter_time_meter = SmoothedValue(fmt='{avg:.4f}')
        data_time_meter = SmoothedValue(fmt='{avg:.4f}')

        # Build log format string
        step_fmt_width = ':' + str(len(str(len(iterable)))) + 'd'
        log_parts = [
            header,
            '[{0' + step_fmt_width + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_parts.append('max mem: {memory:.0f}')
        log_format_str = self.delimiter.join(log_parts)

        MB = 1024.0 * 1024.0  # bytes to megabytes conversion factor

        for batch_data in iterable:
            data_time_meter.update(time.time() - step_end_time)
            yield batch_data  # pass data through to the caller
            iter_time_meter.update(time.time() - step_end_time)

            # Print log every print_freq steps or at the final step
            if iteration_idx % print_freq == 0 or iteration_idx == len(iterable) - 1:
                remaining_steps = len(iterable) - iteration_idx
                eta_seconds = iter_time_meter.global_avg * remaining_steps
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(log_format_str.format(
                        iteration_idx, len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time_meter),
                        data=str(data_time_meter),
                        memory=torch.cuda.max_memory_allocated() / MB
                    ))
                else:
                    print(log_format_str.format(
                        iteration_idx, len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time_meter),
                        data=str(data_time_meter)
                    ))

            iteration_idx += 1
            step_end_time = time.time()

        total_elapsed_time = time.time() - loop_start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_elapsed_time)))
        print('{} Total time: {} ({:.4f} s / step)'.format(
            header, total_time_str, total_elapsed_time / len(iterable)
        ))


# ============================================================
# Distributed training utilities
# ============================================================

def setup_distributed_print(is_master_process):
    """
    Configure the print function so that only the master process (rank 0)
    emits log output, with a timestamp prepended to each line.
    Non-master processes are silenced by default; pass force=True to override.

    Args:
        is_master_process: Whether the current process is the master (rank 0).
    """
    original_print = builtins.print

    def timestamped_print(*args, **kwargs):
        force_print = kwargs.pop('force', False)
        # When world size > 8, all processes print (useful for large-scale distributed debugging)
        force_print = force_print or (get_world_size() > 8)
        if is_master_process or force_print:
            current_time = datetime.datetime.now().time()
            original_print('[{}] '.format(current_time), end='')
            original_print(*args, **kwargs)

    builtins.print = timestamped_print


def is_dist_avail_and_initialized():
    """Return True if PyTorch distributed is available and has been initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Return the total number of processes in the distributed group (1 if not distributed)."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Return the rank of the current process (0 if not distributed)."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Return True if the current process is the master process (rank 0)."""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """Call torch.save only on the master process to avoid redundant file writes."""
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    Initialize the distributed training environment.

    Supports three configuration backends (in priority order):
    1. ITP cluster (via OMPI environment variables).
    2. Standard torchrun / torch.distributed.launch (RANK / WORLD_SIZE env vars).
    3. SLURM cluster (SLURM_PROCID env var).
    If none of the above apply, falls back to single-process (non-distributed) mode.

    Args:
        args: Namespace object; the following fields are set after initialization:
              - distributed:   bool, whether distributed mode is active.
              - rank:          Global process rank.
              - gpu:           Local GPU index for this process.
              - world_size:    Total number of processes.
              - dist_backend:  Communication backend ('nccl').
    """
    if args.dist_on_itp:
        # ITP cluster: use OpenMPI environment variables
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Standard torchrun / torch.distributed.launch
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM cluster
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        # Single-process (non-distributed) mode
        print('Not using distributed mode (single process)')
        setup_distributed_print(is_master_process=True)
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| Distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu
    ), flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    torch.distributed.barrier()
    setup_distributed_print(args.rank == 0)


# ============================================================
# Optimizer utilities
# ============================================================

def add_weight_decay(model, weight_decay=0, skip_list=()):
    """
    Split model parameters into two groups:
    one with weight decay applied and one without.

    Parameters excluded from weight decay (to avoid degrading normalization
    layers and biases):
    - 1-D parameters (e.g., bias, norm weight/bias)
    - Parameters whose name ends with '.bias'
    - Parameters listed in skip_list
    - Parameters whose name contains 'diffloss'

    Args:
        model:        PyTorch model.
        weight_decay: Weight decay coefficient.
        skip_list:    List of parameter names that should not be decayed.

    Returns:
        param_groups: Two-element list of parameter group dicts,
                      suitable for passing directly to an optimizer.
    """
    decay_params = []
    no_decay_params = []

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen parameters
        if (
            len(param.shape) == 1             # 1-D param (bias / norm weight)
            or param_name.endswith(".bias")   # explicit bias parameter
            or param_name in skip_list        # manually specified skip
            or 'diffloss' in param_name       # diffloss-related parameter
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {'params': no_decay_params, 'weight_decay': 0.},
        {'params': decay_params, 'weight_decay': weight_decay}
    ]


# ============================================================
# Checkpoint saving utilities
# ============================================================

def save_model(args, model_without_ddp, optimizer, epoch, epoch_name=None):
    """
    Save a complete training checkpoint containing model weights,
    EMA weights, optimizer state, epoch number, and training args.

    Args:
        args:              Training arguments (used to rebuild config on resume).
        model_without_ddp: Model without DDP wrapper.
        optimizer:         Optimizer.
        epoch:             Current epoch index.
        epoch_name:        Suffix for the checkpoint filename (defaults to str(epoch)).
    """
    if epoch_name is None:
        epoch_name = str(epoch)

    output_dir = Path(args.output_dir)
    checkpoint_save_path = output_dir / ('checkpoint-%s.pth' % epoch_name)

    # Build the checkpoint dictionary
    checkpoint_data = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
    }

    # Save both EMA copies
    ema_state_dict1 = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict2 = copy.deepcopy(model_without_ddp.state_dict())
    for param_idx, (param_name, _) in enumerate(model_without_ddp.named_parameters()):
        assert param_name in ema_state_dict1 and param_name in ema_state_dict2
        ema_state_dict1[param_name] = model_without_ddp.ema_params1[param_idx]
        ema_state_dict2[param_name] = model_without_ddp.ema_params2[param_idx]

    checkpoint_data['model_ema1'] = ema_state_dict1
    checkpoint_data['model_ema2'] = ema_state_dict2

    save_on_master(checkpoint_data, checkpoint_save_path)


def save_model_jit(args, model, model_without_ddp, optimizer, loss_scaler, epoch):
    """
    Save a checkpoint in JiT/SiT mode, where the model is wrapped inside
    a DenoiserSiT.net. EMA parameter keys are prefixed with 'net.' to match
    the DenoiserSiT parameter namespace.

    Args:
        args:              Training arguments.
        model:             DDP-wrapped model (unused; kept for interface compatibility).
        model_without_ddp: Model without DDP wrapper.
        optimizer:         Optimizer.
        loss_scaler:       AMP loss scaler (can be None).
        epoch:             Current epoch index.
    """
    epoch_name = str(epoch)
    output_dir = Path(args.output_dir)
    checkpoint_save_path = output_dir / ('checkpoint-%s.pth' % epoch_name)

    checkpoint_data = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
        'args': args,
    }

    # Save EMA1 parameters (keys prefixed with 'net.')
    if hasattr(model_without_ddp, 'ema_params1') and model_without_ddp.ema_params1 is not None:
        ema_state_dict1 = {}
        inner_param_names = [name for name, _ in model_without_ddp.net.named_parameters()]
        for param_idx, inner_name in enumerate(inner_param_names):
            full_param_name = 'net.' + inner_name  # restore full DenoiserSiT parameter path
            ema_state_dict1[full_param_name] = model_without_ddp.ema_params1[param_idx]
        checkpoint_data['model_ema1'] = ema_state_dict1

    # Save EMA2 parameters
    if hasattr(model_without_ddp, 'ema_params2') and model_without_ddp.ema_params2 is not None:
        ema_state_dict2 = {}
        inner_param_names = [name for name, _ in model_without_ddp.net.named_parameters()]
        for param_idx, inner_name in enumerate(inner_param_names):
            full_param_name = 'net.' + inner_name
            ema_state_dict2[full_param_name] = model_without_ddp.ema_params2[param_idx]
        checkpoint_data['model_ema2'] = ema_state_dict2

    save_on_master(checkpoint_data, checkpoint_save_path)


# ============================================================
# Distributed computation utilities
# ============================================================

def all_reduce_mean(local_value):
    """
    Compute the global mean of a scalar value across all distributed processes
    using All-Reduce followed by division by world_size.
    Returns the value unchanged in single-process mode.

    Args:
        local_value: Local scalar value (float or int) on the current process.

    Returns:
        Global mean across all processes (float).
    """
    world_size = get_world_size()
    if world_size > 1:
        value_tensor = torch.tensor(local_value).cuda()
        dist.all_reduce(value_tensor)
        value_tensor /= world_size
        return value_tensor.item()
    else:
        return local_value
