import math


def adjust_learning_rate(optimizer, current_epoch_float, args):
    """
    Dynamically adjust the learning rate per iteration step,
    supporting linear warmup followed by constant or cosine annealing.

    Scheduling strategy:
    1. Warmup phase (current_epoch_float < warmup_epochs):
       Learning rate increases linearly from 0 to args.lr.
    2. Post-warmup phase (current_epoch_float >= warmup_epochs):
       - 'constant': Keep learning rate fixed at args.lr.
       - 'cosine':   Apply half-cosine annealing from args.lr down to args.min_lr.

    Args:
        optimizer:           PyTorch optimizer whose learning rate will be updated.
        current_epoch_float: Fractional epoch number (batch_step / steps_per_epoch + epoch_int),
                             allowing smooth per-step LR adjustment rather than per-epoch.
        args:                Namespace containing the following fields:
                             - lr:             Base (peak) learning rate.
                             - warmup_epochs:  Number of warmup epochs.
                             - lr_schedule:    Scheduling strategy name ('constant' or 'cosine').
                             - min_lr:         Minimum learning rate for cosine annealing.
                             - epochs:         Total number of training epochs.

    Returns:
        Current learning rate (float).
    """
    if current_epoch_float < args.warmup_epochs:
        # Warmup: linearly ramp up from 0 to args.lr
        lr = args.lr * current_epoch_float / args.warmup_epochs
    else:
        if args.lr_schedule == "constant":
            # Constant schedule: keep learning rate at peak value
            lr = args.lr
        elif args.lr_schedule == "cosine":
            # Half-cosine annealing: smoothly decay from args.lr to args.min_lr
            warmup_elapsed = current_epoch_float - args.warmup_epochs
            total_decay_epochs = args.epochs - args.warmup_epochs
            cosine_factor = 0.5 * (1. + math.cos(math.pi * warmup_elapsed / total_decay_epochs))
            lr = args.min_lr + (args.lr - args.min_lr) * cosine_factor
        else:
            raise NotImplementedError(f"Unknown learning rate schedule: {args.lr_schedule}")

    # Apply the computed learning rate to all parameter groups
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            # Support per-layer learning rate scaling (e.g., Layer-wise LR Decay)
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr
