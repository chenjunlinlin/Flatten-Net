# Code for "TDN: Temporal Difference Networks for Efficient Action Recognition"
# arXiv: 2012.10071
# Limin Wang, Zhan Tong, Bin Ji, Gangshan Wu
# tongzhan@smail.nju.edu.cn

from torch.optim.lr_scheduler import _LRScheduler, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler


def get_scheduler(optimizer, n_iter_per_epoch, args):
    if "cosine" in args.lr_scheduler:
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=5e-7,
            T_max=(args.epochs - args.warmup_epoch) * n_iter_per_epoch)
    elif "step" in args.lr_scheduler:
        scheduler = MultiStepLR(
            optimizer=optimizer,
            gamma=args.lr_decay_rate,
            milestones=[(m - args.warmup_epoch) * n_iter_per_epoch for m in args.lr_steps])
    elif "reduce":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_rate, patience=2, verbose=True)
    else:
        raise NotImplementedError(f"scheduler {args.lr_scheduler} not supported")

    # GradualWarmupScheduler
    # https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    if args.warmup_epoch != 0 :
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=args.warmup_multiplier,
            total_epoch=args.warmup_epoch * n_iter_per_epoch,
            after_scheduler=scheduler)

    return scheduler
