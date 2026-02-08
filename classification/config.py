import argparse
import random

import torch


def get_args():
    parser = argparse.ArgumentParser(description="Dogs vs Cats classification (ResNet18)")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log_interval", type=int, required=True)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
    parser.add_argument("--saveEvery", type=int, default=10)
    parser.add_argument(
        "--scan_corrupted",
        action="store_true",
        help="Enable slow pre-scan that opens every image to drop corrupted files.",
    )

    args = parser.parse_args()

    if not (0.0 <= args.val_split < 1.0):
        raise ValueError("--val_split must be in [0, 1).")
    if not (0.0 <= args.test_split < 1.0):
        raise ValueError("--test_split must be in [0, 1).")
    if args.val_split + args.test_split >= 1.0:
        raise ValueError("--val_split + --test_split must be < 1.0.")
    if args.saveEvery <= 0:
        raise ValueError("--saveEvery must be > 0.")
    if args.log_interval <= 0:
        raise ValueError("--log_interval must be > 0.")

    return args


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
