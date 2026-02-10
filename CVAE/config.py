import argparse
import os

import torch


def get_args():
    parser = argparse.ArgumentParser(description="Train conditional VAE on merged CNN activations.")

    # Required I/O.
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="cvae_merged_acts")

    # Training.
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # VAE.
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.2)

    # Merge / stats.
    parser.add_argument("--target_hw", type=int, default=56)
    parser.add_argument("--normalize_acts", action="store_true")
    parser.add_argument("--no_normalize_acts", dest="normalize_acts", action="store_false")
    parser.set_defaults(normalize_acts=True)

    # Checkpointing / logging.
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--log_every", type=int, default=50)

    # Optional.
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=0.0)
    args = parser.parse_args()

    if args.save_every <= 0:
        raise ValueError("--save_every must be > 0.")
    if args.log_every <= 0:
        raise ValueError("--log_every must be > 0.")
    if args.target_hw <= 0:
        raise ValueError("--target_hw must be > 0.")

    return args


def make_exp_dirs(args):
    exp_root = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(os.path.join(exp_root, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_root, "logs"), exist_ok=True)
    return exp_root
