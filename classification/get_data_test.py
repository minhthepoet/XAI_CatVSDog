import argparse
import os
import sys
from pathlib import Path

import torch
from torchvision import datasets
from torchvision.utils import save_image

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from classification.data_preprocessing import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        _accept_jpeg_file,
        _apply_filtered_samples,
        _collect_samples,
        _resolve_data_roots,
        build_transforms,
    )
else:
    from .data_preprocessing import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        _accept_jpeg_file,
        _apply_filtered_samples,
        _collect_samples,
        _resolve_data_roots,
        build_transforms,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export test images after preprocessing (Resize/CenterCrop/Normalize)."
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--scan_corrupted", action="store_true")
    parser.add_argument(
        "--save_tensor",
        action="store_true",
        help="Also save normalized tensor (.pt) used by the model.",
    )
    return parser.parse_args()


def unnormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=image_tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=image_tensor.dtype).view(3, 1, 1)
    return torch.clamp(image_tensor * std + mean, 0.0, 1.0)


def build_test_dataset(args):
    train_root, test_root = _resolve_data_roots(args.data_dir)
    _, eval_transform = build_transforms(args.img_size)

    if test_root is None:
        raise ValueError(
            "This script expects explicit test data at data/test/{cats,dogs}. "
            "No random split is performed in get_data_test.py."
        )

    test_dataset = datasets.ImageFolder(
        root=test_root, transform=eval_transform, is_valid_file=_accept_jpeg_file
    )
    valid_samples, valid_targets = _collect_samples(
        test_dataset, split_name="test", scan_corrupted=args.scan_corrupted
    )
    _apply_filtered_samples(test_dataset, valid_samples, valid_targets)
    test_indices = list(range(len(valid_samples)))
    return test_dataset, test_indices


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_dataset, test_indices = build_test_dataset(args)
    print(f"Found {len(test_indices)} test images.")

    for i, idx in enumerate(test_indices, start=1):
        image_tensor, target = test_dataset[idx]
        src_path, _ = test_dataset.samples[idx]

        class_name = test_dataset.classes[target]
        stem = Path(src_path).stem
        class_out = out_dir / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        # Save viewable image after resize/crop (de-normalized for visualization).
        save_image(unnormalize(image_tensor), class_out / f"{stem}.png")

        if args.save_tensor:
            torch.save(image_tensor, class_out / f"{stem}.pt")

        if i % 500 == 0 or i == len(test_indices):
            print(f"Saved {i}/{len(test_indices)}")

    print(f"Done. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
