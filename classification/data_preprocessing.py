import os
from typing import List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _has_class_dirs(root: str, class_names: List[str]) -> bool:
    return all(os.path.isdir(os.path.join(root, cls)) for cls in class_names)

def _resolve_data_roots(data_dir: str) -> Tuple[str, Optional[str]]:
    new_train = os.path.join(data_dir, "data", "train")
    new_test = os.path.join(data_dir, "data", "test")
    if _has_class_dirs(new_train, ["cats", "dogs"]):
        if not _has_class_dirs(new_test, ["cats", "dogs"]):
            raise FileNotFoundError(
                "Found new train layout at data/train but missing data/test with both cats/ and dogs/."
            )
        return new_train, new_test

    # Backward-compatible fallback.
    legacy_candidates = [
        os.path.join(data_dir, "PetImages"),
        data_dir,
    ]
    for root in legacy_candidates:
        if _has_class_dirs(root, ["Cat", "Dog"]) or _has_class_dirs(root, ["cat", "dog"]):
            return root, None

    raise FileNotFoundError(
        "Could not resolve dataset layout. Expected either:\n"
        "1) data/train/{cats,dogs} and data/test/{cats,dogs}\n"
        "2) legacy PetImages/{Cat,Dog}\n"
        f"Base path checked: {data_dir}"
    )


def _accept_jpeg_file(path: str) -> bool:
    lower = path.lower()
    return lower.endswith(".jpg") or lower.endswith(".jpeg")


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    return train_transform, eval_transform


def _filter_corrupted_samples(dataset: datasets.ImageFolder, split_name: str):
    total_images = len(dataset.samples)
    valid_samples = []
    valid_targets = []

    skipped = 0
    print(f"[{split_name}] Starting corrupted-image scan for {total_images} files...", flush=True)
    progress_every = 2000
    for idx, (path, target) in enumerate(dataset.samples, start=1):
        try:
            with Image.open(path) as img:
                img.convert("RGB")
            valid_samples.append((path, target))
            valid_targets.append(target)
        except Exception:
            skipped += 1

        if idx % progress_every == 0 or idx == total_images:
            print(
                f"[{split_name}] Scanned {idx}/{total_images} | valid={len(valid_samples)} | skipped={skipped}",
                flush=True,
            )

    dataset.samples = valid_samples
    dataset.imgs = valid_samples
    dataset.targets = valid_targets

    counts_by_class = {name: 0 for name in dataset.classes}
    for t in valid_targets:
        if 0 <= t < len(dataset.classes):
            counts_by_class[dataset.classes[t]] += 1

    print(f"[{split_name}] Total images: {total_images}")
    print(f"[{split_name}] Valid images: {len(valid_samples)}")
    print(f"[{split_name}] Skipped (corrupted): {skipped}")
    for class_name in dataset.classes:
        print(f"[{split_name}] Valid {class_name}: {counts_by_class[class_name]}")

    return valid_samples, valid_targets


def _collect_samples(dataset: datasets.ImageFolder, split_name: str, scan_corrupted: bool):
    if scan_corrupted:
        return _filter_corrupted_samples(dataset, split_name=split_name)

    # Fast path: trust directory contents and skip costly open/verify loop.
    total = len(dataset.samples)
    print(f"[{split_name}] Fast mode: skipping corrupted-image scan. Using {total} JPEG files as-is.", flush=True)
    return list(dataset.samples), list(dataset.targets)


def _apply_filtered_samples(dataset: datasets.ImageFolder, samples, targets):
    dataset.samples = list(samples)
    dataset.imgs = list(samples)
    dataset.targets = list(targets)


def build_dataloaders(args):
    train_root, test_root = _resolve_data_roots(args.data_dir)
    print(f"Train root: {train_root}", flush=True)
    if test_root is not None:
        print(f"Test root: {test_root}", flush=True)
    else:
        print("Test root: <none> (using random split from train root)", flush=True)

    train_transform, eval_transform = build_transforms(args.img_size)

    base_train_dataset = datasets.ImageFolder(root=train_root, is_valid_file=_accept_jpeg_file)
    train_valid_samples, train_valid_targets = _collect_samples(
        base_train_dataset,
        split_name="train",
        scan_corrupted=getattr(args, "scan_corrupted", False),
    )

    num_train_valid = len(train_valid_samples)
    if num_train_valid == 0:
        raise RuntimeError("No valid training images were found after filtering corrupted files.")

    val_size = int(num_train_valid * args.val_split)
    train_size = num_train_valid - val_size
    if train_size <= 0:
        raise ValueError("val_split leaves no samples for training. Adjust --val_split.")

    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(num_train_valid, generator=generator).tolist()
    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    train_dataset = datasets.ImageFolder(
        root=train_root, transform=train_transform, is_valid_file=_accept_jpeg_file
    )
    val_dataset = datasets.ImageFolder(
        root=train_root, transform=eval_transform, is_valid_file=_accept_jpeg_file
    )

    _apply_filtered_samples(train_dataset, train_valid_samples, train_valid_targets)
    _apply_filtered_samples(val_dataset, train_valid_samples, train_valid_targets)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    if test_root is not None:
        test_dataset = datasets.ImageFolder(
            root=test_root, transform=eval_transform, is_valid_file=_accept_jpeg_file
        )
        test_valid_samples, test_valid_targets = _collect_samples(
            test_dataset,
            split_name="test",
            scan_corrupted=getattr(args, "scan_corrupted", False),
        )
        if test_dataset.classes != train_dataset.classes:
            raise RuntimeError(
                "Train/Test class mismatch.\n"
                f"train classes={train_dataset.classes}\n"
                f"test classes={test_dataset.classes}"
            )
        _apply_filtered_samples(test_dataset, test_valid_samples, test_valid_targets)
        test_subset = Subset(test_dataset, list(range(len(test_valid_samples))))
        if args.test_split > 0:
            print("[Info] Ignoring --test_split because explicit data/test split is available.", flush=True)
    else:
        test_size = int(num_train_valid * args.test_split)
        test_indices = perm[:test_size]
        # Remove test indices from train/val if we're in legacy random-split mode.
        val_indices = perm[test_size : test_size + val_size]
        train_indices = perm[test_size + val_size :]
        if len(train_indices) == 0:
            raise ValueError("Split sizes leave no samples for training. Adjust val/test split.")
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

        test_dataset = datasets.ImageFolder(
            root=train_root, transform=eval_transform, is_valid_file=_accept_jpeg_file
        )
        _apply_filtered_samples(test_dataset, train_valid_samples, train_valid_targets)
        test_subset = Subset(test_dataset, test_indices)

    print(
        f"Split sizes | train={len(train_subset)} val={len(val_subset)} test={len(test_subset)}",
        flush=True,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, train_dataset.classes
