import os
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _find_dataset_root(data_dir: str) -> str:
    candidates = [
        os.path.join(data_dir, "PetImages"),
        data_dir,
    ]
    for root in candidates:
        if os.path.isdir(os.path.join(root, "Cat")) and os.path.isdir(os.path.join(root, "Dog")):
            return root
    raise FileNotFoundError(
        "Could not find dataset root containing both 'Cat/' and 'Dog/'. "
        f"Checked: {candidates}"
    )


def _accept_any_file(_path: str) -> bool:
    # Defer actual image validity checks to _filter_corrupted_samples.
    return True


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


def _filter_corrupted_samples(dataset: datasets.ImageFolder):
    total_images = len(dataset.samples)
    valid_samples = []
    valid_targets = []

    skipped = 0
    print(f"Starting corrupted-image scan for {total_images} files...", flush=True)
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
                f"Scanned {idx}/{total_images} | valid={len(valid_samples)} | skipped={skipped}",
                flush=True,
            )

    dataset.samples = valid_samples
    dataset.imgs = valid_samples
    dataset.targets = valid_targets

    cat_count = sum(t == 0 for t in valid_targets)
    dog_count = sum(t == 1 for t in valid_targets)

    print(f"Total images: {total_images}")
    print(f"Valid images: {len(valid_samples)}")
    print(f"Skipped (corrupted) images: {skipped}")
    print(f"Valid Cat count: {cat_count}")
    print(f"Valid Dog count: {dog_count}")

    return valid_samples, valid_targets


def _apply_filtered_samples(dataset: datasets.ImageFolder, samples, targets):
    dataset.samples = list(samples)
    dataset.imgs = list(samples)
    dataset.targets = list(targets)


def build_dataloaders(args):
    root = _find_dataset_root(args.data_dir)
    print(f"Dataset root: {root}", flush=True)
    train_transform, eval_transform = build_transforms(args.img_size)

    base_dataset = datasets.ImageFolder(root=root, is_valid_file=_accept_any_file)
    valid_samples, valid_targets = _filter_corrupted_samples(base_dataset)

    num_valid = len(valid_samples)
    if num_valid == 0:
        raise RuntimeError("No valid images were found after filtering corrupted files.")

    test_size = int(num_valid * args.test_split)
    val_size = int(num_valid * args.val_split)
    train_size = num_valid - val_size - test_size
    if train_size <= 0:
        raise ValueError("Split sizes leave no samples for training. Adjust val/test split.")

    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(num_valid, generator=generator).tolist()

    test_indices = perm[:test_size]
    val_indices = perm[test_size : test_size + val_size]
    train_indices = perm[test_size + val_size :]
    print(
        f"Split sizes | train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}",
        flush=True,
    )

    train_dataset = datasets.ImageFolder(
        root=root, transform=train_transform, is_valid_file=_accept_any_file
    )
    val_dataset = datasets.ImageFolder(
        root=root, transform=eval_transform, is_valid_file=_accept_any_file
    )
    test_dataset = datasets.ImageFolder(
        root=root, transform=eval_transform, is_valid_file=_accept_any_file
    )

    _apply_filtered_samples(train_dataset, valid_samples, valid_targets)
    _apply_filtered_samples(val_dataset, valid_samples, valid_targets)
    _apply_filtered_samples(test_dataset, valid_samples, valid_targets)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    test_subset = Subset(test_dataset, test_indices)

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
