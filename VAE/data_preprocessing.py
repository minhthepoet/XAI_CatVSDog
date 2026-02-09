import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ACT_KEYS = ["stem", "layer1", "layer2", "layer3", "layer4"]


def build_image_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class PairedActsDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        target_hw: int,
        normalize_acts: bool,
        exp_root: str,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.target_hw = target_hw
        self.normalize_acts = normalize_acts
        self.transform = transform if transform is not None else build_image_transform()
        self.stats_path = Path(exp_root) / "acts_stats_merged.pt"

        print("[Data] Fast pairing mode (no file integrity scan) ...", flush=True)
        self.samples = self._scan_pairs_fast()
        print(f"[Data] Paired samples: {len(self.samples)}", flush=True)
        if len(self.samples) == 0:
            raise RuntimeError("No paired (image, acts) samples found in data_dir.")

        self.mean = None
        self.std = None
        if self.normalize_acts:
            print("[Data] normalize_acts=True -> loading/computing merged activation stats ...", flush=True)
            self.mean, self.std = self._load_or_compute_stats()
            print(
                f"[Data] Stats ready: mean={tuple(self.mean.shape)}, std={tuple(self.std.shape)}",
                flush=True,
            )

    def _scan_pairs_fast(self) -> List[Tuple[Path, Path, str]]:
        paired_samples: List[Tuple[Path, Path, str]] = []
        for cls_name in ["cat", "dog"]:
            class_dir = self.data_dir / cls_name
            if not class_dir.is_dir():
                warnings.warn(f"Missing class directory: {class_dir}")
                continue

            image_paths = sorted(class_dir.glob("*.png"))
            print(f"[Data] {cls_name}: found {len(image_paths)} .png files", flush=True)
            for img_path in tqdm(image_paths, desc=f"[Scan {cls_name}]", leave=False):
                sample_id = img_path.stem
                acts_path = class_dir / f"{sample_id}__acts.pt"
                if not acts_path.exists():
                    warnings.warn(f"Missing acts file for {sample_id}: {acts_path}")
                    continue
                paired_samples.append((img_path, acts_path, sample_id))

        return paired_samples

    def _merge_acts(self, acts: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(acts, dict):
            raise TypeError("acts file must contain a dict.")
        for k in ACT_KEYS:
            if k not in acts:
                raise KeyError(f"Missing key '{k}' in acts dict.")

        merged_parts: List[torch.Tensor] = []
        for k in ACT_KEYS:
            t = acts[k]
            if not torch.is_tensor(t):
                raise TypeError(f"acts['{k}'] is not a tensor.")
            if t.ndim == 4:
                if t.shape[0] != 1:
                    raise ValueError(f"acts['{k}'] batch dim must be 1, got {t.shape[0]}.")
                t = t.squeeze(0)
            elif t.ndim != 3:
                raise ValueError(f"acts['{k}'] must have 3 or 4 dims, got {t.ndim}.")

            t = t.float().unsqueeze(0)  # [1, C, H, W]
            t = F.interpolate(
                t,
                size=(self.target_hw, self.target_hw),
                mode="bilinear",
                align_corners=False,
            )
            merged_parts.append(t.squeeze(0))

        merged = torch.cat(merged_parts, dim=0).to(torch.float32)
        if merged.ndim != 3:
            raise ValueError(f"Merged acts must be [C,H,W], got shape={tuple(merged.shape)}")
        return merged

    def _load_or_compute_stats(self):
        if self.stats_path.exists():
            print(f"[Data] Loading cached stats from: {self.stats_path}", flush=True)
            payload = torch.load(self.stats_path, map_location="cpu")
            mean = payload["mean"].float()
            std = payload["std"].float()
            return mean, std

        sum_c: Optional[torch.Tensor] = None
        sum_sq_c: Optional[torch.Tensor] = None
        count = 0
        used = 0

        print("[Data] Computing channel-wise mean/std over merged activations ...", flush=True)
        for _, acts_path, sample_id in tqdm(self.samples, desc="[Stats]", leave=False):
            try:
                acts = torch.load(acts_path, map_location="cpu")
                merged = self._merge_acts(acts).to(torch.float64)
            except Exception as exc:
                warnings.warn(f"Failed while computing stats for {sample_id}: {exc}")
                continue

            if sum_c is None:
                c = merged.shape[0]
                sum_c = torch.zeros(c, dtype=torch.float64)
                sum_sq_c = torch.zeros(c, dtype=torch.float64)

            sum_c += merged.sum(dim=(1, 2))
            sum_sq_c += (merged * merged).sum(dim=(1, 2))
            count += merged.shape[1] * merged.shape[2]
            used += 1

        if used == 0 or sum_c is None or sum_sq_c is None or count == 0:
            raise RuntimeError("Could not compute activation stats from valid samples.")

        mean = (sum_c / count).float().view(-1, 1, 1)
        var = (sum_sq_c / count) - (sum_c / count).pow(2)
        std = torch.sqrt(torch.clamp(var, min=0.0)).float().view(-1, 1, 1)

        payload = {
            "mean": mean,
            "std": std,
            "target_hw": self.target_hw,
            "num_samples": used,
            "keys_order": ACT_KEYS,
        }
        print(f"[Data] Saving stats cache to: {self.stats_path}", flush=True)
        torch.save(payload, self.stats_path)
        return mean, std

    def __len__(self):
        return len(self.samples)

    def _load_item(self, idx: int):
        img_path, acts_path, sample_id = self.samples[idx]

        with Image.open(img_path) as im:
            im = im.convert("RGB")
        x_img = self.transform(im)

        acts = torch.load(acts_path, map_location="cpu")
        y_merged = self._merge_acts(acts)

        if self.normalize_acts and self.mean is not None and self.std is not None:
            y_merged = (y_merged - self.mean) / (self.std + 1e-6)

        return x_img, y_merged.to(torch.float32), sample_id

    def __getitem__(self, idx):
        return self._load_item(idx)


def build_dataloader(args, exp_root):
    print("[Data] Building training dataset ...", flush=True)
    dataset = PairedActsDataset(
        data_dir=args.data_dir,
        target_hw=args.target_hw,
        normalize_acts=args.normalize_acts,
        exp_root=exp_root,
        transform=build_image_transform(),
    )
    print("[Data] Building DataLoader ...", flush=True)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )
    print(
        f"[Data] DataLoader ready: samples={len(dataset)}, batches/epoch={len(loader)}",
        flush=True,
    )
    return loader
