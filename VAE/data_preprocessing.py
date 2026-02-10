import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
        merge_mode: str,
        normalize_acts: bool,
        exp_root: str,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.target_hw = target_hw
        self.merge_mode = merge_mode
        self.normalize_acts = normalize_acts
        self.transform = transform if transform is not None else build_image_transform()
        if self.merge_mode == "merge":
            self.stats_path = Path(exp_root) / "acts_stats_merged.pt"
        else:
            self.stats_path = Path(exp_root) / "acts_stats_split.pt"

        print("[Data] Step 1: Pair image/acts files", flush=True)
        self.samples = self._scan_pairs_fast()
        print(f"[Data] Step 1 done: pairs={len(self.samples)}", flush=True)
        if len(self.samples) == 0:
            raise RuntimeError("No paired (image, acts) samples found in data_dir.")

        self.mean = None
        self.std = None
        if self.normalize_acts:
            print("[Data] Step 2: Prepare activation stats", flush=True)
            self.mean, self.std = self._load_or_compute_stats()
            print("[Data] Step 2 done: stats ready", flush=True)

    def _scan_pairs_fast(self) -> List[Tuple[Path, Path, str]]:
        paired_samples: List[Tuple[Path, Path, str]] = []
        for cls_name in ["cat", "dog"]:
            class_dir = self.data_dir / cls_name
            if not class_dir.is_dir():
                warnings.warn(f"Missing class directory: {class_dir}")
                continue

            image_paths = sorted(class_dir.glob("*.png"))
            for img_path in image_paths:
                sample_id = img_path.stem
                acts_path = class_dir / f"{sample_id}__acts.pt"
                if not acts_path.exists():
                    warnings.warn(f"Missing acts file for {sample_id}: {acts_path}")
                    continue
                paired_samples.append((img_path, acts_path, sample_id))

        return paired_samples

    def _extract_layers(self, acts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not isinstance(acts, dict):
            raise TypeError("acts file must contain a dict.")
        for k in ACT_KEYS:
            if k not in acts:
                raise KeyError(f"Missing key '{k}' in acts dict.")

        out = {}
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
            out[k] = t.float()
        return out

    def _merge_acts(self, acts: Dict[str, torch.Tensor]) -> torch.Tensor:
        layers = self._extract_layers(acts)
        merged_parts: List[torch.Tensor] = []
        for k in ACT_KEYS:
            t = layers[k]

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

    def _load_or_compute_stats_merged(self):
        if self.stats_path.exists():
            print(f"[Data] Load cached stats: {self.stats_path}", flush=True)
            payload = torch.load(self.stats_path, map_location="cpu")
            mean = payload["mean"].float()
            std = payload["std"].float()
            return mean, std

        sum_c: Optional[torch.Tensor] = None
        sum_sq_c: Optional[torch.Tensor] = None
        count = 0
        used = 0

        print("[Data] Compute stats from dataset (this can take time)...", flush=True)
        for _, acts_path, sample_id in self.samples:
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
        torch.save(payload, self.stats_path)
        print(f"[Data] Saved stats cache: {self.stats_path}", flush=True)
        return mean, std

    def _load_or_compute_stats_split(self):
        if self.stats_path.exists():
            print(f"[Data] Load cached stats: {self.stats_path}", flush=True)
            payload = torch.load(self.stats_path, map_location="cpu")
            mean = {k: payload["mean"][k].float() for k in ACT_KEYS}
            std = {k: payload["std"][k].float() for k in ACT_KEYS}
            return mean, std

        print("[Data] Compute split stats from dataset (this can take time)...", flush=True)
        sums = {}
        sums2 = {}
        counts = {}
        used = 0
        for _, acts_path, sample_id in self.samples:
            try:
                acts = torch.load(acts_path, map_location="cpu")
                layers = self._extract_layers(acts)
            except Exception as exc:
                warnings.warn(f"Failed while computing split stats for {sample_id}: {exc}")
                continue

            for k in ACT_KEYS:
                t = layers[k].to(torch.float64)
                if k not in sums:
                    sums[k] = torch.zeros(t.shape[0], dtype=torch.float64)
                    sums2[k] = torch.zeros(t.shape[0], dtype=torch.float64)
                    counts[k] = 0
                sums[k] += t.sum(dim=(1, 2))
                sums2[k] += (t * t).sum(dim=(1, 2))
                counts[k] += t.shape[1] * t.shape[2]
            used += 1

        if used == 0:
            raise RuntimeError("Could not compute split activation stats from valid samples.")

        mean = {}
        std = {}
        for k in ACT_KEYS:
            m = (sums[k] / counts[k]).float().view(-1, 1, 1)
            var = (sums2[k] / counts[k]) - (sums[k] / counts[k]).pow(2)
            s = torch.sqrt(torch.clamp(var, min=0.0)).float().view(-1, 1, 1)
            mean[k] = m
            std[k] = s

        payload = {
            "mean": mean,
            "std": std,
            "num_samples": used,
            "keys_order": ACT_KEYS,
        }
        torch.save(payload, self.stats_path)
        print(f"[Data] Saved split stats cache: {self.stats_path}", flush=True)
        return mean, std

    def _load_or_compute_stats(self):
        if self.merge_mode == "merge":
            return self._load_or_compute_stats_merged()
        return self._load_or_compute_stats_split()

    def __len__(self):
        return len(self.samples)

    def _load_item(self, idx: int):
        img_path, acts_path, sample_id = self.samples[idx]

        with Image.open(img_path) as im:
            im = im.convert("RGB")
        x_img = self.transform(im)

        acts = torch.load(acts_path, map_location="cpu")
        if self.merge_mode == "merge":
            y_merged = self._merge_acts(acts)
            if self.normalize_acts and self.mean is not None and self.std is not None:
                y_merged = (y_merged - self.mean) / (self.std + 1e-6)
            target = y_merged.to(torch.float32)
        else:
            layers = self._extract_layers(acts)
            if self.normalize_acts and self.mean is not None and self.std is not None:
                norm_layers = {}
                for k in ACT_KEYS:
                    norm_layers[k] = (layers[k] - self.mean[k]) / (self.std[k] + 1e-6)
                layers = norm_layers
            target = {k: layers[k].to(torch.float32) for k in ACT_KEYS}

        return x_img, target, sample_id

    def __getitem__(self, idx):
        return self._load_item(idx)


def build_dataloader(args, exp_root):
    print("[Data] Build dataset", flush=True)
    dataset = PairedActsDataset(
        data_dir=args.data_dir,
        target_hw=args.target_hw,
        merge_mode=args.merge_mode,
        normalize_acts=args.normalize_acts,
        exp_root=exp_root,
        transform=build_image_transform(),
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(dataset, **loader_kwargs)
    print(f"[Data] Build dataloader done: batches={len(loader)}", flush=True)
    return loader
