import argparse
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


DEFAULT_LAYERS = ["stem", "layer1", "layer2", "layer3", "layer4"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize spatial activation maps (raw activations, not Grad-CAM)."
    )
    parser.add_argument("--root", type=str, default="data_explain")
    parser.add_argument("--out_dir", type=str, default="viz_out")
    parser.add_argument("--cls", type=str, choices=["cat", "dog", "all"], default="all")
    parser.add_argument("--max_items", type=int, default=50)
    parser.add_argument(
        "--layers",
        type=str,
        default="stem,layer1,layer2,layer3,layer4",
        help="Comma-separated layers. Supports optional 'y_merged' if present in pt file.",
    )
    parser.add_argument("--reduce", type=str, choices=["mean", "max", "l2"], default="mean")
    parser.add_argument("--topk_channels", type=int, default=8)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--mode", type=str, choices=["single", "grid"], default="grid")
    parser.add_argument("--delta_ref", type=str, choices=["none", "full"], default="none")
    parser.add_argument(
        "--full_ref_root",
        type=str,
        default="data_full_acts",
        help="Used when --delta_ref=full. Expected: <full_ref_root>/<class>/ID__acts.pt",
    )
    parser.add_argument("--use_abs", action="store_true")
    parser.add_argument("--percentile_norm", type=float, default=99.5)
    parser.add_argument(
        "--drive_out_dir",
        type=str,
        default="",
        help="Optional second output root (e.g., Google Drive path). Files are saved to both out_dir and drive_out_dir.",
    )
    parser.add_argument(
        "--show_inline",
        action="store_true",
        help="Try to display summary images inline (works best with %%run / direct notebook execution).",
    )
    parser.add_argument("--show_max", type=int, default=8)
    return parser.parse_args()


def _to_chw(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 4:
        if t.shape[0] != 1:
            raise ValueError(f"Expected batch dim 1 for activation, got {tuple(t.shape)}")
        t = t.squeeze(0)
    if t.ndim != 3:
        raise ValueError(f"Expected [C,H,W] or [1,C,H,W], got shape={tuple(t.shape)}")
    return t.float().cpu()


def _scan_samples(root: Path, cls_mode: str) -> List[Tuple[str, str, Path, Path]]:
    class_list = ["cat", "dog"] if cls_mode == "all" else [cls_mode]
    items: List[Tuple[str, str, Path, Path]] = []
    for cls_name in class_list:
        cls_dir = root / cls_name
        if not cls_dir.is_dir():
            warnings.warn(f"Missing class dir: {cls_dir}")
            continue
        for img_path in sorted(cls_dir.glob("*.png")):
            sample_id = img_path.stem
            acts_path = cls_dir / f"{sample_id}__acts.pt"
            if not acts_path.exists():
                warnings.warn(f"Missing acts file for {sample_id}: {acts_path}")
                continue
            items.append((cls_name, sample_id, img_path, acts_path))
    return items


def load_sample(
    img_path: Path,
    acts_path: Path,
    requested_layers: List[str],
    delta_ref: str = "none",
    full_ref_acts_path: Optional[Path] = None,
):
    """
    Load one sample image and activation dict.

    Returns:
        img_rgb: np.ndarray [224,224,3], float32 in [0,1]
        layer_to_act: dict[layer_name] -> torch.Tensor [C,H,W]
    """
    try:
        with Image.open(img_path) as im:
            img_rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    except Exception as exc:
        warnings.warn(f"Failed to open image {img_path}: {exc}")
        return None, None

    try:
        acts_obj = torch.load(acts_path, map_location="cpu")
    except Exception as exc:
        warnings.warn(f"Failed to load acts {acts_path}: {exc}")
        return None, None

    if not isinstance(acts_obj, dict):
        warnings.warn(f"Acts file is not dict: {acts_path}")
        return None, None

    ref_obj = None
    if delta_ref == "full":
        if full_ref_acts_path is None or (not full_ref_acts_path.exists()):
            warnings.warn(f"Missing full reference acts: {full_ref_acts_path}")
            return None, None
        try:
            ref_obj = torch.load(full_ref_acts_path, map_location="cpu")
        except Exception as exc:
            warnings.warn(f"Failed to load full reference acts {full_ref_acts_path}: {exc}")
            return None, None
        if not isinstance(ref_obj, dict):
            warnings.warn(f"Reference acts is not dict: {full_ref_acts_path}")
            return None, None

    layer_to_act: Dict[str, torch.Tensor] = {}
    for layer in requested_layers:
        if layer not in acts_obj:
            warnings.warn(f"Layer '{layer}' not found in {acts_path.name}; skipping layer.")
            continue
        try:
            act = _to_chw(acts_obj[layer])
            if delta_ref == "full":
                if layer not in ref_obj:
                    warnings.warn(f"Layer '{layer}' missing in full ref for {acts_path.name}; skipping layer.")
                    continue
                ref_act = _to_chw(ref_obj[layer])
                if act.shape != ref_act.shape:
                    warnings.warn(
                        f"Shape mismatch for layer '{layer}': part={tuple(act.shape)} vs full={tuple(ref_act.shape)}"
                    )
                    continue
                act = act - ref_act
            layer_to_act[layer] = act
        except Exception as exc:
            warnings.warn(f"Failed to parse layer '{layer}' in {acts_path.name}: {exc}")

    if not layer_to_act:
        return None, None
    return img_rgb, layer_to_act


def reduce_activation(A: torch.Tensor, reduce: str, use_abs: bool) -> np.ndarray:
    """
    Reduce [C,H,W] to [H,W] over channels.
    """
    if use_abs:
        A = A.abs()
    if reduce == "mean":
        M = A.mean(dim=0)
    elif reduce == "max":
        M = A.max(dim=0).values
    elif reduce == "l2":
        M = torch.sqrt(torch.clamp((A ** 2).mean(dim=0), min=0.0))
    else:
        raise ValueError(f"Unknown reduce mode: {reduce}")
    return M.cpu().numpy().astype(np.float32)


def robust_normalize(M: np.ndarray, percentile: float) -> np.ndarray:
    """
    Robustly normalize map to [0,1]:
    - shift by min
    - clip to percentile
    - scale by clipped max
    """
    M = np.asarray(M, dtype=np.float32)
    if M.size == 0:
        return M
    M = M - np.nanmin(M)
    p = float(np.clip(percentile, 1.0, 100.0))
    cap = np.nanpercentile(M, p)
    if not np.isfinite(cap) or cap <= 1e-12:
        return np.zeros_like(M, dtype=np.float32)
    M = np.clip(M, 0.0, cap) / cap
    return M.astype(np.float32)


def upsample_to_img(M: np.ndarray, target_hw=(224, 224)) -> np.ndarray:
    """
    Bilinear upsample [H,W] to target_hw.
    """
    t = torch.from_numpy(M).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def overlay_heatmap(img_rgb: np.ndarray, heat: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend heatmap (viridis colormap) over RGB image.
    """
    heat = np.clip(heat, 0.0, 1.0)
    heat_rgb = cm.get_cmap("viridis")(heat)[..., :3].astype(np.float32)
    a = float(np.clip(alpha, 0.0, 1.0))
    out = (1.0 - a) * img_rgb + a * heat_rgb
    return np.clip(out, 0.0, 1.0)


def _save_rgb_png(arr01: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img_u8 = np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img_u8).save(path)


def _save_heat_png(heat01: np.ndarray, path: Path, cmap_name: str = "inferno"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cm.get_cmap(cmap_name)(np.clip(heat01, 0.0, 1.0))[..., :3]
    _save_rgb_png(rgb.astype(np.float32), path)


def save_topk_channels(
    A: torch.Tensor,
    layer_name: str,
    sample_out_dir: Path,
    topk: int,
    use_abs: bool,
    percentile_norm: float,
):
    """
    Save top-k single-channel activation maps and a montage.
    """
    if topk <= 0:
        return

    A_use = A.abs() if use_abs else A
    energies = A_use.mean(dim=(1, 2)).cpu().numpy()
    k = min(topk, A_use.shape[0])
    if k <= 0:
        return
    top_idx = np.argsort(-energies)[:k]

    ch_dir = sample_out_dir / f"channels_{layer_name}"
    ch_dir.mkdir(parents=True, exist_ok=True)

    maps = []
    titles = []
    for rank, c in enumerate(top_idx):
        ch_map = A_use[c].cpu().numpy().astype(np.float32)
        ch_map = robust_normalize(ch_map, percentile_norm)
        ch_map = upsample_to_img(ch_map, target_hw=(224, 224))
        _save_heat_png(ch_map, ch_dir / f"ch{rank:03d}.png")
        maps.append(ch_map)
        titles.append(f"#{rank} ch={int(c)} E={energies[c]:.4f}")

    cols = min(4, k)
    rows = math.ceil(k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i < k:
            ax.imshow(maps[i], cmap="inferno", vmin=0.0, vmax=1.0)
            ax.set_title(titles[i], fontsize=8)
    fig.tight_layout()
    fig.savefig(ch_dir / "topk_montage.png", dpi=180)
    plt.close(fig)


def save_grid_summary(
    img_rgb: np.ndarray,
    overlay_dict: Dict[str, np.ndarray],
    out_path: Path,
):
    """
    Save one figure containing original image + per-layer overlays.
    """
    panels = [("original", img_rgb)] + [(f"overlay_{k}", v) for k, v in overlay_dict.items()]
    n = len(panels)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.axis("off")
        if i < n:
            title, img = panels[i]
            ax.imshow(img)
            ax.set_title(title, fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def show_inline_image(path: Path):
    try:
        from IPython.display import display
    except Exception:
        return
    try:
        with Image.open(path) as im:
            display(im.copy())
    except Exception:
        return


def main():
    args = parse_args()
    root = Path(args.root)
    out_roots = [Path(args.out_dir)]
    if args.drive_out_dir.strip():
        out_roots.append(Path(args.drive_out_dir))
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]

    samples = _scan_samples(root, args.cls)
    if not samples:
        print("No valid samples found.")
        return

    max_items = min(args.max_items, len(samples))
    samples = samples[:max_items]

    print(f"Found {len(samples)} samples to visualize.")
    print(f"Output roots: {', '.join(str(p) for p in out_roots)}")
    shown = 0
    for cls_name, sample_id, img_path, acts_path in tqdm(samples, desc="Visualize"):
        full_ref_acts_path = None
        if args.delta_ref == "full":
            full_ref_acts_path = Path(args.full_ref_root) / cls_name / f"{sample_id}__acts.pt"

        img_rgb, layer_to_act = load_sample(
            img_path=img_path,
            acts_path=acts_path,
            requested_layers=layers,
            delta_ref=args.delta_ref,
            full_ref_acts_path=full_ref_acts_path,
        )
        if img_rgb is None or layer_to_act is None:
            continue

        sample_out_dirs = []
        for out_root in out_roots:
            sample_out_dir = out_root / cls_name / sample_id
            sample_out_dir.mkdir(parents=True, exist_ok=True)
            sample_out_dirs.append(sample_out_dir)

        overlay_dict: Dict[str, np.ndarray] = {}
        for layer_name, A in layer_to_act.items():
            M = reduce_activation(A, reduce=args.reduce, use_abs=args.use_abs)
            M = robust_normalize(M, percentile=args.percentile_norm)
            M_up = upsample_to_img(M, target_hw=(224, 224))

            overlay = overlay_heatmap(img_rgb, M_up, alpha=args.overlay_alpha)
            overlay_dict[layer_name] = overlay

            for sample_out_dir in sample_out_dirs:
                _save_heat_png(M_up, sample_out_dir / f"heat_{layer_name}.png")
                _save_rgb_png(overlay, sample_out_dir / f"overlay_{layer_name}.png")
                save_topk_channels(
                    A=A,
                    layer_name=layer_name,
                    sample_out_dir=sample_out_dir,
                    topk=args.topk_channels,
                    use_abs=args.use_abs,
                    percentile_norm=args.percentile_norm,
                )

        summary_path = None
        for sample_out_dir in sample_out_dirs:
            _save_rgb_png(img_rgb, sample_out_dir / "original.png")
            if args.mode == "grid":
                summary_path = sample_out_dir / "summary_grid.png"
                save_grid_summary(
                    img_rgb=img_rgb,
                    overlay_dict=overlay_dict,
                    out_path=summary_path,
                )

        if args.show_inline and args.mode == "grid" and summary_path is not None and shown < args.show_max:
            show_inline_image(summary_path)
            shown += 1


if __name__ == "__main__":
    main()
