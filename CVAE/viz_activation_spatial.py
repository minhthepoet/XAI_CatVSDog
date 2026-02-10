import argparse
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

if __package__ is None or __package__ == "":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from CVAE.data_preprocessing import build_image_transform
    from CVAE.model import ConditionalVAE
else:
    from .data_preprocessing import build_image_transform
    from .model import ConditionalVAE


LAYER_ORDER = ["stem", "layer1", "layer2", "layer3", "layer4"]
LAYER_HW = {
    "stem": 56,
    "layer1": 56,
    "layer2": 28,
    "layer3": 14,
    "layer4": 7,
}
LAYER_CH = {
    "stem": 32,
    "layer1": 32,
    "layer2": 64,
    "layer3": 128,
    "layer4": 256,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare CVAE-generated activations vs ground-truth activations on original/bbox images."
    )
    parser.add_argument("--class", dest="class_name", choices=["cat", "dog"], required=True)
    parser.add_argument("--id", dest="item_id", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--data_explain_root", type=str, default="/content/drive/MyDrive/XAI_DogVSCat/data_explain")
    parser.add_argument("--data_test_root", type=str, default="/content/drive/MyDrive/XAI_DogVSCat/data_test")
    parser.add_argument(
        "--bbox_root",
        type=str,
        default="/content/drive/MyDrive/XAI_DogVSCat/out_boundingbox_dir_test",
    )
    parser.add_argument("--out_dir", type=str, default="/content/drive/MyDrive/XAI_DogVSCat/viz_out_cvae_compare")
    parser.add_argument("--layers", type=str, default="stem,layer1,layer2,layer3,layer4")
    parser.add_argument("--reduce", type=str, choices=["mean", "max", "l2"], default="mean")
    parser.add_argument("--use_abs", action="store_true")
    parser.add_argument("--percentile_norm", type=float, default=99.5)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def find_class_dir(root: Path, class_name: str) -> Path:
    candidates = [class_name, class_name + "s", class_name.capitalize(), class_name.capitalize() + "s"]
    for c in candidates:
        p = root / c
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Cannot find class folder for '{class_name}' under {root}")


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def to_chw(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 4:
        if t.shape[0] != 1:
            raise ValueError(f"Expected [1,C,H,W], got {tuple(t.shape)}")
        t = t.squeeze(0)
    if t.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(t.shape)}")
    return t.float().cpu()


def merge_acts_dict(acts: Dict[str, torch.Tensor], target_hw: int) -> torch.Tensor:
    chunks = []
    for layer in LAYER_ORDER:
        if layer not in acts:
            raise KeyError(f"Missing layer '{layer}' in acts.")
        a = to_chw(acts[layer]).unsqueeze(0)
        a = F.interpolate(a, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
        chunks.append(a.squeeze(0))
    return torch.cat(chunks, dim=0).float()


def split_merged_to_layers(merged: torch.Tensor) -> Dict[str, torch.Tensor]:
    # merged: [1,512,H,W] or [512,H,W]
    if merged.ndim == 3:
        merged = merged.unsqueeze(0)
    out = {}
    c0 = 0
    for layer in LAYER_ORDER:
        c = LAYER_CH[layer]
        x = merged[:, c0:c0 + c]
        c0 += c
        hw = LAYER_HW[layer]
        if x.shape[-2:] != (hw, hw):
            x = F.interpolate(x, size=(hw, hw), mode="bilinear", align_corners=False)
        out[layer] = x.squeeze(0).cpu()
    return out


def reduce_activation(A: torch.Tensor, reduce: str, use_abs: bool) -> np.ndarray:
    if use_abs:
        A = A.abs()
    if reduce == "mean":
        M = A.mean(dim=0)
    elif reduce == "max":
        M = A.max(dim=0).values
    else:  # l2
        M = torch.sqrt(torch.clamp((A ** 2).mean(dim=0), min=0.0))
    return M.cpu().numpy().astype(np.float32)


def robust_normalize(M: np.ndarray, percentile: float) -> np.ndarray:
    M = M.astype(np.float32)
    M = M - M.min()
    cap = np.percentile(M, np.clip(percentile, 1.0, 100.0))
    if cap <= 1e-12:
        return np.zeros_like(M, dtype=np.float32)
    M = np.clip(M, 0.0, cap) / cap
    return M


def upsample_to_img(M: np.ndarray, target_hw=(224, 224)) -> np.ndarray:
    t = torch.from_numpy(M).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def overlay_heatmap(img_rgb: np.ndarray, heat: np.ndarray, alpha: float) -> np.ndarray:
    heat_rgb = cm.get_cmap("viridis")(np.clip(heat, 0.0, 1.0))[..., :3].astype(np.float32)
    a = np.clip(alpha, 0.0, 1.0)
    out = (1.0 - a) * img_rgb + a * heat_rgb
    return np.clip(out, 0.0, 1.0)


def save_rgb(path: Path, arr01: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)).save(path)


def save_heat(path: Path, heat01: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cm.get_cmap("inferno")(np.clip(heat01, 0.0, 1.0))[..., :3].astype(np.float32)
    save_rgb(path, rgb)


def save_compare_grid(
    out_path: Path,
    src_img: np.ndarray,
    layers: List[str],
    overlays_gt: Dict[str, np.ndarray],
    overlays_pred: Dict[str, np.ndarray],
    title_prefix: str,
):
    cols = len(layers) + 1
    fig, axes = plt.subplots(2, cols, figsize=(3.2 * cols, 6.2))
    axes[0, 0].imshow(src_img)
    axes[0, 0].set_title(f"{title_prefix}\ninput")
    axes[1, 0].imshow(src_img)
    axes[1, 0].set_title(f"{title_prefix}\ninput")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for i, layer in enumerate(layers, start=1):
        axes[0, i].imshow(overlays_gt[layer])
        axes[0, i].set_title(f"GT {layer}")
        axes[0, i].axis("off")

        axes[1, i].imshow(overlays_pred[layer])
        axes[1, i].set_title(f"CVAE {layer}")
        axes[1, i].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_part_list(explain_cls_dir: Path, class_name: str, item_id: str) -> List[Tuple[str, Path, Path]]:
    base = f"{class_name}.{item_id}"
    items = []
    for p in sorted(explain_cls_dir.glob(f"{base}__*.png")):
        sid = p.stem
        acts = explain_cls_dir / f"{sid}__acts.pt"
        if acts.exists():
            items.append((sid, p, acts))
    return items


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]

    ckpt_path = Path(args.ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    target_hw = int(ckpt_args.get("target_hw", 56))
    latent_dim = int(ckpt_args.get("latent_dim", 64))
    normalize_acts = bool(ckpt_args.get("normalize_acts", False))

    exp_root = ckpt_path.parent.parent
    stats_mean = None
    stats_std = None
    if normalize_acts:
        stats_path = exp_root / "acts_stats_merged.pt"
        if stats_path.exists():
            payload = torch.load(stats_path, map_location="cpu")
            stats_mean = payload.get("mean", None)
            stats_std = payload.get("std", None)
            if torch.is_tensor(stats_mean):
                stats_mean = stats_mean.float().view(1, -1, 1, 1)
            if torch.is_tensor(stats_std):
                stats_std = stats_std.float().view(1, -1, 1, 1)

    model = ConditionalVAE(latent_dim=latent_dim, target_hw=target_hw, y_channels=512).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    explain_cls_dir = Path(args.data_explain_root) / args.class_name
    if not explain_cls_dir.is_dir():
        raise FileNotFoundError(f"Missing explain class dir: {explain_cls_dir}")
    test_cls_dir = find_class_dir(Path(args.data_test_root), args.class_name)
    bbox_cls_dir = find_class_dir(Path(args.bbox_root), args.class_name)

    base_name = f"{args.class_name}.{args.item_id}"
    original_img_path = test_cls_dir / f"{base_name}.png"
    bbox_img_path = bbox_cls_dir / f"{base_name}_boxes_labeled.png"
    if not original_img_path.exists():
        raise FileNotFoundError(f"Missing original image: {original_img_path}")
    if not bbox_img_path.exists():
        raise FileNotFoundError(f"Missing bbox image: {bbox_img_path}")

    img_original = load_rgb(original_img_path)
    img_bbox = load_rgb(bbox_img_path)
    transform = build_image_transform()

    parts = build_part_list(explain_cls_dir, args.class_name, args.item_id)
    if not parts:
        raise RuntimeError(f"No part samples found for base id '{base_name}' in {explain_cls_dir}")

    out_root = Path(args.out_dir) / base_name
    out_root.mkdir(parents=True, exist_ok=True)
    save_rgb(out_root / "original_image.png", img_original)
    save_rgb(out_root / "bbox_image.png", img_bbox)

    print(f"Found {len(parts)} part samples for {base_name}")
    print(f"Output: {out_root}")

    for sample_id, part_img_path, part_acts_path in tqdm(parts, desc="Part compare"):
        part_dir = out_root / sample_id
        part_dir.mkdir(parents=True, exist_ok=True)

        # Load GT part activation.
        acts_obj = torch.load(part_acts_path, map_location="cpu")
        gt_merged = merge_acts_dict(acts_obj, target_hw=target_hw).unsqueeze(0)  # [1,512,56,56]
        gt_layers = split_merged_to_layers(gt_merged)

        # CVAE generation from part-masked image (deterministic: z=0).
        part_img_rgb = load_rgb(part_img_path)
        x = transform(Image.fromarray(np.clip(part_img_rgb * 255.0, 0, 255).astype(np.uint8))).unsqueeze(0).to(device)
        with torch.no_grad():
            hx = model.image_encoder(x)
            z = torch.zeros((x.size(0), model.latent_dim), device=device, dtype=hx.dtype)
            pred_merged = model.decode(hx, z).detach().cpu()
            if normalize_acts and stats_mean is not None and stats_std is not None:
                pred_merged = pred_merged * (stats_std + 1e-6) + stats_mean
            pred_layers = split_merged_to_layers(pred_merged)

        save_rgb(part_dir / "part_input.png", part_img_rgb)

        overlays_gt_orig: Dict[str, np.ndarray] = {}
        overlays_gt_bbox: Dict[str, np.ndarray] = {}
        overlays_pred_orig: Dict[str, np.ndarray] = {}
        overlays_pred_bbox: Dict[str, np.ndarray] = {}

        for layer in layers:
            if layer not in gt_layers or layer not in pred_layers:
                continue

            gt_map = reduce_activation(gt_layers[layer], args.reduce, args.use_abs)
            gt_map = robust_normalize(gt_map, args.percentile_norm)
            gt_map = upsample_to_img(gt_map, target_hw=(224, 224))

            pred_map = reduce_activation(pred_layers[layer], args.reduce, args.use_abs)
            pred_map = robust_normalize(pred_map, args.percentile_norm)
            pred_map = upsample_to_img(pred_map, target_hw=(224, 224))

            save_heat(part_dir / f"heat_gt_{layer}.png", gt_map)
            save_heat(part_dir / f"heat_cvae_{layer}.png", pred_map)

            ov_gt_orig = overlay_heatmap(img_original, gt_map, args.overlay_alpha)
            ov_gt_bbox = overlay_heatmap(img_bbox, gt_map, args.overlay_alpha)
            ov_pred_orig = overlay_heatmap(img_original, pred_map, args.overlay_alpha)
            ov_pred_bbox = overlay_heatmap(img_bbox, pred_map, args.overlay_alpha)

            overlays_gt_orig[layer] = ov_gt_orig
            overlays_gt_bbox[layer] = ov_gt_bbox
            overlays_pred_orig[layer] = ov_pred_orig
            overlays_pred_bbox[layer] = ov_pred_bbox

            save_rgb(part_dir / f"overlay_original_gt_{layer}.png", ov_gt_orig)
            save_rgb(part_dir / f"overlay_bbox_gt_{layer}.png", ov_gt_bbox)
            save_rgb(part_dir / f"overlay_original_cvae_{layer}.png", ov_pred_orig)
            save_rgb(part_dir / f"overlay_bbox_cvae_{layer}.png", ov_pred_bbox)

        valid_layers = [l for l in layers if l in overlays_gt_orig and l in overlays_pred_orig]
        if valid_layers:
            save_compare_grid(
                out_path=part_dir / "summary_original.png",
                src_img=img_original,
                layers=valid_layers,
                overlays_gt=overlays_gt_orig,
                overlays_pred=overlays_pred_orig,
                title_prefix="Original",
            )
            save_compare_grid(
                out_path=part_dir / "summary_bbox.png",
                src_img=img_bbox,
                layers=valid_layers,
                overlays_gt=overlays_gt_bbox,
                overlays_pred=overlays_pred_bbox,
                title_prefix="BBox",
            )

    print("Done.")


if __name__ == "__main__":
    main()

