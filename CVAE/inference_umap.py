import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from CVAE.data_preprocessing import PairedActsDataset, build_image_transform
    from CVAE.model import ConditionalVAE
else:
    from .data_preprocessing import PairedActsDataset, build_image_transform
    from .model import ConditionalVAE


def get_args():
    parser = argparse.ArgumentParser(description="UMAP visualization for CVAE latent space.")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_png", type=str, default="")
    parser.add_argument("--out_npz", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_hw", type=int, default=56)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--normalize_acts", action="store_true")
    parser.add_argument("--no_normalize_acts", dest="normalize_acts", action="store_false")
    parser.set_defaults(normalize_acts=None)
    parser.add_argument("--label_by", type=str, default="part", choices=["part", "class"])
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="euclidean")
    return parser.parse_args()


def infer_from_ckpt_or_cli(args, ckpt_args):
    target_hw = ckpt_args.get("target_hw", args.target_hw)
    latent_dim = ckpt_args.get("latent_dim", args.latent_dim)
    normalize_acts = ckpt_args.get("normalize_acts", args.normalize_acts)
    if normalize_acts is None:
        normalize_acts = True
    return target_hw, latent_dim, normalize_acts


def label_from_sample_id(sample_id: str) -> int:
    if sample_id.startswith("cat"):
        return 0
    if sample_id.startswith("dog"):
        return 1
    return -1


def part_from_sample_id(sample_id: str) -> str:
    chunks = sample_id.split("__")
    if len(chunks) >= 3:
        return chunks[-2]
    return "unknown"


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        import umap
    except Exception:
        print("Error: umap-learn is not installed. Please run: pip install umap-learn")
        raise SystemExit(1)

    ckpt_path = Path(args.ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    exp_root = ckpt_path.parent.parent

    target_hw, latent_dim, normalize_acts = infer_from_ckpt_or_cli(args, ckpt_args)

    model = ConditionalVAE(latent_dim=latent_dim, target_hw=target_hw, y_channels=512).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = PairedActsDataset(
        data_dir=args.data_dir,
        target_hw=target_hw,
        normalize_acts=normalize_acts,
        exp_root=str(exp_root),
        transform=build_image_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    mus = []
    class_labels = []
    part_labels = []
    sample_ids = []
    collected = 0
    max_samples = min(args.max_samples, len(dataset))

    with torch.no_grad():
        for x_img, y_merged, sid in tqdm(loader, total=len(loader), desc="Encode latent"):
            x_img = x_img.to(args.device, non_blocking=True)
            y_merged = y_merged.to(args.device, non_blocking=True)
            _, mu, _ = model(x_img, y_merged)

            mu_np = mu.detach().cpu().numpy()
            mus.append(mu_np)
            for s in sid:
                class_labels.append(label_from_sample_id(s))
                part_labels.append(part_from_sample_id(s))
                sample_ids.append(s)

            collected += mu_np.shape[0]
            if collected >= max_samples:
                break

    z = np.concatenate(mus, axis=0)[:max_samples]
    class_labels = np.array(class_labels[:max_samples], dtype=np.int64)
    part_labels = np.array(part_labels[:max_samples])
    sample_ids = np.array(sample_ids[:max_samples])

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
    )
    z_2d = reducer.fit_transform(z)

    out_png = args.out_png or str(exp_root / "logs" / "latent_umap.png")
    out_npz = args.out_npz or str(exp_root / "logs" / "latent_umap.npz")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    if args.label_by == "class":
        cat_mask = class_labels == 0
        dog_mask = class_labels == 1
        unk_mask = class_labels < 0
        if cat_mask.any():
            plt.scatter(z_2d[cat_mask, 0], z_2d[cat_mask, 1], s=10, alpha=0.7, label="cat")
        if dog_mask.any():
            plt.scatter(z_2d[dog_mask, 0], z_2d[dog_mask, 1], s=10, alpha=0.7, label="dog")
        if unk_mask.any():
            plt.scatter(z_2d[unk_mask, 0], z_2d[unk_mask, 1], s=10, alpha=0.7, label="unknown")
    else:
        unique_parts = sorted(set(part_labels.tolist()))
        for part in unique_parts:
            mask = part_labels == part
            plt.scatter(z_2d[mask, 0], z_2d[mask, 1], s=10, alpha=0.7, label=part)
    plt.title(f"CVAE Latent UMAP by {args.label_by} (N={z.shape[0]})")
    plt.xlabel("UMAP dim 1")
    plt.ylabel("UMAP dim 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

    np.savez_compressed(
        out_npz,
        z_mu=z,
        z_umap=z_2d,
        z_vis2d=z_2d,
        labels_class=class_labels,
        labels_part=part_labels,
        sample_ids=sample_ids,
        ckpt_path=str(ckpt_path),
        method="umap",
    )
    print(f"Saved UMAP plot: {out_png}")
    print(f"Saved latent arrays: {out_npz}")


if __name__ == "__main__":
    main()

