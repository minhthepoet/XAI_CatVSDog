import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

if __package__ is None or __package__ == "":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from VAE.data_preprocessing import build_image_transform
    from VAE.model_vanilla import VanillaVAE
else:
    from .data_preprocessing import build_image_transform
    from .model_vanilla import VanillaVAE


class ImageOnlyDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.root = Path(data_dir)
        self.transform = transform if transform is not None else build_image_transform()
        self.samples = self._scan_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No image samples found under: {self.root}")

    def _scan_samples(self):
        out = []
        for cls_name in ["cat", "dog"]:
            cls_dir = self.root / cls_name
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.glob("*.png")):
                sample_id = img_path.stem
                out.append((img_path, sample_id))
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, sample_id = self.samples[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        x_img = self.transform(im)
        return x_img, sample_id


def get_args():
    parser = argparse.ArgumentParser(description="Raw latent visualization for Vanilla VAE.")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_png", type=str, default="")
    parser.add_argument("--out_npz", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--target_hw", type=int, default=56)
    parser.add_argument("--label_by", type=str, default="part", choices=["part", "class"])
    parser.add_argument("--dims", type=int, default=2, choices=[2, 3])
    return parser.parse_args()


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

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    exp_root = ckpt_path.parent.parent

    latent_dim = int(ckpt_args.get("latent_dim", args.latent_dim))
    target_hw = int(ckpt_args.get("target_hw", args.target_hw))
    merge_mode = str(ckpt_args.get("merge_mode", "merge"))
    recon_loss = str(ckpt_args.get("recon_loss", "mse"))

    if latent_dim < 2:
        raise RuntimeError(f"latent_dim={latent_dim} is too small for plotting.")
    if latent_dim > 3:
        raise RuntimeError(
            f"latent_dim={latent_dim} > 3. Use inference_pca.py / inference_umap.py / inference_tSNE.py instead."
        )
    if args.dims == 3 and latent_dim < 3:
        raise RuntimeError(f"--dims=3 requires latent_dim>=3, got {latent_dim}.")

    model = VanillaVAE(
        latent_dim=latent_dim,
        target_hw=target_hw,
        out_channels=512,
        recon_loss=recon_loss,
        merge_mode=merge_mode,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = ImageOnlyDataset(data_dir=args.data_dir, transform=build_image_transform())
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
    max_samples = min(args.max_samples, len(dataset))
    collected = 0

    with torch.no_grad():
        for x_img, sid in tqdm(loader, total=len(loader), desc="Encode latent"):
            x_img = x_img.to(device, non_blocking=True)
            mu, _ = model.encode(x_img)
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

    out_png = args.out_png or str(exp_root / "logs" / "latent_raw_vanilla.png")
    out_npz = args.out_npz or str(exp_root / "logs" / "latent_raw_vanilla.npz")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)

    if args.dims == 2:
        z_vis = z[:, :2]
        plt.figure(figsize=(8, 6))
        if args.label_by == "class":
            cat_mask = class_labels == 0
            dog_mask = class_labels == 1
            unk_mask = class_labels < 0
            if cat_mask.any():
                plt.scatter(z_vis[cat_mask, 0], z_vis[cat_mask, 1], s=10, alpha=0.7, label="cat")
            if dog_mask.any():
                plt.scatter(z_vis[dog_mask, 0], z_vis[dog_mask, 1], s=10, alpha=0.7, label="dog")
            if unk_mask.any():
                plt.scatter(z_vis[unk_mask, 0], z_vis[unk_mask, 1], s=10, alpha=0.7, label="unknown")
        else:
            unique_parts = sorted(set(part_labels.tolist()))
            for part in unique_parts:
                mask = part_labels == part
                plt.scatter(z_vis[mask, 0], z_vis[mask, 1], s=10, alpha=0.7, label=part)
        plt.title(f"Vanilla VAE Raw Latent (2D) by {args.label_by} (N={z_vis.shape[0]})")
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close()
        np.savez_compressed(
            out_npz,
            z_mu=z,
            z_latent=z_vis,
            z_vis2d=z_vis,
            labels_class=class_labels,
            labels_part=part_labels,
            sample_ids=sample_ids,
            ckpt_path=str(ckpt_path),
            method="raw_latent_2d",
        )
    else:
        z_vis = z[:, :3]
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        if args.label_by == "class":
            cat_mask = class_labels == 0
            dog_mask = class_labels == 1
            unk_mask = class_labels < 0
            if cat_mask.any():
                ax.scatter(z_vis[cat_mask, 0], z_vis[cat_mask, 1], z_vis[cat_mask, 2], s=10, alpha=0.7, label="cat")
            if dog_mask.any():
                ax.scatter(z_vis[dog_mask, 0], z_vis[dog_mask, 1], z_vis[dog_mask, 2], s=10, alpha=0.7, label="dog")
            if unk_mask.any():
                ax.scatter(z_vis[unk_mask, 0], z_vis[unk_mask, 1], z_vis[unk_mask, 2], s=10, alpha=0.7, label="unknown")
        else:
            unique_parts = sorted(set(part_labels.tolist()))
            for part in unique_parts:
                mask = part_labels == part
                ax.scatter(z_vis[mask, 0], z_vis[mask, 1], z_vis[mask, 2], s=10, alpha=0.7, label=part)
        ax.set_title(f"Vanilla VAE Raw Latent (3D) by {args.label_by} (N={z_vis.shape[0]})")
        ax.set_xlabel("z[0]")
        ax.set_ylabel("z[1]")
        ax.set_zlabel("z[2]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        np.savez_compressed(
            out_npz,
            z_mu=z,
            z_latent=z_vis,
            z_vis3d=z_vis,
            labels_class=class_labels,
            labels_part=part_labels,
            sample_ids=sample_ids,
            ckpt_path=str(ckpt_path),
            method="raw_latent_3d",
        )

    print(f"Saved plot: {out_png}")
    print(f"Saved latent arrays: {out_npz}")


if __name__ == "__main__":
    main()

