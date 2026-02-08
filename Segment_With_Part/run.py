#!/usr/bin/env python3
from __future__ import annotations

"""
Dataset runner for pet part bounding boxes.

Example:
python Segment_With_Part/run.py \
  --data_dir /path/to/dataCatVSDog \
  --out_dir /path/to/out_boundingbox_dir \
  --dino_config Segment_With_Part/groundingdino/GroundingDINO_SwinT_OGC.py \
  --dino_checkpoint /path/to/groundingdino_swint_ogc.pth \
  --device cuda \
  --box_threshold 0.16 \
  --text_threshold 0.16 \
  --topk 10 \
  --nms_iou 0.55 \
  --box_expand 0.2 \
  --skip_existing
"""

import json
import traceback

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image

from config import get_args
from main import (
    GroundingDinoEngine,
    detect_dog_or_cat,
    ensure_device,
    run_part_box_pipeline,
    save_labeled_boxes_image,
)


def _collect_class_images(root: Path, class_names: Sequence[str], allowed: set[str]) -> List[Path]:
    images: List[Path] = []
    for cls_name in class_names:
        cls_dir = root / cls_name
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in allowed:
                images.append(p)
    return images


def resolve_path(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p

    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_candidate = (repo_root / p).resolve()
    if repo_candidate.exists():
        return repo_candidate

    # Return repo-relative default for clearer error message.
    return repo_candidate


def collect_dataset_images(data_dir: Path, exts: Sequence[str]) -> List[Path]:
    allowed = {f".{e.lower().lstrip('.')}" for e in exts}
    images: List[Path] = []

    # New layout (preferred): data/train/{cats,dogs} and data/test/{cats,dogs}
    new_train = data_dir / "data" / "train"
    new_test = data_dir / "data" / "test"
    if (new_train / "cats").exists() and (new_train / "dogs").exists():
        images.extend(_collect_class_images(new_train, ("cats", "dogs"), allowed))
        if (new_test / "cats").exists() and (new_test / "dogs").exists():
            images.extend(_collect_class_images(new_test, ("cats", "dogs"), allowed))
        images.sort(key=lambda x: str(x).lower())
        return images

    # Backward-compatible fallback: PetImages/{Cat,Dog}
    pet_root = data_dir / "PetImages"
    if (pet_root / "Cat").exists() and (pet_root / "Dog").exists():
        images.extend(_collect_class_images(pet_root, ("Cat", "Dog"), allowed))
        images.sort(key=lambda x: str(x).lower())
        return images

    raise FileNotFoundError(
        "Could not find dataset class folders.\n"
        "Expected one of:\n"
        f"- {data_dir / 'data' / 'train'} with cats/ and dogs/\n"
        f"- {data_dir / 'PetImages'} with Cat/ and Dog/"
    )


def slice_images(images: List[Path], start_idx: int, end_idx: int) -> List[Path]:
    if not images:
        return []
    start = max(0, start_idx)
    if end_idx < 0:
        end = len(images) - 1
    else:
        end = min(end_idx, len(images) - 1)
    if start > end:
        return []
    return images[start : end + 1]


def should_skip(json_path: Path, overlay_path: Path, skip_existing: bool, write_mode: str) -> bool:
    if not skip_existing:
        return False
    if write_mode == "json_only":
        return json_path.exists()
    if write_mode == "overlay_only":
        return overlay_path.exists()
    return json_path.exists() and overlay_path.exists()


def main() -> None:
    args = get_args()
    repo_root = Path(__file__).resolve().parent

    data_dir = resolve_path(args.data_dir, repo_root)
    out_dir = resolve_path(args.out_dir, repo_root)
    dino_config = resolve_path(args.dino_config, repo_root)
    dino_checkpoint = resolve_path(args.dino_checkpoint, repo_root)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    if not dino_config.exists():
        raise FileNotFoundError(f"dino_config not found: {dino_config}")
    if not dino_checkpoint.exists():
        raise FileNotFoundError(f"dino_checkpoint not found: {dino_checkpoint}")

    out_dir.mkdir(parents=True, exist_ok=True)

    device = ensure_device(args.device)
    dino = GroundingDinoEngine(str(dino_config), str(dino_checkpoint), device)

    all_images = collect_dataset_images(data_dir, args.exts)
    run_images = slice_images(all_images, args.start_idx, args.end_idx)
    print(f"Dataset images discovered: {len(all_images)} from {data_dir}", flush=True)

    summary = {
        "total_images_found": len(all_images),
        "images_selected": len(run_images),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "failures": [],
        "device": device,
    }

    iterable = enumerate(run_images, start=1)
    if tqdm is not None:
        iterable = tqdm(iterable, total=len(run_images), desc="Bounding boxes", unit="img")

    for idx, image_path in iterable:
        rel = image_path.relative_to(data_dir)
        out_subdir = out_dir / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        stem = image_path.stem
        json_path = out_subdir / f"{stem}.json"
        overlay_path = out_subdir / f"{stem}_boxes_labeled.png"

        status_prefix = f"[{idx}/{len(run_images)}] {rel}"
        if tqdm is None:
            print(f"{status_prefix} -> processing", flush=True)
        else:
            iterable.set_postfix_str(str(rel))

        if should_skip(json_path, overlay_path, args.skip_existing, args.write_mode):
            summary["skipped"] += 1
            if tqdm is None:
                print(f"{status_prefix} -> skipped(existing)", flush=True)
            continue

        try:
            image = Image.open(image_path).convert("RGB")

            category, object_bbox, object_score = detect_dog_or_cat(
                dino=dino,
                image_path=str(image_path),
                image_pil=image,
                box_threshold=min(args.box_threshold, 0.2),
                text_threshold=min(args.text_threshold, 0.2),
            )

            selected_parts, dropped_parts = run_part_box_pipeline(
                dino=dino,
                image_path=str(image_path),
                image_pil=image,
                category=category,
                object_bbox=object_bbox,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                topk=args.topk,
                nms_iou=args.nms_iou,
                box_expand=args.box_expand,
            )

            result = {
                "source_image": str(image_path),
                "detected_category": category,
                "primary_object": {
                    "category": category,
                    "confidence": object_score,
                    "bbox_xyxy": object_bbox,
                },
                "parts": [asdict(p) for p in selected_parts],
                "dropped_parts": dropped_parts,
                "output_files": {
                    "json": str(json_path),
                    "overlay": str(overlay_path),
                },
            }

            if args.write_mode in ("json_only", "both"):
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)

            if args.write_mode in ("overlay_only", "both"):
                save_labeled_boxes_image(image=image, parts=selected_parts, output_path=overlay_path)

            summary["processed"] += 1
            if tqdm is None:
                print(f"{status_prefix} -> done (parts={len(selected_parts)})", flush=True)

        except Exception as exc:
            summary["failed"] += 1
            summary["failures"].append(
                {
                    "image": str(image_path),
                    "error": str(exc),
                    "trace": traceback.format_exc(limit=1),
                }
            )
            if tqdm is None:
                print(f"{status_prefix} -> failed: {exc}", flush=True)
            continue

    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
