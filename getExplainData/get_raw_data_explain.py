import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def iter_json_files(data_dir: Path, input_subdir: str):
    root = data_dir / input_subdir
    if not root.exists():
        return []
    return sorted(root.rglob("*.json"))


def clamp_bbox(box, width: int, height: int):
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1 = math.floor(float(box[0]))
        y1 = math.floor(float(box[1]))
        x2 = math.ceil(float(box[2]))
        y2 = math.ceil(float(box[3]))
    except (TypeError, ValueError):
        return None

    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def collect_candidates_from_json(json_path: Path):
    candidates = []
    skipped_parts = 0
    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to parse JSON: {json_path} ({exc})")
        return candidates, skipped_parts

    source_image = payload.get("source_image")
    if not source_image:
        print(f"[WARN] Missing source_image in JSON: {json_path}")
        return candidates, skipped_parts

    image_path = Path(source_image)
    if not image_path.exists():
        print(f"[WARN] source_image not found for {json_path}: {image_path}")
        return candidates, skipped_parts

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"[WARN] Failed to open source_image for {json_path}: {image_path} ({exc})")
        return candidates, skipped_parts

    width, height = image.size

    parts = payload.get("parts", [])
    if not isinstance(parts, list):
        print(f"[WARN] Invalid parts list in {json_path}")
        return candidates, skipped_parts

    stem = json_path.stem

    for idx, part in enumerate(parts):
        if not isinstance(part, dict):
            skipped_parts += 1
            continue

        part_name = str(part.get("part_name", "unknown"))
        bbox = clamp_bbox(part.get("box_xyxy_expanded"), width, height)
        if bbox is None:
            skipped_parts += 1
            continue

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            skipped_parts += 1
            continue

        candidates.append(
            {
                "json_path": json_path,
                "image_path": image_path,
                "stem": stem,
                "part_name": part_name,
                "part_idx": idx,
                "bbox": (x1, y1, x2, y2),
                "size": (width, height),
            }
        )

    return candidates, skipped_parts


def build_unique_output_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    dup_idx = 1
    while True:
        candidate = parent / f"{stem}__dup{dup_idx}{suffix}"
        if not candidate.exists():
            return candidate
        dup_idx += 1


def process_category(
    data_dir: Path,
    out_dir: Path,
    input_subdir: str,
    output_subdir: str,
    target_count: int,
    rng: random.Random,
):
    out_category = out_dir / output_subdir
    out_category.mkdir(parents=True, exist_ok=True)

    json_files = iter_json_files(data_dir, input_subdir)
    all_candidates = []
    total_skipped_parts = 0

    for json_path in json_files:
        candidates, skipped_parts = collect_candidates_from_json(json_path)
        all_candidates.extend(candidates)
        total_skipped_parts += skipped_parts

    available = len(all_candidates)
    if available == 0:
        print(f"[WARN] No valid candidates found for {output_subdir}.")
        return {
            "json_count": len(json_files),
            "available": 0,
            "saved": 0,
            "skipped_parts": total_skipped_parts,
        }

    k = min(target_count, available)
    selected = rng.sample(all_candidates, k=k)
    saved = 0

    for item in tqdm(selected, desc=f"save_{output_subdir}", unit="img"):
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as exc:
            print(f"[WARN] Failed to reopen source image: {item['image_path']} ({exc})")
            continue

        x1, y1, x2, y2 = item["bbox"]
        width, height = item["size"]
        masked = Image.new("RGB", (width, height), (0, 0, 0))
        roi = image.crop((x1, y1, x2, y2))
        masked.paste(roi, (x1, y1))

        output_name = f"{item['stem']}__{item['part_name']}__{item['part_idx']}.png"
        output_path = build_unique_output_path(out_category / output_name)
        try:
            masked.save(output_path, format="PNG")
            saved += 1
        except Exception as exc:
            print(f"[WARN] Failed to save image: {output_path} ({exc})")

    if available < target_count:
        print(
            f"[WARN] {output_subdir}: requested={target_count} but only {available} valid unique candidates found."
        )

    return {
        "json_count": len(json_files),
        "available": available,
        "saved": saved,
        "skipped_parts": total_skipped_parts,
    }


def main():
    parser = argparse.ArgumentParser(description="Build raw_data_explain with random non-repeated part selections.")
    parser.add_argument("--data_dir", required=True, help="Directory containing cats/ and dogs/ JSON files.")
    parser.add_argument("--out_dir", required=True, help="Output root directory containing cat/ and dog/.")
    parser.add_argument("--target_per_class", type=int, default=7000, help="Number of images to save per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Unused compatibility argument. Raw extraction keeps original source image size.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Required order: finish cat first, then dog. No parallelism.
    cat_stats = process_category(
        data_dir=data_dir,
        out_dir=out_base,
        input_subdir="cats",
        output_subdir="cat",
        target_count=args.target_per_class,
        rng=rng,
    )
    dog_stats = process_category(
        data_dir=data_dir,
        out_dir=out_base,
        input_subdir="dogs",
        output_subdir="dog",
        target_count=args.target_per_class,
        rng=rng,
    )

    total_json = cat_stats["json_count"] + dog_stats["json_count"]
    total_saved = cat_stats["saved"] + dog_stats["saved"]
    total_skipped = cat_stats["skipped_parts"] + dog_stats["skipped_parts"]
    print(
        f"done total_json={total_json} total_parts_saved={total_saved} total_parts_skipped={total_skipped} "
        f"cat_saved={cat_stats['saved']}/{args.target_per_class} dog_saved={dog_stats['saved']}/{args.target_per_class}"
    )


if __name__ == "__main__":
    main()
