import argparse
import json
import math
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


def process_json_and_save(json_path: Path, out_category: Path, pbar, saved_name_counts):
    saved = 0
    skipped_parts = 0
    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to parse JSON: {json_path} ({exc})")
        return saved, skipped_parts

    source_image = payload.get("source_image")
    if not source_image:
        print(f"[WARN] Missing source_image in JSON: {json_path}")
        return saved, skipped_parts

    image_path = Path(source_image)
    if not image_path.exists():
        print(f"[WARN] source_image not found for {json_path}: {image_path}")
        return saved, skipped_parts

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"[WARN] Failed to open source_image for {json_path}: {image_path} ({exc})")
        return saved, skipped_parts

    width, height = image.size

    parts = payload.get("parts", [])
    if not isinstance(parts, list):
        print(f"[WARN] Invalid parts list in {json_path}")
        return saved, skipped_parts

    stem = json_path.stem

    for idx, part in enumerate(parts):
        if pbar.n >= pbar.total:
            break
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

        masked = Image.new("RGB", (width, height), (0, 0, 0))
        roi = image.crop((x1, y1, x2, y2))
        masked.paste(roi, (x1, y1))

        base_name = f"{stem}__{part_name}__{idx}"
        name_count = saved_name_counts.get(base_name, 0)
        saved_name_counts[base_name] = name_count + 1
        if name_count == 0:
            output_name = f"{base_name}.png"
        else:
            output_name = f"{base_name}__dup{name_count}.png"

        output_path = out_category / output_name
        try:
            masked.save(output_path, format="PNG")
            saved += 1
            pbar.update(1)
        except Exception as exc:
            print(f"[WARN] Failed to save image: {output_path} ({exc})")

    return saved, skipped_parts


def process_category(
    data_dir: Path,
    out_dir: Path,
    input_subdir: str,
    output_subdir: str,
    target_count: int,
):
    out_category = out_dir / output_subdir
    out_category.mkdir(parents=True, exist_ok=True)

    json_files = iter_json_files(data_dir, input_subdir)
    saved_name_counts = {}
    saved = 0
    total_skipped_parts = 0

    with tqdm(total=target_count, desc=f"save_{output_subdir}", unit="img") as pbar:
        for json_path in json_files:
            if saved >= target_count:
                break
            saved_now, skipped_parts = process_json_and_save(
                json_path=json_path,
                out_category=out_category,
                pbar=pbar,
                saved_name_counts=saved_name_counts,
            )
            saved += saved_now
            total_skipped_parts += skipped_parts

    if saved < target_count:
        print(f"[WARN] {output_subdir}: requested={target_count} but only saved={saved}.")

    return {
        "json_count": len(json_files),
        "saved": saved,
        "skipped_parts": total_skipped_parts,
    }


def main():
    parser = argparse.ArgumentParser(description="Build raw_data_explain quickly by streaming valid parts.")
    parser.add_argument("--data_dir", required=True, help="Directory containing cats/ and dogs/ JSON files.")
    parser.add_argument("--out_dir", required=True, help="Output root directory containing cat/ and dog/.")
    parser.add_argument("--target_per_class", type=int, default=7000, help="Number of images to save per class.")
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

    # Required order: finish cat first, then dog. No parallelism.
    cat_stats = process_category(
        data_dir=data_dir,
        out_dir=out_base,
        input_subdir="cats",
        output_subdir="cat",
        target_count=args.target_per_class,
    )
    dog_stats = process_category(
        data_dir=data_dir,
        out_dir=out_base,
        input_subdir="dogs",
        output_subdir="dog",
        target_count=args.target_per_class,
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
