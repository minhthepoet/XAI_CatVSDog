import argparse
import json
import math
from pathlib import Path

from PIL import Image


def iter_json_files(data_dir: Path):
    for subdir in ("cats", "dogs"):
        root = data_dir / subdir
        if root.exists():
            yield from root.rglob("*.json")


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


def process_json(json_path: Path, out_cat: Path, out_dog: Path):
    saved = 0
    skipped = 0

    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to parse JSON: {json_path} ({exc})")
        return saved, skipped

    source_image = payload.get("source_image")
    if not source_image:
        print(f"[WARN] Missing source_image in JSON: {json_path}")
        return saved, skipped

    image_path = Path(source_image)
    if not image_path.exists():
        print(f"[WARN] source_image not found for {json_path}: {image_path}")
        return saved, skipped

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"[WARN] Failed to open source_image for {json_path}: {image_path} ({exc})")
        return saved, skipped

    width, height = image.size
    category = payload.get("detected_category")
    if category == "cat":
        out_dir = out_cat
    elif category == "dog":
        out_dir = out_dog
    else:
        print(f"[WARN] Unknown detected_category in {json_path}: {category}")
        return saved, skipped

    parts = payload.get("parts", [])
    if not isinstance(parts, list):
        print(f"[WARN] Invalid parts list in {json_path}")
        return saved, skipped

    stem = json_path.stem

    for idx, part in enumerate(parts):
        if not isinstance(part, dict):
            skipped += 1
            continue

        part_name = str(part.get("part_name", "unknown"))
        bbox = clamp_bbox(part.get("box_xyxy_expanded"), width, height)
        if bbox is None:
            skipped += 1
            continue

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue

        masked = Image.new("RGB", (width, height), (0, 0, 0))
        roi = image.crop((x1, y1, x2, y2))
        masked.paste(roi, (x1, y1))

        output_name = f"{stem}__{part_name}__{idx}.png"
        output_path = out_dir / output_name
        masked.save(output_path, format="PNG")
        saved += 1

    return saved, skipped


def main():
    parser = argparse.ArgumentParser(description="Build raw_data_explain from part-detection JSON files.")
    parser.add_argument("--data_dir", required=True, help="Directory containing cats/ and dogs/ JSON files.")
    parser.add_argument("--out_dir", required=True, help="Base output directory.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_base = Path(args.out_dir) / "raw_data_explain"
    out_cat = out_base / "cat"
    out_dog = out_base / "dog"
    out_cat.mkdir(parents=True, exist_ok=True)
    out_dog.mkdir(parents=True, exist_ok=True)

    total_json = 0
    total_saved = 0
    total_skipped = 0

    for json_path in iter_json_files(data_dir):
        total_json += 1
        saved, skipped = process_json(json_path, out_cat, out_dog)
        total_saved += saved
        total_skipped += skipped

    print(f"total_json={total_json}, total_parts_saved={total_saved}, total_parts_skipped={total_skipped}")


if __name__ == "__main__":
    main()
