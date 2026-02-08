#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
except ImportError as exc:
    raise RuntimeError("PyTorch is required. Install dependencies from requirements.txt.") from exc


@dataclass
class SizePrior:
    min_area_ratio: float = 0.0005
    max_area_ratio: float = 0.2


@dataclass
class AspectPrior:
    min_ar: float = 0.2
    max_ar: float = 5.0


@dataclass
class PartSpec:
    part_name: str
    prompt_phrases: List[str]
    expected_count: int = 1
    size_prior: SizePrior = field(default_factory=SizePrior)
    aspect_prior: AspectPrior = field(default_factory=AspectPrior)


@dataclass
class CandidateBox:
    box_xyxy: List[float]
    score: float
    phrase: str
    part_name: str
    final_score: float


@dataclass
class SelectedPart:
    part_name: str
    prompt_used: str
    box_xyxy: List[float]
    box_xyxy_expanded: List[float]
    box_score: float


PART_TAXONOMY: Dict[str, List[PartSpec]] = {
    "dog": [
        PartSpec("left_eye", ["dog left eye", "left eye of the dog"]),
        PartSpec("right_eye", ["dog right eye", "right eye of the dog"]),
        PartSpec("nose", ["dog nose", "dog snout nose"]),
        PartSpec("mouth", ["dog mouth", "dog muzzle"]),
        PartSpec("left_ear", ["dog left ear", "left ear of the dog"]),
        PartSpec("right_ear", ["dog right ear", "right ear of the dog"]),
        PartSpec(
            "body",
            ["dog body", "dog torso", "body of the dog"],
            size_prior=SizePrior(min_area_ratio=0.08, max_area_ratio=0.98),
            aspect_prior=AspectPrior(min_ar=0.35, max_ar=5.0),
        ),
        PartSpec(
            "tail",
            ["dog tail", "tail of the dog"],
            size_prior=SizePrior(min_area_ratio=0.001, max_area_ratio=0.12),
            aspect_prior=AspectPrior(min_ar=0.08, max_ar=8.0),
        ),
    ],
    "cat": [
        PartSpec("left_eye", ["cat left eye", "left eye of the cat"]),
        PartSpec("right_eye", ["cat right eye", "right eye of the cat"]),
        PartSpec("nose", ["cat nose"]),
        PartSpec("mouth", ["cat mouth"]),
        PartSpec("left_ear", ["cat left ear"]),
        PartSpec("right_ear", ["cat right ear"]),
        PartSpec(
            "body",
            ["cat body", "cat torso", "body of the cat"],
            size_prior=SizePrior(min_area_ratio=0.08, max_area_ratio=0.98),
            aspect_prior=AspectPrior(min_ar=0.35, max_ar=5.0),
        ),
        PartSpec(
            "tail",
            ["cat tail", "tail of the cat"],
            size_prior=SizePrior(min_area_ratio=0.001, max_area_ratio=0.12),
            aspect_prior=AspectPrior(min_ar=0.08, max_ar=8.0),
        ),
    ],
}


def clamp_box(box: Sequence[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = float(np.clip(x1, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter + 1e-6
    return inter / union


def aspect_ratio_ok(box: Sequence[float], prior: AspectPrior) -> bool:
    w = max(1e-6, box[2] - box[0])
    h = max(1e-6, box[3] - box[1])
    ar = w / h
    return prior.min_ar <= ar <= prior.max_ar


def size_penalty(area_ratio: float, prior: SizePrior) -> float:
    if area_ratio < prior.min_area_ratio:
        return (prior.min_area_ratio - area_ratio) * 10.0
    if area_ratio > prior.max_area_ratio:
        return (area_ratio - prior.max_area_ratio) * 10.0
    return 0.0


def expand_box(box: Sequence[float], w: int, h: int, expand_ratio: float) -> List[float]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    dx = bw * expand_ratio
    dy = bh * expand_ratio
    return clamp_box([x1 - dx, y1 - dy, x2 + dx, y2 + dy], w, h)


def nms_candidates(cands: List[CandidateBox], iou_thr: float) -> List[CandidateBox]:
    ordered = sorted(cands, key=lambda c: c.final_score, reverse=True)
    kept: List[CandidateBox] = []
    for c in ordered:
        if all(box_iou(c.box_xyxy, k.box_xyxy) < iou_thr for k in kept):
            kept.append(c)
    return kept


def ensure_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if device == "mps":
        if not hasattr(torch.backends, "mps"):
            return "cpu"
        if not torch.backends.mps.is_available():
            return "cpu"
    return device


class GroundingDinoEngine:
    def __init__(self, config_path: str, checkpoint_path: str, device: str) -> None:
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None
        self._load_image = None
        self._predict = None
        self._init_model()

    def _init_model(self) -> None:
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"GroundingDINO config not found: {self.config_path}")
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"GroundingDINO checkpoint not found: {self.checkpoint_path}")

        from groundingdino.util.inference import load_image, load_model, predict

        self._load_image = load_image
        self._predict = predict
        self._model = load_model(self.config_path, self.checkpoint_path, device=self.device)

    def predict(
        self,
        image_path: str,
        image_pil: Image.Image,
        phrase: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        _src, image = self._load_image(image_path)
        boxes, logits, _phrases = self._predict(
            model=self._model,
            image=image,
            caption=phrase,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        if boxes.size == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        w, h = image_pil.size
        boxes = boxes.astype(np.float32)
        if np.max(boxes) <= 1.5:
            cx = boxes[:, 0] * w
            cy = boxes[:, 1] * h
            bw = boxes[:, 2] * w
            bh = boxes[:, 3] * h
            boxes[:, 0] = cx - bw / 2
            boxes[:, 1] = cy - bh / 2
            boxes[:, 2] = cx + bw / 2
            boxes[:, 3] = cy + bh / 2
        return boxes, logits.astype(np.float32)


def detect_dog_or_cat(
    dino: GroundingDinoEngine,
    image_path: str,
    image_pil: Image.Image,
    box_threshold: float,
    text_threshold: float,
) -> Tuple[str, List[float], float]:
    best_cat = "dog"
    best_box = [0.0, 0.0, float(image_pil.size[0] - 1), float(image_pil.size[1] - 1)]
    best_score = -1.0
    for cat in ("dog", "cat"):
        boxes, scores = dino.predict(image_path, image_pil, cat, box_threshold, text_threshold)
        if boxes.shape[0] == 0:
            continue
        idx = int(np.argmax(scores))
        score = float(scores[idx])
        if score > best_score:
            best_score = score
            best_cat = cat
            best_box = clamp_box(boxes[idx].tolist(), image_pil.size[0], image_pil.size[1])
    return best_cat, best_box, max(0.0, best_score)


def run_part_box_pipeline(
    dino: GroundingDinoEngine,
    image_path: str,
    image_pil: Image.Image,
    category: str,
    object_bbox: List[float],
    box_threshold: float,
    text_threshold: float,
    topk: int,
    nms_iou: float,
    box_expand: float,
) -> Tuple[List[SelectedPart], List[Dict[str, str]]]:
    w, h = image_pil.size
    image_area = float(w * h)
    dropped: List[Dict[str, str]] = []
    selected_parts: List[SelectedPart] = []
    all_selected_cands: List[CandidateBox] = []

    specs_by_name = {sp.part_name: sp for sp in PART_TAXONOMY[category]}

    def collect_candidates(spec: PartSpec, phrases: List[str], bthr: float, tthr: float) -> List[CandidateBox]:
        cands: List[CandidateBox] = []
        for phrase in phrases[:3]:
            boxes, scores = dino.predict(image_path, image_pil, phrase, bthr, tthr)
            if boxes.shape[0] == 0:
                continue
            phrase_cands: List[CandidateBox] = []
            for i in range(boxes.shape[0]):
                b = clamp_box(boxes[i].tolist(), w, h)
                if not aspect_ratio_ok(b, spec.aspect_prior):
                    continue
                area_ratio = box_area(b) / image_area
                loc_penalty = max(0.0, 1.0 - box_iou(b, object_bbox))
                cx = (b[0] + b[2]) * 0.5
                cy = (b[1] + b[3]) * 0.5
                ocx = (object_bbox[0] + object_bbox[2]) * 0.5
                ocy = (object_bbox[1] + object_bbox[3]) * 0.5
                center_dist = abs(cx - ocx) / max(1.0, object_bbox[2] - object_bbox[0]) + abs(cy - ocy) / max(1.0, object_bbox[3] - object_bbox[1])
                final = float(scores[i]) - 0.4 * size_penalty(area_ratio, spec.size_prior) - 0.2 * loc_penalty - 0.08 * center_dist
                phrase_cands.append(
                    CandidateBox(
                        box_xyxy=b,
                        score=float(scores[i]),
                        phrase=phrase,
                        part_name=spec.part_name,
                        final_score=final,
                    )
                )
            phrase_cands.sort(key=lambda c: c.final_score, reverse=True)
            cands.extend(phrase_cands[:topk])
            if cands:
                break
        return cands

    preselected: Dict[str, CandidateBox] = {}

    # Bilateral helper for eyes/ears
    def bilateral_stage(base: str, left_name: str, right_name: str, stage_ratio: float) -> None:
        left_spec = specs_by_name.get(left_name)
        right_spec = specs_by_name.get(right_name)
        if not left_spec or not right_spec:
            return
        proxy = PartSpec(
            part_name=base,
            prompt_phrases=[f'{category} {base}', f'{base}s of the {category}', f'{category} head {base}'],
            expected_count=2,
            size_prior=left_spec.size_prior,
            aspect_prior=left_spec.aspect_prior,
        )
        candidates = collect_candidates(
            proxy,
            proxy.prompt_phrases,
            max(0.05, box_threshold * stage_ratio),
            max(0.05, text_threshold * stage_ratio),
        )
        candidates = nms_candidates(candidates, min(0.45, nms_iou))
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        if len(candidates) >= 2:
            pair = candidates[:2]
            pair.sort(key=lambda c: (c.box_xyxy[0] + c.box_xyxy[2]) * 0.5)
            pair[0].part_name = left_name
            pair[1].part_name = right_name
            preselected[left_name] = pair[0]
            preselected[right_name] = pair[1]
        elif len(candidates) == 1:
            only = candidates[0]
            cx = (only.box_xyxy[0] + only.box_xyxy[2]) * 0.5
            ocx = (object_bbox[0] + object_bbox[2]) * 0.5
            side = left_name if cx <= ocx else right_name
            only.part_name = side
            preselected[side] = only

    bilateral_stage('eye', 'left_eye', 'right_eye', 0.75)
    bilateral_stage('ear', 'left_ear', 'right_ear', 0.70)

    for pname, cand in preselected.items():
        all_selected_cands.append(cand)
        selected_parts.append(
            SelectedPart(
                part_name=pname,
                prompt_used=cand.phrase,
                box_xyxy=cand.box_xyxy,
                box_xyxy_expanded=expand_box(cand.box_xyxy, w, h, box_expand),
                box_score=cand.final_score,
            )
        )

    skip_names = {'left_eye', 'right_eye', 'left_ear', 'right_ear'}
    for spec in PART_TAXONOMY[category]:
        if spec.part_name in preselected or spec.part_name in skip_names:
            continue

        part_candidates = collect_candidates(spec, spec.prompt_phrases, box_threshold, text_threshold)
        if not part_candidates:
            dropped.append({"part_name": spec.part_name, "reason": "no_box_found"})
            continue

        kept_for_part = nms_candidates(part_candidates, nms_iou)
        if not kept_for_part:
            dropped.append({"part_name": spec.part_name, "reason": "nms_filtered"})
            continue
        best = kept_for_part[0]

        duplicate = any(box_iou(best.box_xyxy, prev.box_xyxy) >= nms_iou for prev in all_selected_cands)
        if duplicate:
            dropped.append({"part_name": spec.part_name, "reason": "global_nms_filtered"})
            continue

        all_selected_cands.append(best)
        selected_parts.append(
            SelectedPart(
                part_name=spec.part_name,
                prompt_used=best.phrase,
                box_xyxy=best.box_xyxy,
                box_xyxy_expanded=expand_box(best.box_xyxy, w, h, box_expand),
                box_score=best.final_score,
            )
        )

    # Leg stage: detect any visible legs, label leg_1..leg_N (N<=4) left->right.
    leg_proxy = PartSpec(
        part_name='leg',
        prompt_phrases=[f'{category} leg', f'legs of the {category}', f'{category} paw'],
        expected_count=4,
        size_prior=SizePrior(min_area_ratio=0.003, max_area_ratio=0.2),
        aspect_prior=AspectPrior(min_ar=0.08, max_ar=4.5),
    )
    leg_candidates = collect_candidates(
        leg_proxy,
        leg_proxy.prompt_phrases,
        max(0.05, box_threshold * 0.8),
        max(0.05, text_threshold * 0.8),
    )

    # Strict filtering: keep only confident legs in lower body region.
    obj_h = max(1.0, object_bbox[3] - object_bbox[1])
    lower_y = object_bbox[1] + 0.45 * obj_h
    strict_legs: List[CandidateBox] = []
    for cand in leg_candidates:
        cx = (cand.box_xyxy[0] + cand.box_xyxy[2]) * 0.5
        cy = (cand.box_xyxy[1] + cand.box_xyxy[3]) * 0.5
        if cy < lower_y:
            continue
        if cand.final_score < max(0.2, box_threshold * 0.9):
            continue
        if not (object_bbox[0] - 0.1 * (object_bbox[2] - object_bbox[0]) <= cx <= object_bbox[2] + 0.1 * (object_bbox[2] - object_bbox[0])):
            continue
        strict_legs.append(cand)

    strict_legs = nms_candidates(strict_legs, min(0.35, nms_iou))
    strict_legs.sort(key=lambda c: c.final_score, reverse=True)

    final_legs: List[CandidateBox] = []
    for cand in strict_legs:
        if any(box_iou(cand.box_xyxy, prev.box_xyxy) >= 0.35 for prev in final_legs):
            continue
        if any(box_iou(cand.box_xyxy, prev.box_xyxy) >= nms_iou for prev in all_selected_cands):
            continue
        final_legs.append(cand)
        if len(final_legs) >= 4:
            break

    final_legs.sort(key=lambda c: (c.box_xyxy[0] + c.box_xyxy[2]) * 0.5)
    for idx, cand in enumerate(final_legs, start=1):
        label = f'leg_{idx}'
        selected_parts.append(
            SelectedPart(
                part_name=label,
                prompt_used=cand.phrase,
                box_xyxy=cand.box_xyxy,
                box_xyxy_expanded=expand_box(cand.box_xyxy, w, h, box_expand),
                box_score=cand.final_score,
            )
        )
        all_selected_cands.append(cand)

    if not final_legs:
        dropped.append({"part_name": "leg", "reason": "no_confident_leg_found"})

    # Retry bilateral parts with relaxed thresholds.
    existing = {p.part_name for p in selected_parts}
    for miss in ('left_eye', 'right_eye', 'left_ear', 'right_ear'):
        if miss in specs_by_name and miss not in existing:
            spec = specs_by_name[miss]
            retry_phrases = spec.prompt_phrases + [
                f"{category} {'eye' if 'eye' in miss else 'ear'}"
            ]
            retry = collect_candidates(spec, retry_phrases, max(0.05, box_threshold * 0.65), max(0.05, text_threshold * 0.65))
            if retry:
                cand = nms_candidates(retry, nms_iou)[0]
                if all(box_iou(cand.box_xyxy, p.box_xyxy) < nms_iou for p in selected_parts):
                    selected_parts.append(
                        SelectedPart(
                            part_name=miss,
                            prompt_used=cand.phrase,
                            box_xyxy=cand.box_xyxy,
                            box_xyxy_expanded=expand_box(cand.box_xyxy, w, h, box_expand),
                            box_score=cand.final_score,
                        )
                    )
                    existing.add(miss)
                    continue
            dropped.append({"part_name": miss, "reason": "likely_not_visible_or_not_confident"})

    return selected_parts, dropped

def validate_inputs(args: argparse.Namespace) -> None:
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not Path(args.dino_config).exists():
        raise FileNotFoundError(f"GroundingDINO config not found: {args.dino_config}")
    if not Path(args.dino_checkpoint).exists():
        raise FileNotFoundError(f"GroundingDINO checkpoint not found: {args.dino_checkpoint}")


def save_labeled_boxes_image(image: Image.Image, parts: List[SelectedPart], output_path: Path) -> None:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    colors = [
        (255, 64, 64),
        (64, 200, 255),
        (255, 200, 64),
        (64, 255, 120),
        (220, 120, 255),
        (255, 120, 180),
    ]
    for idx, part in enumerate(parts):
        color = colors[idx % len(colors)]
        x1, y1, x2, y2 = part.box_xyxy_expanded
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = part.part_name
        tx = max(0, int(x1))
        ty = max(0, int(y1) - 18)
        draw.rectangle([tx, ty, tx + 8 * max(4, len(label)), ty + 16], fill=color)
        draw.text((tx + 3, ty + 2), label, fill=(0, 0, 0))
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dog/Cat part bounding box extraction (no segmentation).")
    parser.add_argument("--image", required=True, type=str, help="Input image path")
    parser.add_argument("--output_dir", default=".", type=str, help="Base output directory")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"], help="Inference device")

    parser.add_argument("--dino_config", required=True, type=str, help="GroundingDINO config path")
    parser.add_argument("--dino_checkpoint", required=True, type=str, help="GroundingDINO checkpoint path")

    parser.add_argument("--box_threshold", default=0.16, type=float)
    parser.add_argument("--text_threshold", default=0.16, type=float)
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--nms_iou", default=0.55, type=float)
    parser.add_argument("--box_expand", default=0.2, type=float, help="Expand each part box by ratio")

    args = parser.parse_args()
    args.device = ensure_device(args.device)
    validate_inputs(args)

    image_path = str(Path(args.image).resolve())
    image = Image.open(image_path).convert("RGB")

    output_dir = Path(args.output_dir) / Path(args.image).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    dino = GroundingDinoEngine(args.dino_config, args.dino_checkpoint, args.device)

    category, object_bbox, object_score = detect_dog_or_cat(
        dino=dino,
        image_path=image_path,
        image_pil=image,
        box_threshold=min(args.box_threshold, 0.2),
        text_threshold=min(args.text_threshold, 0.2),
    )

    selected_parts, dropped_parts = run_part_box_pipeline(
        dino=dino,
        image_path=image_path,
        image_pil=image,
        category=category,
        object_bbox=object_bbox,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        topk=args.topk,
        nms_iou=args.nms_iou,
        box_expand=args.box_expand,
    )

    results = {
        "detected_category": category,
        "primary_object": {
            "category": category,
            "confidence": object_score,
            "bbox_xyxy": object_bbox,
        },
        "parts": [asdict(p) for p in selected_parts],
        "dropped_parts": dropped_parts,
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for part in selected_parts:
        with open(output_dir / f"{part.part_name}_box.json", "w", encoding="utf-8") as f:
            json.dump(asdict(part), f, indent=2)

    labeled_boxes_path = output_dir / "boxes_labeled.png"
    save_labeled_boxes_image(image=image, parts=selected_parts, output_path=labeled_boxes_path)

    results["output_files"] = {
        "results_json": str(output_dir / "results.json"),
        "labeled_boxes_image": str(labeled_boxes_path),
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "parts_kept": len(selected_parts)}, indent=2))


if __name__ == "__main__":
    main()
