from __future__ import annotations

import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GroundingDINO pet part bounding boxes on an entire dataset."
    )

    # Required
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--dino_config", required=True, type=str)
    parser.add_argument("--dino_checkpoint", required=True, type=str)

    # Optional runtime params (mandatory defaults)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--box_threshold", default=0.16, type=float)
    parser.add_argument("--text_threshold", default=0.16, type=float)
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--nms_iou", default=0.55, type=float)
    parser.add_argument("--box_expand", default=0.2, type=float)

    # Dataset execution control
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--exts", nargs="+", default=["jpg", "jpeg", "png"])

    # Output control
    parser.add_argument(
        "--write_mode",
        choices=["json_only", "overlay_only", "both"],
        default="both",
    )

    return parser.parse_args()
