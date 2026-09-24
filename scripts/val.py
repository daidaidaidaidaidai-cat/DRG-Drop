"""Validate a trained detector and print the standard Ultralytics metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "configs/datasets/VisDrone.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="trained .pt checkpoint")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="dataset YAML path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="exp")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
