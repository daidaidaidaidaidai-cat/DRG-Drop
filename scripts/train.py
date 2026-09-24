"""Train one of the paper's detector configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "configs/models/mm/visdrone_requested/mm_rd_p2_gsconve_add.yaml"
DEFAULT_DATA = ROOT / "configs/datasets/VisDrone.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="model YAML or .pt path")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="dataset YAML path")
    parser.add_argument("--weights", default=None, help="optional pretrained .pt checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="exp")
    args = parser.parse_args()

    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
