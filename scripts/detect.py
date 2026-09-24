"""Run detector inference on an image, directory, video, or stream."""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="trained .pt checkpoint")
    parser.add_argument("--source", required=True, help="image, directory, video, or stream")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--save-txt", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=args.save_txt,
    )


if __name__ == "__main__":
    main()
