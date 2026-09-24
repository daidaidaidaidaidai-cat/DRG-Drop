# VisDrone Multimodal Detection Code

This is a minimal open-source version organized from the experimental directory. It keeps the model code needed to reproduce the paper's experiments, three target YAML configs, the training/validation/inference entry points, and the Depth Anything V2 inference code.

## Directory Structure

```text
.
├── configs/
│   ├── models/mm/visdrone_requested/   # The 3 model YAMLs used in the paper experiments
│   └── datasets/VisDrone.yaml          # Standard VisDrone dataset config
├── scripts/                            # Training, validation and inference entry points
├── ultralytics/                        # Source code based on Ultralytics 8.3.163 plus custom modules
├── ref/Depth-Anything-V2-main/         # Depth Anything V2 model and configuration scripts
├── requirements.txt
└── LICENSE
```

## Installation

Python 3.10/3.11 and a PyTorch build matching your CUDA version are recommended:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Optional: register the local package in editable mode
python -m pip install -e .
```

This repository already includes the local `ultralytics` source code. Do not install the same-named PyPI package, as it would overwrite it. Run all commands from the repository root.

## Weights

The repository does not include large weight files. Prepare them as needed:

- YOLO pre-trained weights: use Ultralytics' automatic download, or pass a local `.pt` path to `--weights`.
- Depth Anything V2 weights: download the corresponding `depth_anything_v2_{encoder}.pth` and place it under
  `ref/Depth-Anything-V2-main/checkpoints/`.

## The Three Paper YAMLs

```text
configs/models/mm/visdrone_requested/mm_rd_p2_gsconve_add.yaml
configs/models/mm/visdrone_requested/mm_rd_p2_gsconve_gate_nodiff.yaml
configs/models/mm/visdrone_requested/mm_rd_p2_gsconve_naive_concat.yaml
```

The three configs are two-branch RGB + X (auxiliary/depth) models that use custom modules such as `DConv`, `GSConvE`, `Add`, and `DepthConcatGateNoDiff`. The data loader must provide two input streams consistent with the project's multimodal input convention; the standard single-stream `VisDrone.yaml` is only used for the plain VisDrone detection baseline. The three model YAMLs currently set `nc: 80`; if you use the 10-class VisDrone2019 labels, update `nc` and the dataset class configuration accordingly.

## Training, Validation and Inference

```bash
python -m scripts.train --model configs/models/mm/visdrone_requested/mm_rd_p2_gsconve_add.yaml --data configs/datasets/VisDrone.yaml --epochs 100 --imgsz 640 --batch 16
python -m scripts.val --weights runs/train/exp/weights/best.pt --data configs/datasets/VisDrone.yaml
python -m scripts.detect --weights runs/train/exp/weights/best.pt --source path/to/images
```

If you use multimodal data, point `--data` to your own two-input data YAML and adjust `source` according to how your data is organized.

## Depth Anything V2

Basic single-image/directory inference:

```bash
python ref/Depth-Anything-V2-main/run.py --img-path path/to/images --encoder vitb --outdir depth_output
```

A configurable batch-processing script is available at `ref/Depth-Anything-V2-main/generate_depth.py`; see `example_configs.py` for an example config class. `ultralytics.nn.mm.generators.depth_anything_v2.DepthGen` automatically looks for the weights under `ref/Depth-Anything-V2-main/checkpoints/`.

## License and Third-Party Code

This project follows the Ultralytics AGPL-3.0 license from the original directory. Depth Anything V2 is retained under `ref/Depth-Anything-V2-main/` and uses its upstream license and copyright notice. Before submitting to a public repository, please add the paper authors, dataset sources, weight download URLs, and the full experimental configuration.
