# YOLO-PRCV: VisDrone 多模态检测代码

这是从实验目录中整理出的最小开源版本，保留论文复现实验所需的模型代码、三份目标 YAML、训练/验证/推理入口，以及 Depth Anything V2 的推理代码。

## 目录结构

```text
.
├── configs/
│   ├── models/mm/visdrone_requested/   # 论文实验使用的 3 个模型 YAML
│   └── datasets/VisDrone.yaml          # 标准 VisDrone 数据配置
├── scripts/                            # 训练、验证、推理入口
├── ultralytics/                        # 基于 Ultralytics 8.3.163 的源码及自定义模块
├── ref/Depth-Anything-V2-main/         # Depth Anything V2 模型与配置脚本
├── requirements.txt
└── LICENSE
```

## 安装

建议使用 Python 3.10/3.11 和与 CUDA 匹配的 PyTorch：

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# 可选：以 editable 方式注册本地包
python -m pip install -e .
```

本仓库已经包含本地 `ultralytics` 源码，不要再安装同名的 PyPI 包覆盖它。请在仓库根目录执行命令。

## 权重

仓库不包含大体积权重文件。请按需准备：

- YOLO 预训练权重：使用 Ultralytics 的自动下载，或将本地 `.pt` 路径传给 `--weights`。
- Depth Anything V2 权重：下载对应的 `depth_anything_v2_{encoder}.pth`，放到
  `ref/Depth-Anything-V2-main/checkpoints/`。

## 三份论文 YAML

```text
configs/models/mm/visdrone_requested/mm_rd_p2_gsconve_add.yaml
configs/models/mm/visdrone_requested/mm_rd_p2_gsconve_gate_nodiff.yaml
configs/models/mm/visdrone_requested/mm_rd_p2_gsconve_naive_concat.yaml
```

三份配置是 RGB + X（辅助/深度）双分支模型，并使用 `DConv`、`GSConvE`、`Add` 或 `DepthConcatGateNoDiff` 等自定义模块。数据加载器需要提供与项目多模态输入约定一致的两路输入；标准单路 `VisDrone.yaml` 仅用于普通 VisDrone 检测基线。三份模型 YAML 当前写的是 `nc: 80`，如果使用 VisDrone2019 的 10 类标签，请同步修改 `nc` 和数据类别配置。

## 训练、验证和推理

```bash
python -m scripts.train --model configs/models/mm/visdrone_requested/mm_rd_p2_gsconve_add.yaml --data configs/datasets/VisDrone.yaml --epochs 100 --imgsz 640 --batch 16
python -m scripts.val --weights runs/train/exp/weights/best.pt --data configs/datasets/VisDrone.yaml
python -m scripts.detect --weights runs/train/exp/weights/best.pt --source path/to/images
```

如果使用多模态数据，请将 `--data` 指向你自己的双输入数据 YAML，并按数据组织方式调整 `source`。

## Depth Anything V2

基础单图/目录推理：

```bash
python ref/Depth-Anything-V2-main/run.py --img-path path/to/images --encoder vitb --outdir depth_output
```

可配置批处理脚本为 `ref/Depth-Anything-V2-main/generate_depth.py`；示例配置类见 `example_configs.py`。`ultralytics.nn.mm.generators.depth_anything_v2.DepthGen` 会从 `ref/Depth-Anything-V2-main/checkpoints/` 自动查找权重。

## 许可证和第三方代码

本项目沿用原目录中的 Ultralytics AGPL-3.0 许可证。Depth Anything V2 保留在 `ref/Depth-Anything-V2-main/`，使用其上游许可证和版权声明。提交到公开仓库前，请补充论文作者、数据集来源、权重下载地址和完整实验配置。
