#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Anything V2 - Enhanced Configurable Depth Generation Script
增强版可配置的深度图生成脚本 (含预处理和后处理功能)

Usage:
    python generate_depth.py
    python generate_depth.py --config custom_config.py
    
修改下面的配置参数，然后运行脚本即可生成深度图
"""

import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from tqdm import tqdm
import time
import warnings
from scipy import ndimage
from depth_anything_v2.dpt import DepthAnythingV2
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

# ================================
# 配置参数 - 在这里修改你的设置
# ================================

class Config:
    """配置类 - 在这里修改所有参数"""
    
    # === 输入输出路径配置 ===
    INPUT_PATH = "assets/examples"  # 输入图像路径（可以是文件夹、单个文件或txt文件列表）
    OUTPUT_DIR = "./depth_output"   # 输出文件夹
    
    # === 模型配置 ===
    ENCODER = "vitb"  # 模型选择: 'vits'(最快), 'vitb'(平衡), 'vitl'(最精确)
    INPUT_SIZE = 518  # 输入尺寸，越大精度越高但速度越慢 (推荐: 256-1024)
    
    # === 输出格式配置 ===
    SAVE_ORIGINAL_COMPARISON = True   # 是否保存原图+深度图的对比图
    SAVE_DEPTH_ONLY = True           # 是否只保存深度图
    SAVE_GRAYSCALE = False           # 是否保存灰度深度图
    SAVE_16BIT_RAW = True            # 是否保存16位原始深度数据
    SAVE_NUMPY = False               # 是否保存numpy格式的深度数据
    
    # === 处理配置 ===
    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']  # 支持的图像格式
    RECURSIVE_SEARCH = True          # 是否递归搜索子文件夹
    SHOW_PROGRESS = True             # 是否显示进度条
    PRINT_STATS = True               # 是否打印统计信息
    
    # === 深度图可视化配置 ===
    COLORMAP = 'Spectral_r'          # 颜色映射: 'Spectral_r', 'plasma', 'viridis', 'jet'
    DEPTH_RANGE_PERCENTILE = (2, 98) # 深度值范围百分位数，用于更好的可视化
    
    # === 高级配置 ===
    BATCH_SIZE = 1                   # 批处理大小（当前版本设为1）
    DEVICE = 'auto'                  # 设备选择: 'auto', 'cuda', 'cpu'
    
    # === 文件命名配置 ===
    DEPTH_SUFFIX = "_depth"          # 深度图文件后缀
    RAW_SUFFIX = "_depth_raw"        # 原始深度数据后缀
    COMPARISON_SUFFIX = "_comparison" # 对比图后缀

    # ===================================
    # 新增：图像预处理高级配置
    # ===================================
    
    # === 预处理开关 ===
    ENABLE_ADVANCED_PREPROCESSING = False  # 启用高级预处理
    
    # === 图像增强 ===
    APPLY_CLAHE = False              # 应用CLAHE对比度增强
    CLAHE_CLIP_LIMIT = 2.0           # CLAHE裁剪限制
    CLAHE_GRID_SIZE = (8, 8)         # CLAHE网格大小
    
    GAMMA_CORRECTION = 1.0           # Gamma校正值 (1.0=不校正, <1暗, >1亮)
    BRIGHTNESS_ADJUSTMENT = 0        # 亮度调整 (-100到100)
    CONTRAST_ADJUSTMENT = 1.0        # 对比度调整 (1.0=不调整)
    
    # === 图像去噪 ===
    APPLY_DENOISING = False          # 应用去噪
    DENOISE_H = 10                   # 降噪强度
    DENOISE_TEMPLATE_WINDOW = 7      # 模板窗口大小
    DENOISE_SEARCH_WINDOW = 21       # 搜索窗口大小
    
    # === 图像锐化 ===
    APPLY_SHARPENING = False         # 应用锐化
    SHARPEN_STRENGTH = 1.0           # 锐化强度
    
    # === 自定义变换参数 ===
    KEEP_ASPECT_RATIO = True         # 保持宽高比
    ENSURE_MULTIPLE_OF = 14          # 确保尺寸是14的倍数
    RESIZE_METHOD = 'lower_bound'    # 'lower_bound', 'upper_bound', 'minimal'
    IMAGE_INTERPOLATION = cv2.INTER_CUBIC  # 图像插值方法
    
    # === 自定义归一化 ===
    CUSTOM_NORMALIZATION = False     # 使用自定义归一化
    CUSTOM_MEAN = [0.485, 0.456, 0.406]  # 自定义均值
    CUSTOM_STD = [0.229, 0.224, 0.225]   # 自定义标准差

    # ===================================
    # 新增：深度后处理高级配置
    # ===================================
    
    # === 后处理开关 ===
    ENABLE_DEPTH_POSTPROCESSING = False  # 启用深度后处理
    
    # === 深度平滑滤波 ===
    APPLY_GAUSSIAN_SMOOTHING = False  # 应用高斯平滑
    GAUSSIAN_KERNEL_SIZE = 5         # 高斯核大小 (奇数)
    GAUSSIAN_SIGMA = 1.0             # 高斯标准差
    
    APPLY_MEDIAN_FILTER = False      # 应用中值滤波
    MEDIAN_KERNEL_SIZE = 5           # 中值滤波核大小
    
    APPLY_BILATERAL_FILTER = False   # 应用双边滤波
    BILATERAL_D = 9                  # 双边滤波直径
    BILATERAL_SIGMA_COLOR = 75       # 颜色空间标准差
    BILATERAL_SIGMA_SPACE = 75       # 坐标空间标准差
    
    # === 深度图修复 ===
    APPLY_INPAINTING = False         # 应用深度修复
    INPAINT_RADIUS = 3               # 修复半径
    
    # === 边缘保护平滑 ===
    APPLY_EDGE_PRESERVING = False    # 应用边缘保护滤波
    EDGE_PRESERVING_FLAGS = 1        # 边缘保护模式 (1或2)
    EDGE_PRESERVING_SIGMA_S = 50     # 邻域大小
    EDGE_PRESERVING_SIGMA_R = 0.4    # 边缘相似性
    
    # === 深度范围调整 ===
    APPLY_DEPTH_SCALING = False      # 应用深度缩放
    DEPTH_SCALE_FACTOR = 1.0         # 深度缩放因子
    DEPTH_OFFSET = 0.0               # 深度偏移
    INVERT_DEPTH = False             # 反转深度值
    
    # === 深度图形态学操作 ===
    APPLY_MORPHOLOGICAL_OPS = False  # 应用形态学操作
    MORPH_OPERATION = 'opening'      # 'opening', 'closing', 'gradient'
    MORPH_KERNEL_SIZE = 3            # 形态学核大小
    MORPH_ITERATIONS = 1             # 迭代次数
    
    # === 深度统计信息 ===
    SAVE_DEPTH_STATISTICS = False    # 保存深度统计信息


def apply_image_preprocessing(image, config):
    """应用高级图像预处理"""
    if not config.ENABLE_ADVANCED_PREPROCESSING:
        return image
    
    processed_image = image.copy()
    
    # 1. CLAHE对比度增强
    if config.APPLY_CLAHE:
        lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, 
                               tileGridSize=config.CLAHE_GRID_SIZE)
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        processed_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 2. Gamma校正
    if config.GAMMA_CORRECTION != 1.0:
        inv_gamma = 1.0 / config.GAMMA_CORRECTION
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        processed_image = cv2.LUT(processed_image, table)
    
    # 3. 亮度和对比度调整
    if config.BRIGHTNESS_ADJUSTMENT != 0 or config.CONTRAST_ADJUSTMENT != 1.0:
        processed_image = cv2.convertScaleAbs(processed_image, 
                                            alpha=config.CONTRAST_ADJUSTMENT, 
                                            beta=config.BRIGHTNESS_ADJUSTMENT)
    
    # 4. 图像去噪
    if config.APPLY_DENOISING:
        processed_image = cv2.fastNlMeansDenoisingColored(
            processed_image, None, 
            h=config.DENOISE_H,
            hColor=config.DENOISE_H,
            templateWindowSize=config.DENOISE_TEMPLATE_WINDOW,
            searchWindowSize=config.DENOISE_SEARCH_WINDOW
        )
    
    # 5. 图像锐化
    if config.APPLY_SHARPENING:
        kernel = np.array([[-1,-1,-1], 
                          [-1, 9,-1], 
                          [-1,-1,-1]]) * config.SHARPEN_STRENGTH
        processed_image = cv2.filter2D(processed_image, -1, kernel)
        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    
    return processed_image


def create_custom_transform(config):
    """创建自定义图像变换管道"""
    transforms = []
    
    # 添加Resize变换
    transforms.append(Resize(
        width=config.INPUT_SIZE,
        height=config.INPUT_SIZE,
        resize_target=False,
        keep_aspect_ratio=config.KEEP_ASPECT_RATIO,
        ensure_multiple_of=config.ENSURE_MULTIPLE_OF,
        resize_method=config.RESIZE_METHOD,
        image_interpolation_method=config.IMAGE_INTERPOLATION,
    ))
    
    # 添加归一化变换
    if config.CUSTOM_NORMALIZATION:
        transforms.append(NormalizeImage(mean=config.CUSTOM_MEAN, std=config.CUSTOM_STD))
    else:
        transforms.append(NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    # 添加网络准备变换
    transforms.append(PrepareForNet())
    
    return Compose(transforms)


def apply_depth_postprocessing(depth, config):
    """应用深度后处理"""
    if not config.ENABLE_DEPTH_POSTPROCESSING:
        return depth
    
    processed_depth = depth.copy()
    
    # 1. 高斯平滑
    if config.APPLY_GAUSSIAN_SMOOTHING:
        processed_depth = cv2.GaussianBlur(
            processed_depth, 
            (config.GAUSSIAN_KERNEL_SIZE, config.GAUSSIAN_KERNEL_SIZE), 
            config.GAUSSIAN_SIGMA
        )
    
    # 2. 中值滤波
    if config.APPLY_MEDIAN_FILTER:
        processed_depth = cv2.medianBlur(
            processed_depth.astype(np.float32), 
            config.MEDIAN_KERNEL_SIZE
        )
    
    # 3. 双边滤波
    if config.APPLY_BILATERAL_FILTER:
        # 转换为8位进行双边滤波
        depth_8bit = ((processed_depth - processed_depth.min()) / 
                     (processed_depth.max() - processed_depth.min()) * 255).astype(np.uint8)
        filtered_8bit = cv2.bilateralFilter(
            depth_8bit, 
            config.BILATERAL_D, 
            config.BILATERAL_SIGMA_COLOR, 
            config.BILATERAL_SIGMA_SPACE
        )
        # 转换回原始范围
        processed_depth = (filtered_8bit.astype(np.float32) / 255.0 * 
                          (processed_depth.max() - processed_depth.min()) + 
                          processed_depth.min())
    
    # 4. 深度图修复（对于有缺失值的情况）
    if config.APPLY_INPAINTING:
        # 创建掩码（假设0值为缺失值）
        mask = (processed_depth == 0).astype(np.uint8)
        if np.any(mask):
            depth_8bit = ((processed_depth - processed_depth.min()) / 
                         (processed_depth.max() - processed_depth.min()) * 255).astype(np.uint8)
            inpainted = cv2.inpaint(depth_8bit, mask, config.INPAINT_RADIUS, cv2.INPAINT_TELEA)
            processed_depth = (inpainted.astype(np.float32) / 255.0 * 
                              (processed_depth.max() - processed_depth.min()) + 
                              processed_depth.min())
    
    # 5. 边缘保护滤波
    if config.APPLY_EDGE_PRESERVING:
        depth_8bit = ((processed_depth - processed_depth.min()) / 
                     (processed_depth.max() - processed_depth.min()) * 255).astype(np.uint8)
        filtered_8bit = cv2.edgePreservingFilter(
            depth_8bit, 
            flags=config.EDGE_PRESERVING_FLAGS,
            sigma_s=config.EDGE_PRESERVING_SIGMA_S,
            sigma_r=config.EDGE_PRESERVING_SIGMA_R
        )
        processed_depth = (filtered_8bit.astype(np.float32) / 255.0 * 
                          (processed_depth.max() - processed_depth.min()) + 
                          processed_depth.min())
    
    # 6. 深度缩放和偏移
    if config.APPLY_DEPTH_SCALING:
        processed_depth = processed_depth * config.DEPTH_SCALE_FACTOR + config.DEPTH_OFFSET
    
    # 7. 深度反转
    if config.INVERT_DEPTH:
        processed_depth = processed_depth.max() - processed_depth
    
    # 8. 形态学操作
    if config.APPLY_MORPHOLOGICAL_OPS:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE))
        depth_8bit = ((processed_depth - processed_depth.min()) / 
                     (processed_depth.max() - processed_depth.min()) * 255).astype(np.uint8)
        
        if config.MORPH_OPERATION == 'opening':
            morphed = cv2.morphologyEx(depth_8bit, cv2.MORPH_OPEN, kernel, 
                                     iterations=config.MORPH_ITERATIONS)
        elif config.MORPH_OPERATION == 'closing':
            morphed = cv2.morphologyEx(depth_8bit, cv2.MORPH_CLOSE, kernel, 
                                     iterations=config.MORPH_ITERATIONS)
        elif config.MORPH_OPERATION == 'gradient':
            morphed = cv2.morphologyEx(depth_8bit, cv2.MORPH_GRADIENT, kernel, 
                                     iterations=config.MORPH_ITERATIONS)
        else:
            morphed = depth_8bit
        
        processed_depth = (morphed.astype(np.float32) / 255.0 * 
                          (processed_depth.max() - processed_depth.min()) + 
                          processed_depth.min())
    
    return processed_depth


def calculate_depth_statistics(depth):
    """计算深度统计信息"""
    stats = {
        'min': float(np.min(depth)),
        'max': float(np.max(depth)),
        'mean': float(np.mean(depth)),
        'std': float(np.std(depth)),
        'median': float(np.median(depth)),
        'p25': float(np.percentile(depth, 25)),
        'p75': float(np.percentile(depth, 75)),
        'p95': float(np.percentile(depth, 95)),
        'p99': float(np.percentile(depth, 99)),
    }
    return stats


def load_model(config):
    """加载深度估计模型"""
    # 设备选择
    if config.DEVICE == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    else:
        device = config.DEVICE
    
    print(f"Using device: {device}")
    
    # 模型配置
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    if config.ENCODER not in model_configs:
        raise ValueError(f"Unsupported encoder: {config.ENCODER}")
    
    # 加载模型
    model = DepthAnythingV2(**model_configs[config.ENCODER])
    checkpoint_path = f'checkpoints/depth_anything_v2_{config.ENCODER}.pth'
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model = model.to(device).eval()
    
    print(f"Model loaded: {config.ENCODER} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters)")
    
    return model, device


def get_image_files(config):
    """获取需要处理的图像文件列表"""
    if os.path.isfile(config.INPUT_PATH):
        if config.INPUT_PATH.endswith('.txt'):
            # 从文本文件读取图像路径列表
            with open(config.INPUT_PATH, 'r') as f:
                filenames = [line.strip() for line in f.readlines() if line.strip()]
        else:
            # 单个图像文件
            filenames = [config.INPUT_PATH]
    else:
        # 文件夹
        filenames = []
        if config.RECURSIVE_SEARCH:
            pattern = os.path.join(config.INPUT_PATH, '**', '*')
            all_files = glob.glob(pattern, recursive=True)
        else:
            pattern = os.path.join(config.INPUT_PATH, '*')
            all_files = glob.glob(pattern)
        
        # 过滤支持的图像格式
        for file_path in all_files:
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file_path.lower())
                if ext in [e.lower() for e in config.SUPPORTED_EXTENSIONS]:
                    filenames.append(file_path)
    
    return sorted(filenames)


def process_depth(raw_depth, config):
    """处理原始深度数据"""
    # 使用百分位数来改善可视化效果
    if config.DEPTH_RANGE_PERCENTILE:
        p_low, p_high = config.DEPTH_RANGE_PERCENTILE
        depth_min = np.percentile(raw_depth, p_low)
        depth_max = np.percentile(raw_depth, p_high)
        raw_depth_clipped = np.clip(raw_depth, depth_min, depth_max)
    else:
        raw_depth_clipped = raw_depth
        depth_min, depth_max = raw_depth.min(), raw_depth.max()
    
    # 归一化到0-255
    if depth_max > depth_min:
        depth_normalized = (raw_depth_clipped - depth_min) / (depth_max - depth_min) * 255.0
    else:
        depth_normalized = np.zeros_like(raw_depth_clipped)
    
    depth_uint8 = depth_normalized.astype(np.uint8)
    
    # 16位深度
    depth_16bit = ((raw_depth_clipped - depth_min) / (depth_max - depth_min) * 65535).astype(np.uint16)
    
    return depth_uint8, depth_16bit, (depth_min, depth_max)


def save_outputs(image, raw_depth, base_name, output_dir, config):
    """保存各种格式的输出"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理深度数据
    depth_uint8, depth_16bit, depth_range = process_depth(raw_depth, config)
    
    saved_files = []
    depth_stats = None
    
    # 计算深度统计信息
    if config.SAVE_DEPTH_STATISTICS:
        depth_stats = calculate_depth_statistics(raw_depth)
        stats_path = os.path.join(output_dir, f"{base_name}_depth_stats.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Depth Statistics for {base_name}\n")
            f.write("=" * 40 + "\n")
            for key, value in depth_stats.items():
                f.write(f"{key}: {value:.4f}\n")
        saved_files.append(stats_path)
    
    # 1. 保存彩色深度图
    if config.SAVE_DEPTH_ONLY:
        if config.SAVE_GRAYSCALE:
            depth_colored = np.repeat(depth_uint8[..., np.newaxis], 3, axis=-1)
        else:
            cmap = matplotlib.colormaps.get_cmap(config.COLORMAP)
            depth_colored = (cmap(depth_uint8)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        depth_path = os.path.join(output_dir, f"{base_name}{config.DEPTH_SUFFIX}.png")
        cv2.imwrite(depth_path, depth_colored)
        saved_files.append(depth_path)
    
    # 2. 保存16位原始深度数据
    if config.SAVE_16BIT_RAW:
        raw_path = os.path.join(output_dir, f"{base_name}{config.RAW_SUFFIX}.png")
        cv2.imwrite(raw_path, depth_16bit)
        saved_files.append(raw_path)
    
    # 3. 保存numpy格式
    if config.SAVE_NUMPY:
        numpy_path = os.path.join(output_dir, f"{base_name}{config.DEPTH_SUFFIX}.npy")
        np.save(numpy_path, raw_depth)
        saved_files.append(numpy_path)
    
    # 4. 保存对比图
    if config.SAVE_ORIGINAL_COMPARISON:
        if config.SAVE_GRAYSCALE:
            depth_colored = np.repeat(depth_uint8[..., np.newaxis], 3, axis=-1)
        else:
            cmap = matplotlib.colormaps.get_cmap(config.COLORMAP)
            depth_colored = (cmap(depth_uint8)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # 确保两个图像高度一致
        h1, h2 = image.shape[0], depth_colored.shape[0]
        if h1 != h2:
            if h1 > h2:
                depth_colored = cv2.resize(depth_colored, (depth_colored.shape[1], h1))
            else:
                image = cv2.resize(image, (image.shape[1], h2))
        
        # 创建分隔线
        split_region = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([image, split_region, depth_colored])
        
        comparison_path = os.path.join(output_dir, f"{base_name}{config.COMPARISON_SUFFIX}.png")
        cv2.imwrite(comparison_path, combined_result)
        saved_files.append(comparison_path)
    
    return saved_files, depth_range, depth_stats


def infer_image_with_custom_preprocessing(model, raw_image, config):
    """使用自定义预处理进行深度推理"""
    # 1. 应用图像预处理
    preprocessed_image = apply_image_preprocessing(raw_image, config)
    
    # 2. 创建自定义变换
    transform = create_custom_transform(config)
    
    # 3. 准备输入
    h, w = preprocessed_image.shape[:2]
    image_rgb = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB) / 255.0
    image_tensor = transform({'image': image_rgb})['image']
    image_tensor = torch.from_numpy(image_tensor).unsqueeze(0)
    
    # 4. 设备转换
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # 5. 模型推理
    with torch.no_grad():
        depth = model(image_tensor)
        depth = torch.nn.functional.interpolate(
            depth[:, None], (h, w), mode="bilinear", align_corners=True
        )[0, 0]
    
    # 6. 转换为numpy并应用后处理
    depth_np = depth.cpu().numpy()
    depth_processed = apply_depth_postprocessing(depth_np, config)
    
    return depth_processed


def print_processing_config(config):
    """打印处理配置信息"""
    print("\n" + "=" * 60)
    print("Processing Configuration")
    print("=" * 60)
    
    if config.ENABLE_ADVANCED_PREPROCESSING:
        print("Image Preprocessing: ENABLED")
        if config.APPLY_CLAHE:
            print(f"  - CLAHE: Clip={config.CLAHE_CLIP_LIMIT}, Grid={config.CLAHE_GRID_SIZE}")
        if config.GAMMA_CORRECTION != 1.0:
            print(f"  - Gamma Correction: {config.GAMMA_CORRECTION}")
        if config.APPLY_DENOISING:
            print(f"  - Denoising: H={config.DENOISE_H}")
        if config.APPLY_SHARPENING:
            print(f"  - Sharpening: Strength={config.SHARPEN_STRENGTH}")
    else:
        print("Image Preprocessing: DISABLED")
    
    if config.ENABLE_DEPTH_POSTPROCESSING:
        print("Depth Postprocessing: ENABLED")
        if config.APPLY_GAUSSIAN_SMOOTHING:
            print(f"  - Gaussian Smoothing: Kernel={config.GAUSSIAN_KERNEL_SIZE}, Sigma={config.GAUSSIAN_SIGMA}")
        if config.APPLY_MEDIAN_FILTER:
            print(f"  - Median Filter: Kernel={config.MEDIAN_KERNEL_SIZE}")
        if config.APPLY_BILATERAL_FILTER:
            print(f"  - Bilateral Filter: D={config.BILATERAL_D}")
        if config.APPLY_EDGE_PRESERVING:
            print(f"  - Edge Preserving Filter: Enabled")
    else:
        print("Depth Postprocessing: DISABLED")
    
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Depth Anything V2 - Enhanced Configurable Depth Generation')
    parser.add_argument('--config', type=str, help='Custom config file (optional)')
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        print(f"Using config file: {args.config}")
        # TODO: 实现配置文件加载
    
    config = Config()
    
    print("=" * 60)
    print("Depth Anything V2 - Enhanced Depth Generation Starting")
    print("=" * 60)
    print(f"Input Path: {config.INPUT_PATH}")
    print(f"Output Directory: {config.OUTPUT_DIR}")
    print(f"Model: {config.ENCODER}")
    print(f"Input Size: {config.INPUT_SIZE}")
    
    # 打印处理配置
    print_processing_config(config)
    
    # 加载模型
    try:
        model, device = load_model(config)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 获取图像文件列表
    image_files = get_image_files(config)
    
    if not image_files:
        print(f"No image files found in: {config.INPUT_PATH}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # 处理图像
    start_time = time.time()
    processed_count = 0
    error_count = 0
    total_depth_range = []
    all_depth_stats = []
    
    iterator = tqdm(image_files, desc="Processing") if config.SHOW_PROGRESS else image_files
    
    for image_path in iterator:
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                error_count += 1
                continue
            
            # 生成深度图（使用增强的处理流程）
            if config.ENABLE_ADVANCED_PREPROCESSING or config.ENABLE_DEPTH_POSTPROCESSING:
                depth = infer_image_with_custom_preprocessing(model, image, config)
            else:
                depth = model.infer_image(image, config.INPUT_SIZE)
            
            # 保存结果
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            saved_files, depth_range, depth_stats = save_outputs(image, depth, base_name, config.OUTPUT_DIR, config)
            
            total_depth_range.append(depth_range)
            if depth_stats:
                all_depth_stats.append(depth_stats)
            processed_count += 1
            
            if config.SHOW_PROGRESS and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    'success': processed_count, 
                    'errors': error_count,
                    'depth_range': f"{depth_range[0]:.2f}-{depth_range[1]:.2f}"
                })
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            error_count += 1
    
    # 处理完成统计
    end_time = time.time()
    processing_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total images: {len(image_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Average time per image: {processing_time/max(processed_count, 1):.2f} seconds")
    print(f"Output directory: {os.path.abspath(config.OUTPUT_DIR)}")
    
    if config.PRINT_STATS and total_depth_range:
        min_depths = [r[0] for r in total_depth_range]
        max_depths = [r[1] for r in total_depth_range]
        print(f"Depth range statistics:")
        print(f"  Min depth: {min(min_depths):.2f} - {max(min_depths):.2f}")
        print(f"  Max depth: {min(max_depths):.2f} - {max(max_depths):.2f}")
    
    # 保存批量统计信息
    if config.SAVE_DEPTH_STATISTICS and all_depth_stats:
        batch_stats_path = os.path.join(config.OUTPUT_DIR, "batch_depth_statistics.txt")
        with open(batch_stats_path, 'w') as f:
            f.write("Batch Depth Statistics Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total processed images: {len(all_depth_stats)}\n\n")
            
            # 计算整体统计
            for key in all_depth_stats[0].keys():
                values = [stats[key] for stats in all_depth_stats]
                f.write(f"{key.upper()}:\n")
                f.write(f"  Mean: {np.mean(values):.4f}\n")
                f.write(f"  Std:  {np.std(values):.4f}\n")
                f.write(f"  Min:  {np.min(values):.4f}\n")
                f.write(f"  Max:  {np.max(values):.4f}\n\n")
    
    print("=" * 60)


if __name__ == '__main__':
    main() 