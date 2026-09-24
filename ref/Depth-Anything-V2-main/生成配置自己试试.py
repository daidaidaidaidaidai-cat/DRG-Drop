# -*- coding: utf-8 -*-
"""

使用方法：
1. 复制所需的配置类到 generate_depth.py 中替换原有的 Config 类
2. 运行 python generate_depth.py
"""

import cv2

# =====================================
# 配置示例 1: 低光照图像增强
# Low-light Image Enhancement
# =====================================
class LowLightEnhancementConfig:
    """低光照图像增强配置 - 适用于暗光环境拍摄的图像"""
    
    # 基本设置
    INPUT_PATH = "assets/examples"
    OUTPUT_DIR = "./depth_output_lowlight"
    ENCODER = "vitb"
    INPUT_SIZE = 518
    
    # 输出设置
    SAVE_ORIGINAL_COMPARISON = True
    SAVE_DEPTH_ONLY = True
    SAVE_16BIT_RAW = True
    SAVE_DEPTH_STATISTICS = True
    
    # === 图像预处理 - 针对低光照优化 ===
    ENABLE_ADVANCED_PREPROCESSING = True
    
    # CLAHE增强对比度
    APPLY_CLAHE = True
    CLAHE_CLIP_LIMIT = 3.0        # 较高的裁剪限制提升对比度
    CLAHE_GRID_SIZE = (8, 8)
    
    # Gamma校正提亮图像
    GAMMA_CORRECTION = 0.7        # <1 提亮图像
    BRIGHTNESS_ADJUSTMENT = 20    # 增加亮度
    CONTRAST_ADJUSTMENT = 1.2     # 增加对比度
    
    # 降噪处理
    APPLY_DENOISING = True
    DENOISE_H = 8                # 中等降噪强度
    
    # === 深度后处理 - 平滑噪声 ===
    ENABLE_DEPTH_POSTPROCESSING = True
    
    # 高斯平滑去除噪声
    APPLY_GAUSSIAN_SMOOTHING = True
    GAUSSIAN_KERNEL_SIZE = 3
    GAUSSIAN_SIGMA = 0.8
    
    # 双边滤波保持边缘
    APPLY_BILATERAL_FILTER = True
    BILATERAL_D = 5
    BILATERAL_SIGMA_COLOR = 50
    BILATERAL_SIGMA_SPACE = 50


# =====================================
# 配置示例 2: 高精度细节保护
# High-precision Detail Preservation
# =====================================
class HighPrecisionConfig:
    """高精度细节保护配置 - 适用于需要保留精细细节的场景"""
    
    # 基本设置
    INPUT_PATH = "assets/examples"
    OUTPUT_DIR = "./depth_output_precision"
    ENCODER = "vitl"             # 使用大模型提高精度
    INPUT_SIZE = 1024            # 更高的输入分辨率
    
    # 输出设置
    SAVE_ORIGINAL_COMPARISON = True
    SAVE_DEPTH_ONLY = True
    SAVE_16BIT_RAW = True
    SAVE_NUMPY = True            # 保存原始数据用于后续分析
    SAVE_DEPTH_STATISTICS = True
    
    # === 图像预处理 - 细节增强 ===
    ENABLE_ADVANCED_PREPROCESSING = True
    
    # 锐化增强细节
    APPLY_SHARPENING = True
    SHARPEN_STRENGTH = 0.5       # 适度锐化避免过度
    
    # 高质量插值
    IMAGE_INTERPOLATION = cv2.INTER_LANCZOS4
    
    # === 深度后处理 - 边缘保护 ===
    ENABLE_DEPTH_POSTPROCESSING = True
    
    # 边缘保护滤波
    APPLY_EDGE_PRESERVING = True
    EDGE_PRESERVING_FLAGS = 2    # 使用标准化卷积
    EDGE_PRESERVING_SIGMA_S = 30
    EDGE_PRESERVING_SIGMA_R = 0.2
    
    # 轻微的中值滤波去除离群点
    APPLY_MEDIAN_FILTER = True
    MEDIAN_KERNEL_SIZE = 3


# =====================================
# 配置示例 3: 噪声图像处理
# Noisy Image Processing
# =====================================
class NoisyImageConfig:
    """噪声图像处理配置 - 适用于高噪声环境"""
    
    # 基本设置
    INPUT_PATH = "assets/examples"
    OUTPUT_DIR = "./depth_output_denoised"
    ENCODER = "vitb"
    INPUT_SIZE = 518
    
    # 输出设置
    SAVE_ORIGINAL_COMPARISON = True
    SAVE_DEPTH_ONLY = True
    SAVE_16BIT_RAW = True
    SAVE_DEPTH_STATISTICS = True
    
    # === 图像预处理 - 强力去噪 ===
    ENABLE_ADVANCED_PREPROCESSING = True
    
    # 强力去噪
    APPLY_DENOISING = True
    DENOISE_H = 15               # 高强度去噪
    DENOISE_TEMPLATE_WINDOW = 7
    DENOISE_SEARCH_WINDOW = 21
    
    # CLAHE适度增强
    APPLY_CLAHE = True
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_GRID_SIZE = (8, 8)
    
    # === 深度后处理 - 多层滤波 ===
    ENABLE_DEPTH_POSTPROCESSING = True
    
    # 中值滤波去除椒盐噪声
    APPLY_MEDIAN_FILTER = True
    MEDIAN_KERNEL_SIZE = 5
    
    # 双边滤波平滑
    APPLY_BILATERAL_FILTER = True
    BILATERAL_D = 9
    BILATERAL_SIGMA_COLOR = 80
    BILATERAL_SIGMA_SPACE = 80
    
    # 形态学操作清理
    APPLY_MORPHOLOGICAL_OPS = True
    MORPH_OPERATION = 'opening'  # 开运算去除小噪点
    MORPH_KERNEL_SIZE = 3
    MORPH_ITERATIONS = 1


# =====================================
# 配置示例 4: 快速批量处理
# Fast Batch Processing
# =====================================
class FastBatchConfig:
    """快速批量处理配置 - 适用于大量图像的快速处理"""
    
    # 基本设置
    INPUT_PATH = "assets/examples"
    OUTPUT_DIR = "./depth_output_fast"
    ENCODER = "vits"             # 使用最快的模型
    INPUT_SIZE = 256             # 较小的输入尺寸提高速度
    
    # 输出设置 - 简化输出
    SAVE_ORIGINAL_COMPARISON = False
    SAVE_DEPTH_ONLY = True
    SAVE_16BIT_RAW = False
    SAVE_NUMPY = False
    SAVE_DEPTH_STATISTICS = False
    
    # === 预处理和后处理关闭以提高速度 ===
    ENABLE_ADVANCED_PREPROCESSING = False
    ENABLE_DEPTH_POSTPROCESSING = False
    
    # 基本设置优化
    IMAGE_INTERPOLATION = cv2.INTER_LINEAR  # 更快的插值
    COLORMAP = 'viridis'                   # 简单颜色映射


# =====================================
# 配置示例 5: 艺术效果处理
# Artistic Effect Processing
# =====================================
class ArtisticEffectConfig:
    """艺术效果处理配置 - 创建艺术化的深度图"""
    
    # 基本设置
    INPUT_PATH = "assets/examples"
    OUTPUT_DIR = "./depth_output_artistic"
    ENCODER = "vitb"
    INPUT_SIZE = 518
    
    # 输出设置
    SAVE_ORIGINAL_COMPARISON = True
    SAVE_DEPTH_ONLY = True
    SAVE_16BIT_RAW = True
    COLORMAP = 'plasma'          # 艺术化颜色映射
    
    # === 图像预处理 - 艺术化效果 ===
    ENABLE_ADVANCED_PREPROCESSING = True
    
    # 增强对比度
    APPLY_CLAHE = True
    CLAHE_CLIP_LIMIT = 4.0       # 高对比度
    CLAHE_GRID_SIZE = (4, 4)     # 较小网格创建局部效果
    
    # 锐化增强
    APPLY_SHARPENING = True
    SHARPEN_STRENGTH = 1.5       # 强锐化
    
    # === 深度后处理 - 艺术化深度 ===
    ENABLE_DEPTH_POSTPROCESSING = True
    
    # 深度反转创建特殊效果
    INVERT_DEPTH = True
    
    # 形态学梯度突出边缘
    APPLY_MORPHOLOGICAL_OPS = True
    MORPH_OPERATION = 'gradient'
    MORPH_KERNEL_SIZE = 5
    MORPH_ITERATIONS = 1
    
    # 轻微高斯模糊创建柔和效果
    APPLY_GAUSSIAN_SMOOTHING = True
    GAUSSIAN_KERNEL_SIZE = 3
    GAUSSIAN_SIGMA = 1.0


# =====================================
# 配置示例 6: 科学研究配置
# Scientific Research Configuration
# =====================================
class ScientificResearchConfig:
    """科学研究配置 - 用于精确测量和分析"""
    
    # 基本设置
    INPUT_PATH = "assets/examples"
    OUTPUT_DIR = "./depth_output_research"
    ENCODER = "vitl"             # 最高精度模型
    INPUT_SIZE = 1024            # 最高分辨率
    
    # 输出设置 - 保存所有数据
    SAVE_ORIGINAL_COMPARISON = True
    SAVE_DEPTH_ONLY = True
    SAVE_GRAYSCALE = True
    SAVE_16BIT_RAW = True
    SAVE_NUMPY = True            # 原始数据
    SAVE_DEPTH_STATISTICS = True # 详细统计
    
    # === 最小化预处理保持原始性 ===
    ENABLE_ADVANCED_PREPROCESSING = False
    
    # === 保守的后处理 ===
    ENABLE_DEPTH_POSTPROCESSING = True
    
    # 仅轻微平滑去除明显错误
    APPLY_MEDIAN_FILTER = True
    MEDIAN_KERNEL_SIZE = 3       # 最小核大小
    
    # 高质量插值
    IMAGE_INTERPOLATION = cv2.INTER_CUBIC
    
    # 自定义参数保持精度
    KEEP_ASPECT_RATIO = True
    ENSURE_MULTIPLE_OF = 14
    RESIZE_METHOD = 'lower_bound'


# =====================================
# 配置示例 7: 移动设备优化
# Mobile Device Optimization
# =====================================
class MobileOptimizedConfig:
    """移动设备优化配置 - 适用于移动设备拍摄的图像"""
    
    # 基本设置
    INPUT_PATH = "assets/examples"
    OUTPUT_DIR = "./depth_output_mobile"
    ENCODER = "vits"             # 轻量模型
    INPUT_SIZE = 384             # 适中分辨率
    
    # 输出设置
    SAVE_ORIGINAL_COMPARISON = True
    SAVE_DEPTH_ONLY = True
    SAVE_16BIT_RAW = False
    SAVE_DEPTH_STATISTICS = False
    
    # === 移动图像特定预处理 ===
    ENABLE_ADVANCED_PREPROCESSING = True
    
    # 针对移动设备的图像质量问题
    APPLY_CLAHE = True
    CLAHE_CLIP_LIMIT = 2.5
    CLAHE_GRID_SIZE = (8, 8)
    
    # 轻微去噪
    APPLY_DENOISING = True
    DENOISE_H = 6
    DENOISE_TEMPLATE_WINDOW = 5
    DENOISE_SEARCH_WINDOW = 15
    
    # === 深度后处理 - 平衡质量和速度 ===
    ENABLE_DEPTH_POSTPROCESSING = True
    
    # 轻微平滑
    APPLY_GAUSSIAN_SMOOTHING = True
    GAUSSIAN_KERNEL_SIZE = 3
    GAUSSIAN_SIGMA = 0.8
    
    # 快速插值
    IMAGE_INTERPOLATION = cv2.INTER_LINEAR


# =====================================
# 使用说明
# Usage Instructions
# =====================================
"""
使用这些配置的步骤：

1. 选择适合的配置类
2. 在 generate_depth.py 中找到 Config 类
3. 用选择的配置类内容替换 Config 类
4. 运行脚本：python generate_depth.py

示例替换：
- 将 class Config: 改为 class Config(LowLightEnhancementConfig):
- 或者直接复制配置类的内容到 Config 类中

自定义调整：
- 可以混合不同配置的参数
- 根据具体需求调整参数值
- 启用/禁用特定功能

性能影响：
- 预处理会增加处理时间但改善输入质量
- 后处理会增加处理时间但改善输出质量
- 高分辨率设置会显著增加处理时间和内存使用
""" 