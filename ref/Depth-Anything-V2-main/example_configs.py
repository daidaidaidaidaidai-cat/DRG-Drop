# -*- coding: utf-8 -*-
"""
配置示例文件
Different configuration examples for various use cases
"""

# =====================================
# 配置示例 1: 基础图像处理
# Basic image processing
# =====================================
class BasicConfig:
    # 基本路径设置
    INPUT_PATH = "assets/examples"
    OUTPUT_DIR = "./depth_output"
    
    # 模型设置 - 使用中等模型平衡速度和精度
    ENCODER = "vitb"
    INPUT_SIZE = 518
    
    # 输出设置 - 只保存深度图和对比图
    SAVE_ORIGINAL_COMPARISON = True
    SAVE_DEPTH_ONLY = True
    SAVE_GRAYSCALE = False
    SAVE_16BIT_RAW = False
    SAVE_NUMPY = False
    
    # 其他设置
    SHOW_PROGRESS = True
    RECURSIVE_SEARCH = True


# =====================================
# 配置示例 2: 高质量处理
# High quality processing
# =====================================
class HighQualityConfig:
    # 基本路径设置
    INPUT_PATH = "/path/to/your/images"
    OUTPUT_DIR = "./high_quality_depth"
    
    # 模型设置 - 使用大模型获得最佳精度
    ENCODER = "vitl"  # 最高精度
    INPUT_SIZE = 1024  # 更高分辨率
    
    # 输出设置 - 保存所有格式
    SAVE_ORIGINAL_COMPARISON = True
    SAVE_DEPTH_ONLY = True
    SAVE_GRAYSCALE = True
    SAVE_16BIT_RAW = True
    SAVE_NUMPY = True
    
    # 可视化设置
    COLORMAP = 'plasma'
    DEPTH_RANGE_PERCENTILE = (1, 99)
    
    # 其他设置
    SHOW_PROGRESS = True
    PRINT_STATS = True


# =====================================
# 配置示例 3: 快速批量处理
# Fast batch processing
# =====================================
class FastBatchConfig:
    # 基本路径设置
    INPUT_PATH = "/path/to/batch/images"
    OUTPUT_DIR = "./batch_depth_output"
    
    # 模型设置 - 使用小模型获得最快速度
    ENCODER = "vits"  # 最快速度
    INPUT_SIZE = 256  # 较低分辨率
    
    # 输出设置 - 只保存必要的文件
    SAVE_ORIGINAL_COMPARISON = False
    SAVE_DEPTH_ONLY = True
    SAVE_GRAYSCALE = True  # 灰度图文件更小
    SAVE_16BIT_RAW = False
    SAVE_NUMPY = False
    
    # 其他设置
    SHOW_PROGRESS = True
    RECURSIVE_SEARCH = True


# =====================================
# 配置示例 4: 研究用途 - 保留原始数据
# Research purpose - keep raw data
# =====================================
class ResearchConfig:
    # 基本路径设置
    INPUT_PATH = "/path/to/research/images"
    OUTPUT_DIR = "./research_depth_data"
    
    # 模型设置
    ENCODER = "vitb"
    INPUT_SIZE = 518
    
    # 输出设置 - 重点保存原始数据
    SAVE_ORIGINAL_COMPARISON = True
    SAVE_DEPTH_ONLY = True
    SAVE_GRAYSCALE = False
    SAVE_16BIT_RAW = True  # 保存16位原始深度
    SAVE_NUMPY = True      # 保存numpy格式便于后续处理
    
    # 可视化设置
    COLORMAP = 'Spectral_r'
    DEPTH_RANGE_PERCENTILE = None  # 不使用百分位数裁剪
    
    # 文件命名
    DEPTH_SUFFIX = "_depth"
    RAW_SUFFIX = "_raw_depth"
    COMPARISON_SUFFIX = "_analysis"
    
    # 其他设置
    SHOW_PROGRESS = True
    PRINT_STATS = True


# =====================================
# 配置示例 5: 单张图像处理
# Single image processing
# =====================================
class SingleImageConfig:
    # 基本路径设置
    INPUT_PATH = "/path/to/single/image.jpg"
    OUTPUT_DIR = "./single_image_depth"
    
    # 模型设置
    ENCODER = "vitl"  # 高精度
    INPUT_SIZE = 1024
    
    # 输出设置
    SAVE_ORIGINAL_COMPARISON = True
    SAVE_DEPTH_ONLY = True
    SAVE_GRAYSCALE = False
    SAVE_16BIT_RAW = True
    SAVE_NUMPY = False
    
    # 可视化设置
    COLORMAP = 'viridis'
    DEPTH_RANGE_PERCENTILE = (2, 98)
    
    # 其他设置
    SHOW_PROGRESS = False  # 单张图像不需要进度条
    RECURSIVE_SEARCH = False


# =====================================
# 如何使用这些配置:
# How to use these configurations:
# =====================================
"""
方法1: 直接修改 generate_depth.py 中的 Config 类
Method 1: Directly modify the Config class in generate_depth.py

方法2: 在 generate_depth.py 的开头导入并替换
Method 2: Import and replace at the beginning of generate_depth.py

在 generate_depth.py 中添加:
Add to generate_depth.py:

from example_configs import HighQualityConfig as Config

然后运行:
Then run:
python generate_depth.py

方法3: 创建自定义配置类
Method 3: Create custom configuration class

class MyCustomConfig:
    INPUT_PATH = "/my/custom/path"
    OUTPUT_DIR = "./my_output"
    ENCODER = "vitb"
    # ... other settings
""" 