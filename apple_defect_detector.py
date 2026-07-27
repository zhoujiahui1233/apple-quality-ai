"""
苹果表面缺陷识别程序 — Apple Surface Defect Detector
基于 OpenCV + NumPy，通过图像处理技术检测苹果表面缺陷并进行品质分级。
"""
import cv2
import numpy as np
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# ==================== 全局配置参数 ====================
IMAGE_WIDTH = 500
PREMIUM_THRESHOLD = 0.03
STANDARD_THRESHOLD = 0.15

APPLE_HSV_RANGES = [
    (np.array([0, 30, 40]), np.array([10, 255, 255])),
    (np.array([160, 30, 40]), np.array([180, 255, 255])),
    (np.array([35, 30, 40]), np.array([85, 255, 255])),
]

DEFECT_HSV_RANGES = [
    (np.array([0, 30, 0]), np.array([30, 255, 100])),
    (np.array([0, 0, 0]), np.array([180, 255, 60])),
    (np.array([0, 0, 0]), np.array([180, 255, 40])),
]

# ==================== 工具函数 ====================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法加载图片: {path}")
    return img

def ensure_output_dir(path="output"):
    os.makedirs(path, exist_ok=True)


# ==================== 图像预处理 ====================

def preprocess(img):
    """
    图像预处理：
    1. 等比例缩放到统一宽度 (IMAGE_WIDTH)
    2. 高斯模糊降噪
    返回: (img_blurred, img_resized)
    """
    h, w = img.shape[:2]
    scale = IMAGE_WIDTH / w
    new_size = (IMAGE_WIDTH, int(round(h * scale)))
    img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # 高斯模糊去噪（5×5 核）
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)

    return img_blurred, img_resized


# ==================== 主函数占位 ====================
def main():
    parser = argparse.ArgumentParser(description="苹果表面缺陷识别程序")
    parser.add_argument("input", help="输入图片路径或包含图片的目录")
    args = parser.parse_args()
    input_path = args.input
    if os.path.isdir(input_path):
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [f for f in os.listdir(input_path) if Path(f).suffix.lower() in image_exts]
        if not files:
            print(f"错误: 目录 '{input_path}' 中没有找到图片文件")
            sys.exit(1)
        for fname in files:
            process_single_image(os.path.join(input_path, fname))
    else:
        process_single_image(input_path)

def process_single_image(image_path):
    """处理单张图片的完整流程。"""
    print(f"\n{'='*50}")
    print(f"处理: {image_path}")
    print(f"{'='*50}")
    try:
        # 1. 加载 & 预处理
        img_original = load_image(image_path)
        img_processed, img_resized = preprocess(img_original)
        print(f"  [预处理] 完成 → 尺寸: {img_processed.shape[1]}x{img_processed.shape[0]}")
    except Exception as e:
        print(f"  [错误] ❌ {e}")

if __name__ == "__main__":
    main()