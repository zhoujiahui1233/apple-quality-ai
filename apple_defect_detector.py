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


# ==================== 苹果区域分割 ====================

def segment_apple(img_bgr):
    """
    从背景中分割出苹果区域。
    利用 HSV 色彩空间 + 多范围阈值 + 形态学操作。
    返回: (mask, apple_contour)
      - mask: 二值图，苹果区域为白色 (255)
      - apple_contour: 苹果最外层轮廓（用于后续绘制）
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 合并所有 HSV 颜色范围
    full_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in APPLE_HSV_RANGES:
        mask = cv2.inRange(hsv, lower, upper)
        full_mask = cv2.bitwise_or(full_mask, mask)

    # 形态学操作
    # 开运算：先腐蚀后膨胀，去除小噪点
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel_open)

    # 闭运算：先膨胀后腐蚀，填充苹果内部小空洞
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # 找到所有外部轮廓，取面积最大的（即苹果）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    apple_contour = max(contours, key=cv2.contourArea)

    # 用最大轮廓重新生成精确 mask（去掉轮廓外的残留）
    apple_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(apple_mask, [apple_contour], -1, 255, -1)

    return apple_mask, apple_contour


# ==================== 缺陷检测 ====================

def detect_defects(img_bgr, apple_mask):
    """
    结合 HSV 颜色阈值 + Laplacian 纹理分析的缺陷检测。
    返回: (defect_mask, defect_contours)
      - defect_mask: 二值图，缺陷像素为白色 (255)
      - defect_contours: 过滤后缺陷轮廓列表（面积 > 20 像素）
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Method 1: HSV 颜色阈值
    hsv_defect_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in DEFECT_HSV_RANGES:
        mask = cv2.inRange(hsv, lower, upper)
        hsv_defect_mask = cv2.bitwise_or(hsv_defect_mask, mask)

    # Method 2: Laplacian 纹理分析
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_abs = np.uint8(np.abs(laplacian))
    _, texture_defect_mask = cv2.threshold(laplacian_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 结合两种方法
    combined = cv2.bitwise_or(hsv_defect_mask, texture_defect_mask)

    # 只保留苹果区域内的缺陷
    defect_mask = cv2.bitwise_and(combined, apple_mask)

    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_CLOSE, kernel)

    # 找轮廓，过滤小区域
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_contours = [c for c in contours if cv2.contourArea(c) > 20]

    return defect_mask, defect_contours



# ==================== 主函数 ====================
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

        # 2. 分割苹果区域
        apple_mask, apple_contour = segment_apple(img_processed)
        if apple_mask is None:
            print(f"  [分割] ❌ 未检测到苹果区域，跳过")
            return
        apple_pixels = np.sum(apple_mask > 0)
        total_pixels = apple_mask.shape[0] * apple_mask.shape[1]
        area_ratio = apple_pixels / total_pixels * 100
        print(f"  [分割] 完成 → 苹果区域: {apple_pixels} 像素 ({area_ratio:.1f}% 画面占比)")

        # 3. 缺陷检测
        defect_mask, defect_contours = detect_defects(img_processed, apple_mask)
        defect_pixels = np.sum(defect_mask > 0)
        defect_count = len(defect_contours)
        print(f"  [检测] 完成 → 发现 {defect_count} 个缺陷区域，共 {defect_pixels} 像素")

    except Exception as e:
        print(f"  [错误] ❌ {e}")

if __name__ == "__main__":
    main()