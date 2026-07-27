"""
测试脚本 — 验证苹果缺陷识别程序的各功能模块是否正常。
生成模拟苹果图片进行端到端测试。
"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, "E:/apple")
from apple_defect_detector import (
    preprocess, segment_apple, detect_defects, classify_quality,
    visualize_result, save_result
)


def create_dummy_apple(with_defects=True):
    """
    生成模拟苹果图片用于测试。
    画一个红色圆形作为苹果，可选添加缺陷。
    """
    img = np.ones((400, 500, 3), dtype=np.uint8) * 200  # 灰色背景

    # 画一个红色苹果（圆形渐变让它更真实）
    center = (250, 200)
    radius = 150
    cv2.circle(img, center, radius, (0, 0, 200), -1)

    # 加一点渐变色
    for i in range(radius, 0, -1):
        alpha = 1.0 - i / radius * 0.3
        color = (int(0 * alpha), int(0 * alpha), int(200 * alpha))
        cv2.circle(img, center, i, color, -1)

    if with_defects:
        # 褐色碰伤斑
        cv2.circle(img, (200, 150), 25, (30, 80, 120), -1)
        # 黑色腐烂点
        cv2.circle(img, (310, 220), 15, (10, 10, 10), -1)
        # 褐色划痕（细长形）
        cv2.line(img, (180, 280), (260, 250), (50, 90, 130), 5)

    return img


def run_test(img, test_name):
    """对一张图片运行完整检测流程并输出结果。"""
    print(f"\n{'='*50}")
    print(f"测试: {test_name}")
    print(f"{'='*50}")

    # 1. 预处理
    img_p, img_r = preprocess(img)
    print(f"  [预处理] 尺寸: {img_p.shape[1]}x{img_p.shape[0]}")

    # 2. 分割
    mask, contour = segment_apple(img_p)
    if mask is None:
        print(f"  [分割] [FAIL] 未检测到苹果")
        return None
    apple_pixels = np.sum(mask > 0)
    print(f"  [分割] 苹果区域: {apple_pixels} 像素")

    # 3. 缺陷检测
    defect_mask, defect_contours = detect_defects(img_p, mask)
    defect_pixels = np.sum(defect_mask > 0)
    print(f"  [检测] {len(defect_contours)} 个缺陷, {defect_pixels} 缺陷像素")

    # 4. 分级
    grade, ratio, color = classify_quality(defect_pixels, apple_pixels)
    print(f"  [分级] 缺陷占比: {ratio*100:.2f}%")
    print(f"  [分级] 等级: {grade}")

    # 5. 可视化
    result = visualize_result(img_p, contour, defect_contours,
                              defect_mask, grade, ratio, color)
    out_dir = "E:/apple/output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test_{test_name}.jpg")
    cv2.imwrite(out_path, result)
    print(f"  [输出] 已保存: {out_path}")

    return grade, ratio


def test_good_apple():
    """测试好苹果：期望为优质或合格。"""
    img = create_dummy_apple(with_defects=False)
    result = run_test(img, "good_apple")
    if result is None:
        print("  [FAIL] 测试失败：未检测到苹果！")
        return False
    grade, ratio = result
    print(f"  {'[OK] 通过!' if '优质' in grade else '[!] 注意：等级不是优质'}")
    print()
    return True


def test_bad_apple():
    """测试坏苹果：期望检测到缺陷。"""
    img = create_dummy_apple(with_defects=True)
    result = run_test(img, "bad_apple")
    if result is None:
        print("  [FAIL] 测试失败：未检测到苹果！")
        return False
    grade, ratio = result
    has_defects = ratio > 0
    print(f"  {'[OK] 通过!' if has_defects else '[!] 注意：未检测到缺陷'}")
    print()
    return True


if __name__ == "__main__":
    print("="*50)
    print("苹果表面缺陷识别程序 — 自测试")
    print("="*50)

    ok1 = test_good_apple()
    ok2 = test_bad_apple()

    print("="*50)
    if ok1 and ok2:
        print("[ALL PASS] 所有测试通过！")
    else:
        print("[!]  部分测试未达到预期，请检查输出图片。")
    print(f"输出图片保存在 E:/apple/output/ 目录")
