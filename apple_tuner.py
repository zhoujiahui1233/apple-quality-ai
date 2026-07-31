"""
苹果缺陷检测 - 参数调优工具
运行: python apple_tuner.py <图片路径>
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np

# 从主程序导入函数
from apple_defect_detector import (
    preprocess, segment_apple, detect_defects, classify_quality,
    visualize_result, IMAGE_WIDTH, APPLE_HSV_RANGES, DEFECT_HSV_RANGES,
    PREMIUM_THRESHOLD, STANDARD_THRESHOLD
)


def show_diagnostic(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("无法加载图片"); return

    img_p, _ = preprocess(img)
    hsv = cv2.cvtColor(img_p, cv2.COLOR_BGR2HSV)

    # === 1. 苹果分割诊断 ===
    print("=" * 50)
    print("【1】苹果区域分割诊断")
    print("=" * 50)

    apple_mask, apple_contour = segment_apple(img_p)
    if apple_mask is None:
        print("  [X] 未检测到苹果！")
        print("  原因可能是 HSV 阈值不匹配当前图片")
        print("  建议调整 APPLE_HSV_RANGES 的数值")
        return

    # 显示每个 HSV 范围的单独效果
    for i, (lo, hi) in enumerate(APPLE_HSV_RANGES):
        m = cv2.inRange(hsv, lo, hi)
        pix = np.sum(m > 0)
        print(f"  APPLE 范围{i+1}: H({lo[0]}-{hi[0]}) S({lo[1]}-{hi[1]}) V({lo[2]}-{hi[2]}) -> {pix} 像素")

    apple_pixels = np.sum(apple_mask > 0)
    print(f"  [OK] 合并后苹果区域: {apple_pixels} 像素")

    # === 2. 缺陷检测诊断 ===
    print("\n" + "=" * 50)
    print("【2】缺陷检测诊断")
    print("=" * 50)

    # HSV 颜色方法 - 逐范围显示
    print("  --- HSV 颜色方法 ---")
    hsv_defect = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for i, (lo, hi) in enumerate(DEFECT_HSV_RANGES):
        m = cv2.inRange(hsv, lo, hi)
        m = cv2.bitwise_and(m, apple_mask)
        pix = np.sum(m > 0)
        hsv_defect = cv2.bitwise_or(hsv_defect, m)
        print(f"  缺陷范围{i+1}: H({lo[0]}-{hi[0]}) S({lo[1]}-{hi[1]}) V({lo[2]}-{hi[2]}) -> {pix} 像素")

    # Laplacian 纹理方法
    print("  --- Laplacian 纹理方法 ---")
    gray = cv2.cvtColor(img_p, cv2.COLOR_BGR2GRAY)
    gray_filtered = cv2.bilateralFilter(gray, 9, 50, 50)
    lap = cv2.Laplacian(gray_filtered, cv2.CV_64F)
    lap_abs = np.uint8(np.abs(lap))
    mean_val = np.mean(lap_abs[apple_mask > 0])
    std_val = np.std(lap_abs[apple_mask > 0])
    threshold = mean_val + std_val * 1.0
    _, tex_mask = cv2.threshold(lap_abs, threshold, 255, cv2.THRESH_BINARY)
    tex_defect = cv2.bitwise_and(tex_mask, apple_mask)
    tex_pix = np.sum(tex_defect > 0)
    print(f"  Laplacian 纹理检测 -> {tex_pix} 像素")

    # 合并结果
    defect_mask, defect_contours = detect_defects(img_p, apple_mask)
    defect_pixels = np.sum(defect_mask > 0)
    print(f"  [OK] 合并后缺陷: {defect_pixels} 像素 ({defect_pixels/apple_pixels*100:.2f}%)")

    # === 3. 分级结果 ===
    grade, ratio, color = classify_quality(defect_pixels, apple_pixels)
    print(f"\n【3】品质分级: {grade}  (缺陷占比 {ratio*100:.2f}%)")
    print(f"  阈值: 优质 < {PREMIUM_THRESHOLD*100}% | 合格 < {STANDARD_THRESHOLD*100}% | 不合格 >= {STANDARD_THRESHOLD*100}%")

    # === 4. 生成诊断图 ===
    result = visualize_result(img_p, apple_contour, defect_contours, defect_mask, grade, ratio, color)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    # 保存各步骤的中间结果
    h, w = img_p.shape[:2]
    diag = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    # 原图
    diag[0:h, 0:w] = img_p
    cv2.putText(diag, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 苹果 mask
    apple_vis = cv2.cvtColor(apple_mask, cv2.COLOR_GRAY2BGR)
    diag[0:h, w:w*2] = apple_vis
    cv2.putText(diag, "Apple Mask", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 缺陷 mask (HSV)
    hsv_vis = cv2.cvtColor(hsv_defect, cv2.COLOR_GRAY2BGR)
    diag[h:h*2, 0:w] = hsv_vis
    cv2.putText(diag, "HSV Defects", (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 缺陷 mask (Texture)
    tex_vis = cv2.cvtColor(tex_defect, cv2.COLOR_GRAY2BGR)
    diag[h:h*2, w:w*2] = tex_vis
    cv2.putText(diag, "Texture Defects", (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    diag_path = os.path.join(out_dir, "diagnostic.jpg")
    cv2.imwrite(diag_path, diag)
    print(f"\n诊断图已保存: {diag_path}")

    # 保存检测结果
    result_path = os.path.join(out_dir, "tuned_result.jpg")
    cv2.imwrite(result_path, result)
    print(f"检测结果: {result_path}")

    print("\n[TIP] 提示: 查看 output/diagnostic.jpg 了解各步骤效果")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python apple_tuner.py <图片路径>")
        sys.exit(1)
    show_diagnostic(sys.argv[1])
