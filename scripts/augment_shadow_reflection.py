# -*- coding: utf-8 -*-
"""
阴影 / 反光 / 光照 数据增强脚本
用途：为训练集离线生成"带阴影、反光、明暗变化"的合成样本，
      缓解真实场景中阴影被误判为缺陷、反光被漏检的问题。
用法：
    python scripts/augment_shadow_reflection.py \
        --input E:\apple\dataset_collected \
        --output E:\apple\dataset_collected_augmented_shadow \
        --per-image 2
输出目录会保持与输入相同的目录结构（品类/标签/图片）。
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def soft_shadow(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """在图片一侧叠加一块半透明渐变阴影（模拟拍摄时的阴影遮挡）"""
    h, w = img.shape[:2]
    out = img.copy()
    mask = np.zeros((h, w), dtype=np.float32)
    cx = rng.uniform(0.0, 1.0) * w
    cy = rng.uniform(0.0, 1.0) * h
    rx = rng.uniform(0.25, 0.55) * w
    ry = rng.uniform(0.25, 0.55) * h
    yy, xx = np.mgrid[0:h, 0:w]
    dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    mask = np.clip(1.0 - dist, 0.0, 1.0)
    alpha = rng.uniform(0.25, 0.55)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(rx, ry) * 0.25)
    mask = (mask * alpha * 255).astype(np.uint8)
    dark = np.full_like(out, (10, 10, 15))
    out = cv2.addWeighted(out, 1.0, dark, 1.0, 0)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    out = (out.astype(np.float32) * (1 - mask3) + dark.astype(np.float32) * mask3).astype(np.uint8)
    return out


def specular_highlight(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """叠加一两个高光光斑（模拟光滑表面的反光）"""
    h, w = img.shape[:2]
    out = img.copy()
    n = rng.randint(1, 2)
    overlay = np.zeros_like(img)
    for _ in range(n):
        cx = int(rng.uniform(0.15, 0.85) * w)
        cy = int(rng.uniform(0.15, 0.85) * h)
        rx = int(rng.uniform(0.06, 0.16) * w)
        ry = int(rng.uniform(0.06, 0.16) * h)
        angle = rng.uniform(0, 180)
        light = np.full_like(img, (235, 235, 245))
        cv2.ellipse(light, (cx, cy), (rx, ry), angle, 0, 360, (255, 255, 255), -1)
        light = cv2.GaussianBlur(light, (0, 0), sigmaX=rx * 0.4)
        overlay = cv2.max(overlay, light)
    # 用"变亮"方式混合，让高光只提亮、不压暗
    out = cv2.max(out, cv2.addWeighted(out, 0.55, overlay, 0.45, 0))
    return out


def brightness_contrast(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """随机亮度/对比度扰动（模拟不同曝光环境）"""
    alpha = rng.uniform(0.75, 1.25)
    beta = rng.uniform(-25, 25)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def vignette(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """轻微暗角（模拟镜头边缘暗影）"""
    h, w = img.shape[:2]
    strength = rng.uniform(0.10, 0.30)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    mask = 1.0 - strength * np.clip(dist - 0.6, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=w * 0.08)
    out = img.astype(np.float32) * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


AUG_FUNCS = [soft_shadow, specular_highlight, brightness_contrast, vignette]


def main():
    ap = argparse.ArgumentParser(description="生成阴影/反光/光照增强样本")
    ap.add_argument("--input", default="dataset_collected", help="输入数据集目录")
    ap.add_argument("--output", default="dataset_collected_augmented_shadow", help="输出目录")
    ap.add_argument("--per-image", type=int, default=2, help="每张图生成几张增强图")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src_root = Path(args.input)
    dst_root = Path(args.output)
    if not src_root.is_dir():
        print(f"[错误] 输入目录不存在: {src_root}")
        return

    rng = random.Random(args.seed)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in src_root.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    total = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        rel = img_path.relative_to(src_root)
        out_dir = dst_root / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = img_path.stem
        suffix = ".jpg"
        for k in range(args.per_image):
            aug = img.copy()
            # 随机 1~2 种增强叠加，保证多样且不至于过度失真
            chosen = rng.sample(AUG_FUNCS, rng.randint(1, 2))
            for fn in chosen:
                aug = fn(aug, rng)
            out_path = out_dir / f"{stem}__shadow{k}{suffix}"
            cv2.imwrite(str(out_path), aug, [cv2.IMWRITE_JPEG_QUALITY, 95])
            total += 1
    print(f"完成：处理 {len(images)} 张原图，生成 {total} 张增强图 -> {dst_root}")


if __name__ == "__main__":
    main()