# -*- coding: utf-8 -*-
"""
把收集的数据集合并导出为 YOLO 检测训练数据（自动生成检测框）

用法:
  python scripts/yolo_export.py --category apple

输出:
  yolo_detect_dataset/images/{train,val}/collected_*.jpg
  yolo_detect_dataset/labels/{train,val}/collected_*.txt
"""
import argparse
import hashlib
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import config
from apple_defect_detector import segment_apple

DETECT_DIR = config.ROOT_DIR / "yolo_detect_dataset"
IMG_W = 640
MIN_BOX_RATIO = 0.003
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _log(log, msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    if log is not None:
        log(line)


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def gen_yolo_label(mask, img_w, img_h):
    """按分割图产生 YOLO 检测框（0 x_center y_center width height）"""
    co, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not co:
        return None
    largest = max(co, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_BOX_RATIO * img_h * img_w:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"


def merge(category="apple", seed=42, log=None):
    src_dir = config.ROOT_DIR / "dataset_collected" / category
    if not src_dir.is_dir():
        raise RuntimeError(f"品类目录不存在: {category}")

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (DETECT_DIR / sub).mkdir(parents=True, exist_ok=True)

    images = []
    for label_dir in sorted(src_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        for f in sorted(label_dir.iterdir()):
            if f.suffix.lower() in IMG_EXTS:
                images.append((f, label_dir.name))
    if not images:
        raise RuntimeError("没有可导出的图片，请先收集数据")

    rng = random.Random(seed)
    rng.shuffle(images)
    n_train = max(1, int(len(images) * 0.8))

    exported = skipped = 0
    for i, (f, label) in enumerate(images):
        img = imread_unicode(f)
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        scale = IMG_W / w
        new_size = (IMG_W, max(1, int(round(h * scale))))
        img_r = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        rh, rw = img_r.shape[:2]

        mask, _ = segment_apple(cv2.GaussianBlur(img_r, (5, 5), 0))
        if mask is None:
            skipped += 1
            continue
        lbl = gen_yolo_label(mask, rw, rh)
        if lbl is None:
            skipped += 1
            continue

        digest = hashlib.md5(str(f).encode("utf-8")).hexdigest()[:8]
        stem = f"collected_{category}_{digest}_{f.stem}"
        split = "train" if i < n_train else "val"
        img_dst = DETECT_DIR / "images" / split / f"{stem}.jpg"
        lbl_dst = DETECT_DIR / "labels" / split / f"{stem}.txt"
        ok, buf = cv2.imencode(".jpg", img_r)
        if not ok:
            skipped += 1
            continue
        buf.tofile(str(img_dst))
        lbl_dst.write_text(lbl, encoding="utf-8")
        exported += 1
        if (i + 1) % 50 == 0:
            _log(log, f"  导出 {i + 1}/{len(images)}")

    _log(log, f"YOLO 训练集合并完成: 导出 {exported} 张（含自动检测框）, 跳过 {skipped} 张（未检测出主体）")
    return {"exported": exported, "skipped": skipped, "total": len(images)}


def main():
    parser = argparse.ArgumentParser(description="合并导出到 YOLO 检测训练集")
    parser.add_argument("--category", default="apple")
    args = parser.parse_args()
    merge(args.category)


if __name__ == "__main__":
    main()
