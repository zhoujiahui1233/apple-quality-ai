# -*- coding: utf-8 -*-
"""切片分割数据集构建：伪标签生成 + 训练集打包。

流程：
  1. 扫描 dataset_collected/apple_slice 下所有切片照片（合格/不合格_* 等子目录均可）；
  2. 用内置规则算法为每张图生成 5 类分割掩膜（0 背景/1 果肉/2 果核/3 褐变/4 果皮）作为"伪标签"；
  3. 输出到 slice_seg_dataset/：images/ + masks/ + train.txt + val.txt + manifest.json + class_weights.json；
  4. 之后在"切片掩膜标注"工作台里人工修正掩膜，再运行训练脚本即可。

用法：
  python scripts/slice_seg_prepare.py [--src dataset_collected/apple_slice] [--out slice_seg_dataset]
"""
import argparse
import hashlib
import json
import sys
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import slice_detector as sd

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def imread_unicode(p: Path):
    """兼容中文路径读取（OpenCV imread 在 Windows 上不支持非 ASCII 路径）"""
    data = Path(p).read_bytes()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(p: Path, img, params=None):
    """兼容中文路径写入"""
    ok, buf = cv2.imencode(p.suffix, img, params or [])
    if not ok:
        return False
    Path(p).write_bytes(buf.tobytes())
    return True


def find_slice_images(src_root: Path):
    files = []
    if not src_root.is_dir():
        return files
    for f in sorted(src_root.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMG_SUFFIXES and not f.name.startswith("."):
            files.append(f)
    return files


def name_for(src_rel: str, index: int) -> str:
    h = hashlib.md5(src_rel.encode("utf-8")).hexdigest()[:10]
    return f"slice_{h}_{index:04d}.jpg"


def _class_weights(mask_files):
    """按类别像素频率反比计算损失权重（褐变/果核样本少，权重高）。"""
    counts = np.zeros(sd.SEG_PEEL + 1, dtype=np.float64)
    for mf in mask_files:
        m = imread_unicode(mf)
        if m is not None and len(m.shape) == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        if m is None:
            continue
        for c in range(sd.SEG_PEEL + 1):
            counts[c] += int(np.count_nonzero(m == c))
    eps = 1.0
    total = counts.sum()
    if total <= 0:
        return [1.0] * (sd.SEG_PEEL + 1)
    freq = (counts + eps) / (total + eps * (sd.SEG_PEEL + 1))
    w = 1.0 / freq
    w = np.clip(w, 0.05, 20.0)          # 防止稀有类权重过大会训练不稳定
    w = w / w.mean()
    return [round(float(x), 4) for x in w]


def build_dataset(src_root: Path, out_root: Path, force: bool = False,
                  split: float = 0.85, seed: int = 42) -> dict:
    images_dir = out_root / "images"
    masks_dir = out_root / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.json"
    manifest = {}
    if manifest_path.exists() and not force:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    src_files = find_slice_images(src_root)
    added = skipped = 0
    errors = []
    index = 0
    for f in src_files:
        try:
            rel = str(f.relative_to(src_root)).replace("\\", "/")
        except ValueError:
            rel = f.name
        name = name_for(rel, index)
        index += 1
        if name in manifest and not force:
            skipped += 1
            continue
        img = imread_unicode(f)
        if img is None:
            errors.append(f"{rel}: 读取失败")
            continue
        try:
            seg = sd.rule_segmentation(img)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{rel}: 分割失败 {e}")
            continue
        # 无有效切片区域则跳过
        if int(np.count_nonzero(seg)) == 0:
            errors.append(f"{rel}: 未检测到切片区域")
            continue
        img_path = images_dir / name
        mask_path = masks_dir / name.replace(".jpg", ".png")
        imwrite_unicode(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        imwrite_unicode(mask_path, seg)
        manifest[name] = {"src": rel, "label_dir": str(f.parent.relative_to(src_root)) if f.parent != src_root else ""}
        added += 1

    # 重写 manifest（保持稳定顺序）
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    names = sorted(manifest.keys())
    mask_files = [masks_dir / n.replace(".jpg", ".png") for n in names]

    # 训练/验证划分
    rng = random.Random(seed)
    rng.shuffle(names)
    n_train = max(1, int(len(names) * split)) if names else 0
    train_names = names[:n_train]
    val_names = names[n_train:]
    (out_root / "train.txt").write_text("\n".join(
        f"images/{n} masks/{n.replace('.jpg', '.png')}" for n in train_names), encoding="utf-8")
    (out_root / "val.txt").write_text("\n".join(
        f"images/{n} masks/{n.replace('.jpg', '.png')}" for n in val_names), encoding="utf-8")

    # 类别权重
    weights = _class_weights(mask_files)
    (out_root / "class_weights.json").write_text(
        json.dumps({"weights": weights, "classes": sd.SEG_CLASSES if hasattr(sd, "SEG_CLASSES") else None},
                   ensure_ascii=False), encoding="utf-8")

    return {"total_sources": len(src_files), "added": added, "skipped": skipped,
            "errors": errors, "total_in_dataset": len(names),
            "train": len(train_names), "val": len(val_names),
            "class_weights": weights}


def main():
    ap = argparse.ArgumentParser(description="切片分割伪标签与数据集构建")
    ap.add_argument("--src", default=str(ROOT / "dataset_collected" / "apple_slice"))
    ap.add_argument("--out", default=str(ROOT / "slice_seg_dataset"))
    ap.add_argument("--force", action="store_true", help="重新生成所有掩膜")
    ap.add_argument("--split", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(f"扫描: {args.src}")
    info = build_dataset(Path(args.src), Path(args.out), force=args.force,
                         split=args.split, seed=args.seed)
    print(json.dumps(info, ensure_ascii=False, indent=1))
    if info["errors"]:
        print("跳过图片:")
        for e in info["errors"]:
            print(" -", e)


if __name__ == "__main__":
    sys.exit(main())