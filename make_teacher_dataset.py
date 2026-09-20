# -*- coding: utf-8 -*-
"""
把人工复核后的种子标签打包成 YOLO 训练集（给 Kaggle 训练“教师模型”用）
=======================================================================
输入: 一张图一个目录(images) + 同名 .txt(labels)
输出: E:/apple/teacher_dataset/  (images/train images/val labels/train labels/val + data.yaml)

用法:
  python make_teacher_dataset.py
  （默认读取 E:/apple/vlm_seed_400/ 下的 images 与 out/labels）
  可参数覆盖: --images <目录> --labels <目录> --out <输出目录>
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "vlm_seed_400" / "images"))
    ap.add_argument("--labels", default=str(ROOT / "vlm_seed_400" / "out" / "labels"))
    ap.add_argument("--out", default=str(ROOT / "teacher_dataset"))
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out = Path(args.out)
    if not images_dir.is_dir():
        print(f"图片目录不存在: {images_dir}")
        sys.exit(2)

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print("图片目录为空")
        sys.exit(2)

    random.seed(args.seed)
    random.shuffle(images)
    n_val = max(1, round(len(images) * args.val_ratio))
    val_set, train_set = set(images[:n_val]), images[n_val:]

    def ensure(*parts):
        d = out.joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    total_pos = total_neg = 0
    for split, imgs in (("train", train_set), ("val", val_set)):
        idir = ensure("images", split)
        ldir = ensure("labels", split)
        for f in imgs:
            lab = labels_dir / (f.stem + ".txt")
            boxes = []
            if lab.exists():
                for line in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
                    sp = line.split()
                    if len(sp) == 5:
                        boxes.append(line.strip())
            shutil.copy2(f, idir / f.name)
            (ldir / (f.stem + ".txt")).write_text("\n".join(boxes), encoding="utf-8")
            if boxes:
                total_pos += 1
            else:
                total_neg += 1

    yaml = (
        f"path: {out.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        'names: ["defect"]\n'
    )
    (out / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"输出: {out}")
    print(f"总图数: {len(images)}  有缺陷: {total_pos}  无缺陷(负样本): {total_neg}")
    print(f"  train: {len(train_set)} 张  val: {len(val_set)} 张")
    print("打包 zip 后即可传到 Kaggle 训练：")
    print("  (可选) python -c \"import shutil; shutil.make_archive(r'teacher_dataset', 'zip', r'%s')\"" % out)


if __name__ == "__main__":
    main()
