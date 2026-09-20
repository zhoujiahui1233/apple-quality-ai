# -*- coding: utf-8 -*-
"""
合并生成缺陷检测训练集 v2
=========================
来源1: 400 张人工复核种子 (vlm_seed_400)
来源2: 教师模型高置信自动框 (teacher_autolabel_out 中 status=accept)
输出: E:/apple/defect_dataset_v2  (images/train|val, labels/train|val, data.yaml)
"""
import argparse
import csv
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


def copy_entry(img_src: Path, labels_dir: Path, dst_img: Path, dst_lab: Path,
               keep_empty: bool):
    lab = labels_dir / (img_src.stem + ".txt")
    lines = []
    if lab.exists():
        for ln in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
            sp = ln.split()
            if len(sp) == 5:
                lines.append(ln.strip())
    if not lines and not keep_empty:
        return False
    shutil.copy2(img_src, dst_img)
    dst_lab.write_text("\n".join(lines), encoding="utf-8")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "defect_dataset_v2"))
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()
    out = Path(args.out)

    # ---- 来源1: 400 张人工种子 ----
    seed_img_dir = ROOT / "vlm_seed_400" / "images"
    seed_lab_dir = ROOT / "vlm_seed_400" / "out" / "labels"
    entries = []  # (img_path, labels_dir)
    if seed_img_dir.is_dir():
        for f in seed_img_dir.iterdir():
            if f.suffix.lower() in IMG_EXTS:
                entries.append((f, seed_lab_dir, True))
    print(f"来源1 人工种子图片: {len(entries)}")

    # ---- 来源2: 高置信自动框 ----
    auto_img_root = ROOT / "dataset_collected" / "apple" / "不合格"
    auto_lab_dir = ROOT / "teacher_autolabel_out" / "labels"
    summary = ROOT / "teacher_autolabel_out" / "summary.csv"
    n2 = 0
    if summary.exists():
        with open(summary, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "accept":
                    # 找源图片(唯一 stem, 优先 jpg)
                    cand = None
                    for ext in (".jpg", ".jpeg", ".png"):
                        p = auto_img_root / (row["file"])
                        if p.exists():
                            cand = p
                            break
                    if cand is None:
                        p = auto_img_root / (Path(row["file"]).stem + ".jpg")
                        if p.exists():
                            cand = p
                    if cand is not None:
                        entries.append((cand, auto_lab_dir, False))
                        n2 += 1
    print(f"来源2 高置信自动框: {n2}")

    random.seed(args.seed)
    random.shuffle(entries)
    n_val = max(1, round(len(entries) * args.val_ratio))
    n_neg = total = 0
    for split, group in (("train", entries[n_val:]), ("val", entries[:n_val])):
        idir = out / "images" / split
        ldir = out / "labels" / split
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        for img_src, lab_dir, keep_empty in group:
            if copy_entry(img_src, lab_dir, idir / img_src.name,
                          ldir / (img_src.stem + ".txt"), keep_empty):
                total += 1
                lab = ldir / (img_src.stem + ".txt")
                if lab.stat().st_size == 0:
                    n_neg += 1

    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\n"
        'names: ["defect"]\n', encoding="utf-8")
    print(f"\n输出: {out}")
    print(f"总有效图: {total}  (train {len(entries) - n_val}, val {n_val})")
    print(f"其中负样本(人工确认干净): {n_neg}")
    import shutil as _s
    zp = _s.make_archive(str(out), "zip", str(out))
    print("zip:", zp)


if __name__ == "__main__":
    main()
