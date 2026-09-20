# -*- coding: utf-8 -*-
"""
打包 DL 外观分类训练集 v2（用你自己的 合格/不合格 图）
=====================================================
来源: E:\apple\dataset_collected\apple\合格 与 不合格（去重，优先 .jpg）
输出: E:\apple\dl_dataset_v2\   (train/good|bad, val/good|bad)
      label: good=0(合格)  bad=1(不合格)
"""
import random
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def unique_images(d: Path):
    seen = {}
    for f in d.iterdir():
        if f.suffix.lower() not in IMG_EXTS:
            continue
        if f.stem not in seen or f.suffix.lower() == ".jpg":
            seen[f.stem] = f
    return list(seen.values())


def main():
    out = ROOT / "dl_dataset_v2"
    good = unique_images(ROOT / "dataset_collected" / "apple" / "合格")
    bad = unique_images(ROOT / "dataset_collected" / "apple" / "不合格")
    print(f"合格唯一图: {len(good)}  不合格唯一图: {len(bad)}")

    # 类别均衡：各自都取 min 张
    n = min(len(good), len(bad))
    random.seed(20260903)
    random.shuffle(good); random.shuffle(bad)
    good = good[:n]; bad = bad[:n]
    print(f"均衡后各 {n} 张，共 {2 * n} 张")

    n_val = max(1, round(n * 0.15))
    for split, gi, bi in (("train", good[n_val:], bad[n_val:]),
                          ("val", good[:n_val], bad[:n_val])):
        for label, cls, files in (("good", "0", gi), ("bad", "1", bi)):
            d = out / "images" / split / label
            d.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, d / f.name)
        print(f"{split}: good {len(gi)} / bad {len(bi)}")

    print(f"\n输出: {out}")
    zp = shutil.make_archive(str(out), "zip", str(out))
    print("zip:", zp)


if __name__ == "__main__":
    main()
