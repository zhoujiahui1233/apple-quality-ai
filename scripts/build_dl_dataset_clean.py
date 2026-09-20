# -*- coding: utf-8 -*-
"""按 clean_dataset_v1/manifest.csv 打包无泄漏的 DL 分类训练集。

输出 dl_dataset_clean/{train,val}/{good,bad}/ ，并打包成 zip 供 Kaggle 上传。
用硬链接避免复制占用双倍磁盘（同盘符），失败则回退为复制。

用法:
  python scripts/build_dl_dataset_clean.py [--zip]
"""
import argparse
import csv
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "clean_dataset_v1", "manifest.csv")
OUT = os.path.join(ROOT, "dl_dataset_clean")


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return "skip"
    try:
        os.link(src, dst)
        return "link"
    except Exception:
        shutil.copy2(src, dst)
        return "copy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    with open(MANIFEST, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print("manifest rows:", len(rows))

    from collections import Counter
    stat = Counter()
    for r in rows:
        split, cls = r["split"], r["class"]
        if split == "gap":
            continue          # 隔离带：靠近切分边界，train/val 都不用
        d = os.path.join(OUT, "images", split, cls)
        os.makedirs(d, exist_ok=True)
        src = os.path.join(ROOT, r["path"])
        dst = os.path.join(d, r["filename"])
        try:
            mode = link_or_copy(src, dst)
            stat[(split, cls, mode)] += 1
        except Exception as e:
            stat[(split, cls, "FAIL")] += 1
            if stat[(split, cls, "FAIL")] <= 3:
                print("  失败:", r["filename"], e)

    print("\n输出:", OUT)
    for split in ("train", "val"):
        for cls in ("good", "bad"):
            n = sum(v for (s, c, _), v in stat.items() if s == split and c == cls)
            print("  %-5s %-4s %d" % (split, cls, n))
    modes = Counter(m for (_, _, m), v in stat.items() for _ in range(v))
    print("  链接方式统计:", dict(modes))

    if args.zip:
        print("打包 zip ...")
        zp = shutil.make_archive(OUT, "zip", OUT)
        print("zip:", zp, "(%.1f MB)" % (os.path.getsize(zp) / 1e6))


if __name__ == "__main__":
    main()
