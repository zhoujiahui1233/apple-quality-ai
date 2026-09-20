# -*- coding: utf-8 -*-
"""重建缺陷检测数据集的无泄漏切分。

问题：defect_dataset_v2 的 train/val 是按文件名切开的，同一段视频的相邻帧
（如 extracted_00006 在 train、00029 在 val）高度相似却分处两边，导致验证
mAP 虚高。

做法：把 train+val 全部图合池 -> dHash 聚类（同源帧并簇）-> 按簇切分。

用法:
  python scripts/rebuild_defect_split.py [--val-ratio 0.15]
"""
import argparse
import csv
import os
import random
import shutil
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "defect_dataset_v2")
OUT = os.path.join(ROOT, "defect_dataset_clean")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def dhash(path):
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None
    if img is None:
        return None
    g = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if not os.path.exists(dst):
            os.link(src, dst)
    except Exception:
        if not os.path.exists(dst):
            shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--thresh", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    imgs = []
    for split in ("train", "val"):
        d = os.path.join(SRC, "images", split)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(IMG_EXT):
                imgs.append(os.path.join(d, f))
    imgs.sort(key=os.path.basename)
    print("合池图片:", len(imgs))

    hashes, keep = [], []
    for i, p in enumerate(imgs):
        h = dhash(p)
        if h is not None:
            hashes.append(h)
            keep.append(i)
    imgs = [imgs[i] for i in keep]
    hashes = np.array(hashes, dtype=np.uint64)
    print("有效哈希:", len(hashes))

    n = len(hashes)
    dsu = DSU(n)
    seen = {}
    for i in range(n):
        k = int(hashes[i])
        if k in seen:
            dsu.union(seen[k], i)
        else:
            seen[k] = i
    for s in range(0, n, 256):
        e = min(n, s + 256)
        xor = hashes[s:e, None] ^ hashes[None, :]
        x = xor - ((xor >> 1) & 0x5555555555555555)
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
        pc = (x * 0x0101010101010101) >> 56
        ii, jj = np.where(pc <= args.thresh)
        for a, b in zip(ii.tolist(), jj.tolist()):
            gi = s + a
            if gi != b:
                dsu.union(gi, b)
    groups = {}
    for i in range(len(hashes)):
        groups.setdefault(dsu.find(i), []).append(i)
    print("簇数:", len(groups), "最大簇:", max(len(v) for v in groups.values()))

    # 只保留有标注框的图
    def label_of(p):
        t = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        t = os.path.splitext(t)[0] + ".txt"
        return t if os.path.exists(t) else None

    keys = list(groups.items())
    random.Random(args.seed).shuffle(keys)
    n_val_target = int(len(imgs) * args.val_ratio)
    stat = {"train": 0, "val": 0}
    rows = []
    n_val = 0
    for r, idxs in keys:
        # 整簇归一边；避免巨簇把 val 撑爆
        if n_val + len(idxs) <= n_val_target * 1.15:
            split = "val"
            n_val += len(idxs)
        else:
            split = "train"
        for i in idxs:
            p = imgs[i]
            lp = label_of(p)
            rows.append([os.path.basename(p), split, r, os.path.relpath(p, ROOT)])
            link_or_copy(p, os.path.join(OUT, "images", split, os.path.basename(p)))
            if lp:
                link_or_copy(lp, os.path.join(OUT, "labels", split, os.path.basename(lp)))
            stat[split] += 1

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\nnc: 1\nnames: [\"defect\"]\n"
                % OUT.replace("\\", "/"))
    with open(os.path.join(OUT, "manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filename", "split", "cluster", "path"])
        w.writerows(rows)

    print("\n输出:", OUT)
    print("  train %d / val %d" % (stat["train"], stat["val"]))
    print("  data.yaml + manifest.csv 已写入")


if __name__ == "__main__":
    main()
