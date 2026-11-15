# -*- coding: utf-8 -*-
"""数据集体检：规模、重复度、标注质量、目标尺寸分布。

用法: python scripts/audit_datasets.py
"""
import os
import sys
import glob
import statistics

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def imread_gray(p):
    try:
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def dhash(p):
    g = imread_gray(p)
    if g is None:
        return None
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    v = 0
    for b in (s[:, 1:] > s[:, :-1]).flatten():
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


def cluster(paths, thresh=5):
    hs = np.array([h for h in (dhash(p) for p in paths) if h is not None], dtype=np.uint64)
    n = len(hs)
    dsu = DSU(n)
    seen = {}
    for i in range(n):
        k = int(hs[i])
        if k in seen:
            dsu.union(seen[k], i)
        else:
            seen[k] = i
    for s in range(0, n, 256):
        e = min(n, s + 256)
        xor = hs[s:e, None] ^ hs[None, :]
        x = xor - ((xor >> 1) & 0x5555555555555555)
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
        pc = (x * 0x0101010101010101) >> 56
        ii, jj = np.where(pc <= thresh)
        for a, b in zip(ii.tolist(), jj.tolist()):
            gi = s + a
            if gi != b:
                dsu.union(gi, b)
    groups = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)
    return hs, groups


def audit_classification():
    print("=" * 70)
    print("【1】外观二分类数据")
    print("=" * 70)
    pool = os.path.join(ROOT, "dataset_collected", "apple")
    for cls in ("合格", "不合格"):
        d = os.path.join(pool, cls)
        n = len([f for f in os.listdir(d) if f.lower().endswith(IMG_EXT)]) if os.path.isdir(d) else 0
        print("  %-6s %6d 张" % (cls, n))

    man = os.path.join(ROOT, "clean_dataset_v1", "manifest.csv")
    if os.path.exists(man):
        import csv
        rows = list(csv.DictReader(open(man, encoding="utf-8-sig")))
        tr = sum(1 for r in rows if r["split"] == "train")
        va = sum(1 for r in rows if r["split"] == "val")
        print("  清洗后切分: train %d / val %d" % (tr, va))
        bad = 0
        for r in rows:
            if not os.path.exists(os.path.join(ROOT, r["path"])):
                bad += 1
        print("  清单中文件缺失: %d" % bad)


def audit_defect():
    print()
    print("=" * 70)
    print("【2】缺陷检测数据")
    print("=" * 70)
    src = os.path.join(ROOT, "defect_dataset_v2")
    imgs = []
    for sp in ("train", "val"):
        d = os.path.join(src, "images", sp)
        if os.path.isdir(d):
            imgs += [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT)]
    print("  图片总数: %d" % len(imgs))

    # 标注统计
    boxes = []
    per_img = []
    bad_imgs = 0
    for p in imgs:
        lp = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        lp = os.path.splitext(lp)[0] + ".txt"
        if not os.path.exists(lp):
            bad_imgs += 1
            per_img.append(0)
            continue
        lines = [l.split() for l in open(lp, encoding="utf-8").read().strip().splitlines() if len(l.split()) >= 5]
        per_img.append(len(lines))
        for q in lines:
            cx, cy, w, h = (float(x) for x in q[1:5])
            boxes.append((w, h, w * h, cx, cy))
    print("  无标注文件: %d 张" % bad_imgs)
    print("  缺陷框总数: %d | 每图平均 %.2f (中位 %d, 最多 %d)"
          % (len(boxes), len(boxes) / max(1, len(imgs)), statistics.median(per_img), max(per_img)))
    areas = [b[2] for b in boxes]
    print("  框面积占画面: 中位 %.3f%% | 均值 %.3f%% | 最小 %.5f%% | 最大 %.2f%%"
          % (100 * statistics.median(areas), 100 * sum(areas) / len(areas),
             100 * min(areas), 100 * max(areas)))
    print("  小目标(<1%%画面): %d 个 (%.0f%%)"
          % (sum(1 for a in areas if a < 0.01), 100 * sum(1 for a in areas if a < 0.01) / len(areas)))

    # 可疑标注
    huge = sum(1 for b in boxes if b[2] > 0.5)
    tiny = sum(1 for b in boxes if b[2] < 0.0002)
    extreme = sum(1 for b in boxes if max(b[0] / max(b[1], 1e-9), b[1] / max(b[0], 1e-9)) > 8)
    oob = sum(1 for b in boxes if b[3] - b[0] / 2 < -0.001 or b[3] + b[0] / 2 > 1.001
              or b[4] - b[1] / 2 < -0.001 or b[4] + b[1] / 2 > 1.001)
    print("  可疑框: 面积>50%% %d 个 | 极小(<0.02%%) %d 个 | 长宽比>8 %d 个 | 越界 %d 个"
          % (huge, tiny, extreme, oob))

    print()
    print("  计算同源重复度 ...")
    hs, groups = cluster(imgs)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    covered = sum(sizes)
    print("  独立来源: %d 个 (压缩率 %.0f%%)" % (len(groups), 100 * (1 - len(groups) / covered)))
    print("  来源规模前10: %s" % sizes[:10])
    print("  最大来源占全量: %.0f%%" % (100 * sizes[0] / covered))
    print("  >50 张的来源数: %d 个，合计 %d 张 (%.0f%%)"
          % (sum(1 for s in sizes if s > 50), sum(s for s in sizes if s > 50),
             100 * sum(s for s in sizes if s > 50) / covered))


if __name__ == "__main__":
    audit_classification()
    audit_defect()
