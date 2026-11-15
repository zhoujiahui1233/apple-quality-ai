# -*- coding: utf-8 -*-
"""重建「干净」的缺陷检测数据集（YOLO 格式）。

清洗规则
  1. 剔除整图级标注：面积 > 50% 的框视为「整果不合格」而非缺陷定位，删除；
  2. 剔除退化框：面积 < 0.02%；剔除单图框数 > 50 的异常图（自动标注噪声）；
  3. 删框后没有剩余框的图片整张剔除（它们只有口径错误的大框，留作负样本会误导）；
  4. 按 dHash 聚类识别同源帧，每个来源最多保留 N 张（默认 8），避免单段视频主导；
  5. 按「簇内目标尺寸中位数」分层切分，使 train/val 的目标尺寸分布接近；
  6. 全部按簇切分，同源图不跨集合。

用法:
  python scripts/build_defect_dataset_v2clean.py [--max-per-source 8] [--val-ratio 0.15]
"""
import argparse
import csv
import os
import random
import shutil
import statistics

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "defect_dataset_v2")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

BIG_AREA = 0.50        # 超过视为整图级标注
TINY_AREA = 0.0002     # 小于视为退化框
MAX_BOXES = 50         # 单图框数上限


def dhash(p):
    try:
        g = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None
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


def label_path(img):
    return os.path.splitext(img.replace(os.sep + "images" + os.sep,
                                        os.sep + "labels" + os.sep))[0] + ".txt"


def read_boxes(lp):
    out = []
    if not os.path.exists(lp):
        return out
    for line in open(lp, encoding="utf-8"):
        q = line.split()
        if len(q) < 5:
            continue
        out.append((int(float(q[0])), float(q[1]), float(q[2]),
                    float(q[3]), float(q[4])))
    return out


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
    ap.add_argument("--max-per-source", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--out", default=os.path.join(ROOT, "defect_dataset_v2clean"))
    args = ap.parse_args()

    # ---- 1. 收集 + 框级清洗 ----
    items = []   # dict(path, label_path, boxes[list])
    stat = {"total": 0, "only_big": 0, "too_many": 0, "boxes_all": 0, "boxes_keep": 0}
    for sp in ("train", "val"):
        d = os.path.join(SRC, "images", sp)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.lower().endswith(IMG_EXT):
                continue
            p = os.path.join(d, f)
            lp = label_path(p)
            boxes = read_boxes(lp)
            stat["total"] += 1
            stat["boxes_all"] += len(boxes)
            if len(boxes) > MAX_BOXES:
                stat["too_many"] += 1
                continue
            kept = []
            for b in boxes:
                a = b[3] * b[4]
                if TINY_AREA <= a <= BIG_AREA:
                    kept.append(b)
            stat["boxes_keep"] += len(kept)
            if not kept:
                if boxes:
                    stat["only_big"] += 1
                continue
            items.append({"path": p, "lp": lp, "boxes": kept})

    print("原始 %d 张 / %d 框" % (stat["total"], stat["boxes_all"]))
    print("  剔除整图级标注图: %d 张" % stat["only_big"])
    print("  剔除框数异常图:   %d 张" % stat["too_many"])
    print("  保留 %d 张 / %d 框 (删掉 %.0f%% 的框)"
          % (len(items), stat["boxes_keep"],
             100 * (1 - stat["boxes_keep"] / max(1, stat["boxes_all"]))))

    # ---- 2. 同源聚类 ----
    print("聚类 ...")
    hh, keep = [], []
    for i, it in enumerate(items):
        h = dhash(it["path"])
        if h is not None:
            hh.append(h)
            keep.append(i)
    items = [items[i] for i in keep]
    hs = np.array(hh, dtype=np.uint64)
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
        ii, jj = np.where(pc <= 5)
        for a, b in zip(ii.tolist(), jj.tolist()):
            gi = s + a
            if gi != b:
                dsu.union(gi, b)
    groups = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)
    print("  独立来源 %d 个" % len(groups))

    # ---- 3. 每个来源限流 ----
    rng = random.Random(args.seed)
    capped = []
    for root, idxs in groups.items():
        idxs = sorted(idxs)
        # 均匀抽取，避免只取相邻帧
        if len(idxs) > args.max_per_source:
            step = len(idxs) / args.max_per_source
            pick = [idxs[int(k * step)] for k in range(args.max_per_source)]
        else:
            pick = idxs
        capped.append((root, pick))
    n_after = sum(len(v) for _, v in capped)
    print("  限流(每来源<=%d): %d -> %d 张" % (args.max_per_source, n, n_after))

    # ---- 4. 按目标尺寸分层切分 ----
    def cluster_area(idxs):
        a = []
        for i in idxs:
            for b in items[i]["boxes"]:
                a.append(b[3] * b[4])
        return statistics.median(a) if a else 0.0

    enriched = [(root, idxs, cluster_area(idxs)) for root, idxs in capped]
    enriched.sort(key=lambda t: t[2])
    # 按面积分 5 层，每层内部随机
    n_buckets = 5
    buckets = [[] for _ in range(n_buckets)]
    for k, t in enumerate(enriched):
        buckets[min(n_buckets - 1, k * n_buckets // max(1, len(enriched)))].append(t)

    val_roots = set()
    cur = 0
    for b in buckets:
        rng.shuffle(b)
        # 每个尺寸层按比例分摊 val 配额，否则按面积从小到大贪心填充会
        # 把配额全用在小目标层，导致 val 与 train 尺寸分布严重不匹配。
        bucket_n = sum(len(idxs) for _, idxs, _ in b)
        target_b = int(round(bucket_n * args.val_ratio))
        cur_b = 0
        for root, idxs, _ in b:
            if cur_b + len(idxs) <= target_b:
                val_roots.add(root)
                cur_b += len(idxs)
        cur += cur_b
    print("  分层切分: val %d 张 / train %d 张" % (cur, n_after - cur))

    # ---- 5. 落盘 ----
    out = args.out
    if os.path.isdir(out):
        shutil.rmtree(out)
    rows = []
    stat2 = {"train": 0, "val": 0, "train_boxes": 0, "val_boxes": 0}
    areas = {"train": [], "val": []}
    for root, idxs, _ in enriched:
        split = "val" if root in val_roots else "train"
        for i in idxs:
            it = items[i]
            fname = os.path.basename(it["path"])
            link_or_copy(it["path"], os.path.join(out, "images", split, fname))
            dst_lp = os.path.join(out, "labels", split, os.path.splitext(fname)[0] + ".txt")
            os.makedirs(os.path.dirname(dst_lp), exist_ok=True)
            with open(dst_lp, "w", encoding="utf-8") as f:
                for c, cx, cy, w, h in it["boxes"]:
                    f.write("%d %.6f %.6f %.6f %.6f\n" % (c, cx, cy, w, h))
            stat2[split] += 1
            stat2[split + "_boxes"] += len(it["boxes"])
            areas[split] += [b[3] * b[4] for b in it["boxes"]]
            rows.append([fname, split, root, len(it["boxes"]),
                         round(statistics.median([b[3] * b[4] for b in it["boxes"]]), 6),
                         os.path.relpath(it["path"], ROOT)])

    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\nnc: 1\nnames: [\"defect\"]\n"
                % out.replace("\\", "/"))
    with open(os.path.join(out, "manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filename", "split", "source", "n_boxes", "median_area", "origin"])
        w.writerows(rows)

    def pct(x):
        return "%.2f%%" % (100 * x) if x else "n/a"

    report = []
    report.append("# 缺陷检测数据集（clean v2）\n")
    report.append("## 规模\n")
    report.append("| split | 图片 | 标注框 | 框面积中位数 |")
    report.append("| --- | --- | --- | --- |")
    for s in ("train", "val"):
        report.append("| %s | %d | %d | %s |"
                      % (s, stat2[s], stat2[s + "_boxes"],
                         pct(statistics.median(areas[s])) if areas[s] else "n/a"))
    report.append("\n## 清洗动作\n")
    report.append("- 原始: %d 张 / %d 框" % (stat["total"], stat["boxes_all"]))
    report.append("- 剔除「整图级」大框图: %d 张" % stat["only_big"])
    report.append("- 剔除框数>%d 的异常图: %d 张" % (MAX_BOXES, stat["too_many"]))
    report.append("- 删除面积>50%% 或 <0.02%% 的框后保留: %d 框" % stat["boxes_keep"])
    report.append("- 同源聚类: %d 个来源；每来源最多 %d 张" % (len(groups), args.max_per_source))
    report.append("- 切分: 按来源隔离 + 按目标尺寸 5 层分层；同源图不跨集合")
    with open(os.path.join(out, "dataset_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print("\n输出:", out)
    print("  train %d 张 / %d 框 | val %d 张 / %d 框"
          % (stat2["train"], stat2["train_boxes"], stat2["val"], stat2["val_boxes"]))
    print("  data.yaml / manifest.csv / dataset_report.md 已写入")


if __name__ == "__main__":
    main()
