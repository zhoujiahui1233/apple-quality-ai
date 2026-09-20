# -*- coding: utf-8 -*-
"""从所有缺陷相关图片池里，挑出「干净、不重复」的候选集，供人工精标。

做法：
  1. 汇总所有缺陷相关文件夹的图（跨文件夹大量重复）；
  2. dHash 全量两两聚类（距离 <= thresh 视为同源），把重复/近重复并簇；
  3. 每簇只保留 1 张代表图 —— 按「清晰度(拉普拉斯方差) x 分辨率」选最佳；
  4. 输出到 defect_clean_pool/images/ + manifest.csv + 可视化的 gallery.html。

结果：把 N 千张高度重复的图，压成"每个来源一张"的干净候选池。

用法:
  python scripts/build_defect_clean_pool.py [--thresh 5] [--out DIR]
"""
import argparse
import csv
import os
import sys
import html
import shutil

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

POOLS = [
    ("defect_dataset_v2", os.path.join(ROOT, "defect_dataset_v2", "images")),
    ("defect_dataset_yolo", os.path.join(ROOT, "defect_dataset_yolo", "images")),
    ("defect_annotate", os.path.join(ROOT, "defect_annotate", "images")),
    ("defect_label", os.path.join(ROOT, "defect_label", "images")),
    ("missed", os.path.join(ROOT, "漏检样本")),
    ("teacher", os.path.join(ROOT, "teacher_dataset")),
    ("vlm_seed", os.path.join(ROOT, "vlm_seed_400")),
    ("collected_bad", os.path.join(ROOT, "dataset_collected", "apple", "不合格")),
]


def imread_u(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def dhash_and_sharp(path):
    g = imread_u(path)
    if g is None:
        return None, 0.0, (0, 0)
    h, w = g.shape[:2]
    small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    gs = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA) if max(h, w) > 256 else g
    sharp = float(cv2.Laplacian(gs, cv2.CV_64F).var())
    return v, sharp, (w, h)


def label_exists(img_path):
    base = os.path.splitext(img_path)[0] + ".txt"
    for a, b in (("images", "labels"), ("images", "lables")):
        if os.sep + a + os.sep in img_path:
            p = img_path.replace(os.sep + a + os.sep, os.sep + b + os.sep)
            p = os.path.splitext(p)[0] + ".txt"
            if os.path.exists(p) and os.path.getsize(p) > 0:
                return True
    return os.path.exists(base) and os.path.getsize(base) > 0


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "defect_clean_pool"))
    args = ap.parse_args()

    items = []   # (path, pool_name)
    for name, d in POOLS:
        if not os.path.isdir(d):
            print("跳过(不存在):", name)
            continue
        n0 = len(items)
        for root, _, fs in os.walk(d):
            for f in fs:
                if f.lower().endswith(IMG_EXT):
                    items.append((os.path.join(root, f), name))
        print("%-18s +%d" % (name, len(items) - n0))
    print("合计图片:", len(items))

    print("计算 dHash + 清晰度 ...")
    hashes, sharp, size, keep = [], [], [], []
    for i, (p, _) in enumerate(items):
        h, s, wh = dhash_and_sharp(p)
        if h is not None:
            hashes.append(h); sharp.append(s); size.append(wh); keep.append(i)
        if (i + 1) % 3000 == 0:
            print("  ", i + 1, "/", len(items), flush=True)
    items = [items[i] for i in keep]
    hashes = np.array(hashes, dtype=np.uint64)
    print("有效:", len(hashes))

    print("聚类 (thresh=%d) ..." % args.thresh)
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
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print("独立来源 %d 个 (最大来源 %d 张, 前10: %s)"
          % (len(groups), sizes[0], sizes[:10]))
    print("压缩率: %d -> %d 张 (去掉 %.0f%%)"
          % (n, len(groups), 100 * (1 - len(groups) / max(1, n))))

    out = args.out
    imgdir = os.path.join(out, "images")
    thdir = os.path.join(out, "thumbs")
    for d in (imgdir, thdir):
        os.makedirs(d, exist_ok=True)

    rows = []
    print("导出代表图 ...")
    for ci, (r, idxs) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1])), 1):
        # 代表图 = 清晰度 x 分辨率 最高者
        best = max(idxs, key=lambda i: sharp[i] * (1 + size[i][0] * size[i][1] / 1e6))
        src = items[best][0]
        w, h = size[best]
        fname = "c%05d.jpg" % ci
        dst = os.path.join(imgdir, fname)
        try:
            im = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                continue
            if max(im.shape[:2]) > 1600:
                sc = 1600.0 / max(im.shape[:2])
                im = cv2.resize(im, (int(im.shape[1] * sc), int(im.shape[0] * sc)),
                                interpolation=cv2.INTER_AREA)
            cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(dst)
            th = cv2.resize(im, (220, int(220 * im.shape[0] / im.shape[1])))
            cv2.imencode(".jpg", th, [int(cv2.IMWRITE_JPEG_QUALITY), 80])[1].tofile(
                os.path.join(thdir, "c%05d.jpg" % ci))
        except Exception:
            continue
        pools = sorted({items[i][1] for i in idxs})
        has_lab = any(label_exists(items[i][0]) for i in idxs)
        rows.append([fname, ci, len(idxs), "|".join(pools), int(has_lab),
                     round(sharp[best], 1), "%dx%d" % (w, h),
                     os.path.relpath(src, ROOT)])

    with open(os.path.join(out, "manifest.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filename", "cluster", "n_merged", "pools", "has_label",
                    "sharpness", "size", "origin"])
        w.writerows(rows)

    # 简单可视化画廊
    cards = []
    for r in rows:
        tag = "有标注" if r[4] else "无标注"
        cards.append(
            '<figure><img src="thumbs/%s" loading="lazy"><figcaption>%s<br>'
            '<small>%s · 合并%d张 · %s</small></figcaption></figure>'
            % (r[0], r[0], tag, r[2], html.escape(r[3])))
    with open(os.path.join(out, "gallery.html"), "w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'><title>缺陷候选池</title>"
                "<style>body{background:#111;color:#ddd;font:13px system-ui;margin:16px}"
                ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px}"
                "figure{margin:0;background:#1b1b1b;border-radius:8px;overflow:hidden}"
                "img{width:100%%;display:block}figcaption{padding:6px 8px}"
                "small{color:#888}</style>"
                "<h2>缺陷候选池（每个来源只留 1 张）</h2><p>共 %d 张</p>"
                "<div class='grid'>%s</div>" % (len(rows), "".join(cards)))

    print("\n输出: %s" % out)
    print("  images/  %d 张干净代表图" % len(rows))
    print("  manifest.csv / gallery.html（浏览器打开可逐张浏览挑选）")


if __name__ == "__main__":
    main()
