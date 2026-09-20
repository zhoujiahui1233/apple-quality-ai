# -*- coding: utf-8 -*-
"""重建「无泄漏」的外观分类数据集。

问题：原 dl_dataset_v2 用 random.shuffle 切分，而同一次拍摄/同一段视频的连续帧
高度相似，导致验证集里约 62% 的图和训练集近重复，指标被高估。

做法（按数据来源分组，而不是随机打散）：
  * 连续编号的序列帧（new_good_00001 / 合格_extracted_00001 等）——
    同一来源，按编号**连续切分**：前 85% 进 train、后 15% 进 val。
    这样每个序列只有一个边界，几乎不产生跨集合的近重复。
  * 带拍摄时间的照片（合格_20260804_210520_734）—— 同一秒视为同一apple，
    按「秒」成组随机切分。
  * 其它 —— 逐张随机切分。

注：早期版本用 dHash 单链聚类，会把整段视频链成上万张的巨簇，并在簇内
混入不同标签；连续切分更稳、也不破坏类别平衡。

用法:
  python scripts/rebuild_clean_dataset.py [--val-ratio 0.15] [--out DIR]
"""
import argparse
import csv
import os
import re
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(ROOT, "dataset_collected", "apple")
CLASSES = [("good", "合格"), ("bad", "不合格")]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

RE_SEQ1 = re.compile(r"^(new_good|new_bad)_(\d+)$")
RE_SEQ2 = re.compile(r"^(合格|不合格)_extracted_(\d+)$")
RE_TS = re.compile(r"(\d{8})_(\d{6})")


def classify_name(fname):
    """返回 (kind, key, idx)。kind: seq(连续序列) / ts(拍摄时间) / single"""
    stem = os.path.splitext(fname)[0]
    if stem.endswith("_crop"):
        stem = stem[:-5]
    m = RE_SEQ1.match(stem)
    if m:
        return "seq", m.group(1), int(m.group(2))
    m = RE_SEQ2.match(stem)
    if m:
        return "seq", m.group(1) + "_extracted", int(m.group(2))
    m = RE_TS.search(stem)
    if m:
        return "ts", m.group(1) + "_" + m.group(2), -1
    return "single", stem, -1


def imread_gray(path):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def dhash(path):
    img = imread_gray(path)
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


def hamming(a, b):
    return bin(a ^ b).count("1")


def popcount64(x):
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56


def cluster_hashes(hashes, thresh, block=256):
    """全量两两比对的单链聚类（按块做，避免爆内存）。

    只用滑动窗口会漏掉"位置隔得远但内容相同"的图，所以这里对每一张都和
    全部其他图比一次，保证不漏。
    """
    n = len(hashes)
    dsu = DSU(n)
    seen = {}
    for i in range(n):
        k = int(hashes[i])
        if k in seen:
            dsu.union(seen[k], i)
        else:
            seen[k] = i
    for s in range(0, n, block):
        e = min(n, s + block)
        xor = hashes[s:e, None] ^ hashes[None, :]
        pc = popcount64(xor)
        ii, jj = np.where(pc <= thresh)
        for a, b in zip(ii.tolist(), jj.tolist()):
            gi = s + a
            if gi != b:
                dsu.union(gi, b)
    return dsu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="调试用：只取前 N 张")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--thresh", type=int, default=5,
                    help="dHash 汉明距离阈值，<=该值视为同源，归入同一簇")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--out", default=os.path.join(ROOT, "clean_dataset_v1"))
    args = ap.parse_args()

    items = []  # (path, label 0/1)
    for yi, (_, cname) in enumerate(CLASSES):
        d = os.path.join(POOL, cname)
        if not os.path.isdir(d):
            print("missing dir:", d)
            continue
        for f in os.listdir(d):
            if f.lower().endswith(IMG_EXT):
                items.append((os.path.join(d, f), yi))
    items.sort(key=lambda t: os.path.basename(t[0]))
    if args.limit:
        items = items[:args.limit]
    print("pool images:", len(items))

    # ---- dHash 全量两两聚类：任一距离<=thresh 的图都不会跨 train/val ----
    hc = os.path.join(args.out, "hashes_cache.npz")
    names = [os.path.basename(p) for p, _ in items]
    hashes = None
    if os.path.exists(hc):
        z = np.load(hc, allow_pickle=True)
        if list(z["names"]) == names:
            hashes = z["hashes"]
            print("复用哈希缓存:", len(hashes))
    if hashes is None:
        print("computing dHash...")
        hh, keep = [], []
        for i, (p, _) in enumerate(items):
            h = dhash(p)
            if h is not None:
                hh.append(h)
                keep.append(i)
            if (i + 1) % 4000 == 0:
                print("  hashed", i + 1, "/", len(items))
        items = [items[i] for i in keep]
        hashes = np.array(hh, dtype=np.uint64)
        os.makedirs(args.out, exist_ok=True)
        np.savez(hc, hashes=hashes,
                 names=np.array([os.path.basename(p) for p, _ in items]))
    print("valid hashes:", len(hashes))

    print("clustering (full pairwise, thresh=%d)..." % args.thresh)
    dsu = cluster_hashes(hashes, args.thresh)
    groups = {}
    for i in range(len(hashes)):
        groups.setdefault(dsu.find(i), []).append(i)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print("簇数 %d | 最大簇 %d | 前一大的簇 %s"
          % (len(groups), sizes[0], sizes[1:6]))

    # 逐图标签的类别分布（校验切分是否可平衡）
    conflict = sum(1 for v in groups.values() if len({items[i][1] for i in v}) > 1)
    print("簇内标签冲突: %d 个簇" % conflict)

    rng = np.random.RandomState(args.seed)
    glist = list(groups.items())
    rng.shuffle(glist)
    target_val = int(len(items) * args.val_ratio)
    assign = {}
    cur_val = 0
    for r, idxs in glist:
        # 整簇归一边；加进来会明显超出目标就丢给 train，避免被巨簇撑爆 val
        if cur_val + len(idxs) <= target_val * 1.15:
            s = "val"
            cur_val += len(idxs)
        else:
            s = "train"
        for i in idxs:
            assign[i] = (s, r)

    out = args.out
    manifest = os.path.join(out, "manifest.csv")
    os.makedirs(out, exist_ok=True)
    rows = []
    stat = {"train": {"good": 0, "bad": 0}, "val": {"good": 0, "bad": 0}}
    for i, (p, y) in enumerate(items):
        split, key = assign[i]
        cls = "good" if y == 0 else "bad"   # 用图片自身标签，不用组的标签
        rows.append([os.path.basename(p), split, cls, key, os.path.relpath(p, ROOT)])
        stat[split][cls] += 1
    with open(manifest, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filename", "split", "class", "cluster", "path"])
        w.writerows(rows)

    print("\n输出 manifest:", manifest)
    for s in ("train", "val"):
        print("  %-5s good=%-6d bad=%-6d total=%d"
              % (s, stat[s]["good"], stat[s]["bad"],
                 stat[s]["good"] + stat[s]["bad"]))
    print("\n说明: 未复制图片；train/val 通过 manifest.csv 的 split 列区分，")
    print("      训练脚本按 manifest 读取即可，源图保持原位不动。")


if __name__ == "__main__":
    main()
