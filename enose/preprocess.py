# -*- coding: utf-8 -*-
"""
电子鼻数据预处理与特征提取

输入：data/raw 下每个 CSV = 一个样本的一次采集（collect.py 生成）
流程：读数据 -> Savitzky-Golay 滤波 -> 逐通道特征提取 -> 汇总特征表

用法：
    python preprocess.py data/raw --out data/features.csv
    python preprocess.py data/raw --out data/features.csv --baseline data/raw/baseline_day1.csv

说明：
    - label 从 CSV 的 label 列读取（collect.py 每行写入），
      或从文件名自动推断（s001_fresh_day3.csv -> fresh）。
    - --baseline 传入空罐基线 CSV，会按通道做差分校正（样品值 - 基线均值）。
    - 每通道提取 6 个特征：start(起始基线) steady(稳态均值) range(极差)
      slope1(前5点斜率) auc(响应面积) ratio(稳态/起始)，共 10 通道 60 维。
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

CHANNELS = ["MQ1", "MQ2", "MQ3", "MQ4", "tempC", "humPct",
            "pressPa", "gasOhm", "eco2", "tvoc"]
WINDOW = 11      # S-G 滤波窗口（奇数）
POLY = 3         # S-G 多项式阶数


def infer_label_from_name(name):
    """s001_fresh_day3.csv -> fresh；找不到返回 unknown"""
    base = os.path.basename(name).lower()
    for kw in ("rotten", "mid", "fresh", "baseline"):
        if kw in base:
            return kw
    return "unknown"


def load_samples(raw_dir):
    """返回 [(name, label, df), ...]"""
    samples = []
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        sys.exit(f"目录下没有 CSV 文件: {raw_dir}")
    for p in paths:
        df = pd.read_csv(p, encoding="utf-8")
        if "label" in df.columns and df["label"].notna().any():
            label = str(df["label"].dropna().iloc[0]).strip() or infer_label_from_name(p)
        else:
            label = infer_label_from_name(p)
        samples.append((os.path.basename(p), label, df))
    return samples


def savgol_filter_df(df):
    """对每个传感器通道做 S-G 滤波（原地替换），非数值/占位值保持原样"""
    from scipy.signal import savgol_filter
    for ch in CHANNELS:
        if ch not in df.columns:
            continue
        v = pd.to_numeric(df[ch], errors="coerce").values.astype(float)
        v = np.where(np.isfinite(v), v, np.nan)
        if np.isnan(v).any():
            # 用线性插值补齐少量缺失
            n = np.arange(len(v))
            mask = np.isfinite(v)
            if mask.sum() >= 3:
                v = np.interp(n, n[mask], v[mask])
        df[ch] = savgol_filter(v, WINDOW, POLY)
    return df


def extract_features(df):
    """对单个样本提取每通道 6 特征，返回 dict"""
    feats = {}
    for ch in CHANNELS:
        if ch not in df.columns:
            continue
        v = pd.to_numeric(df[ch], errors="coerce").values.astype(float)
        v = v[np.isfinite(v)]
        if len(v) < 15:
            continue
        start = float(np.mean(v[:5]))           # 起始基线
        steady = float(np.mean(v[int(len(v) * 0.7):]))  # 稳态均值（后 30%）
        rng = float(np.max(v) - np.min(v))      # 极差
        slope1 = float((v[5] - v[0]) / 5)       # 前 5 点斜率（对齐论文式）
        _trapz = getattr(np, "trapezoid", np.trapz)  # numpy<2.0 compat
        auc = float(_trapz(v - start))  # AUC of response curve
        ratio = float(steady / start) if start != 0 else 0.0
        feats[f"{ch}_start"] = start
        feats[f"{ch}_steady"] = steady
        feats[f"{ch}_range"] = rng
        feats[f"{ch}_slope1"] = slope1
        feats[f"{ch}_auc"] = auc
        feats[f"{ch}_ratio"] = ratio
    return feats


def load_baseline(path):
    """空罐基线：返回每个通道均值"""
    df = pd.read_csv(path, encoding="utf-8")
    base = {}
    for ch in CHANNELS:
        if ch in df.columns:
            base[ch] = float(pd.to_numeric(df[ch], errors="coerce").mean())
    return base


def main():
    ap = argparse.ArgumentParser(description="电子鼻预处理与特征提取")
    ap.add_argument("raw_dir", help="原始数据目录（每文件一个样本）")
    ap.add_argument("--out", default="data/features.csv", help="输出特征 CSV")
    ap.add_argument("--baseline", default="", help="空罐基线 CSV（可选，做差分校正）")
    args = ap.parse_args()

    base = load_baseline(args.baseline) if args.baseline else None
    samples = load_samples(args.raw_dir)

    rows = []
    for name, label, df in samples:
        savgol_filter_df(df)
        feats = extract_features(df)
        if not feats:
            print(f"[skip] {name}: 可用通道不足", file=sys.stderr)
            continue
        if base:
            for ch, b in base.items():
                key = f"{ch}_start"
                if key in feats:
                    feats[key] -= b  # 对起始基线做差分，近似"样品-空罐"
        feats["label"] = label
        feats["source"] = name
        rows.append(feats)

    if not rows:
        sys.exit("没有可用的样本特征")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"完成：{len(rows)} 个样本 -> {args.out}")
    print("标签分布：")
    print(out["label"].value_counts().to_string())
    print(f"特征维度：{out.shape[1] - 2}（不含 label/source）")


if __name__ == "__main__":
    main()
