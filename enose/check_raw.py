# -*- coding: utf-8 -*-
"""
数据完整性检查：核对 raw 目录 CSV ↔ record.xlsx ↔ 照片命名，输出检查报告。
用法：
    python enose/check_raw.py
    python enose/check_raw.py --raw enose/data/raw --record enose/data/record.xlsx --photos enose/data/photos
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd

SAMPLE_RE = re.compile(r"^s\d+_slice_\d+min_[a-z]\d+\.csv$")
BASELINE_RE = re.compile(r"^baseline_day\d+\.csv$")
VALID_LABELS = {"fresh", "mid", "rotten"}


def main():
    ap = argparse.ArgumentParser(description="核对电子鼻实验数据完整性")
    ap.add_argument("--raw", default=os.path.join("enose", "data", "raw"))
    ap.add_argument("--record", default=os.path.join("enose", "data", "record.xlsx"))
    ap.add_argument("--photos", default=os.path.join("enose", "data", "photos"))
    args = ap.parse_args()

    issues = []
    csvs = [os.path.basename(p) for p in glob.glob(os.path.join(args.raw, "*.csv"))]
    samples = [c for c in csvs if not c.startswith("baseline")]
    baselines = [c for c in csvs if c.startswith("baseline")]

    # 1. 命名规范
    for c in samples:
        if not SAMPLE_RE.match(c):
            issues.append(f"[命名] 不符合规范（应为 s001_slice_0min_a1.csv）：{c}")
    for c in baselines:
        if not BASELINE_RE.match(c):
            issues.append(f"[命名] 基线命名应为 baseline_day1.csv：{c}")

    # 2. 记录表
    rec = pd.read_excel(args.record, sheet_name=0)
    rec_ids = set(rec["样本编号"].dropna().astype(str).str.strip())
    rec_ids = {x for x in rec_ids if x and x != "s001_slice_0min_a1"}
    csv_ids = {c[:-4] for c in samples}

    for c in sorted(csv_ids - rec_ids):
        issues.append(f"[缺失] CSV 在记录表中无对应行：{c}")
    for c in sorted(rec_ids - csv_ids):
        issues.append(f"[缺失] 记录表有行但 raw 目录无 CSV：{c}")

    # 3. label 合法性（读每个 CSV 的 label 列）
    for c in samples:
        try:
            df = pd.read_csv(os.path.join(args.raw, c), encoding="utf-8")
        except Exception as e:
            issues.append(f"[读取失败] {c}: {e}")
            continue
        if "label" in df.columns:
            lab = str(df["label"].dropna().iloc[0]).strip() if df["label"].notna().any() else ""
            if lab not in VALID_LABELS:
                issues.append(f"[标签] {c} 的 label 不是 fresh/mid/rotten：{lab!r}")
        if len(df) < 60:
            issues.append(f"[时长] {c} 采集行数不足（{len(df)} 行，应 ≥ 60）")
        for ch in ["MQ1", "MQ2", "MQ3", "MQ4", "tempC"]:
            if ch in df.columns and (pd.to_numeric(df[ch], errors="coerce") == -999).any():
                issues.append(f"[异常] {c} 含 -999 占位（{ch}）")
                break

    # 4. 照片核对
    photos = set()
    if os.path.isdir(args.photos):
        photos = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(args.photos, "*"))}
    for c in sorted(csv_ids - photos):
        issues.append(f"[照片] 缺少配对照片：{c}")

    # 报告
    print("=" * 50)
    print(f"CSV 样本数：{len(samples)}　基线数：{len(baselines)}")
    print(f"记录表行数（不含示例）：{len(rec_ids)}　照片数：{len(photos)}")
    print(f"发现问题：{len(issues)} 个")
    for i in issues:
        print("  -", i)
    if not issues:
        print("[OK] 全部通过：数据完整，可以进入特征提取。")
    print("=" * 50)
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()