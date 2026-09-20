# -*- coding: utf-8 -*-
"""
把 record.xlsx 的「时间点(min)」合并进 features.csv 的 minutes 列（回归目标）。
用法：
    python enose/merge_minutes.py
    python enose/merge_minutes.py --features enose/data/features.csv --record enose/data/record.xlsx
"""
import argparse
import os
import sys

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="合并回归目标 minutes 到特征表")
    ap.add_argument("--features", default=os.path.join("enose", "data", "features.csv"))
    ap.add_argument("--record", default=os.path.join("enose", "data", "record.xlsx"))
    ap.add_argument("--time-col", default="时间点(min)")
    args = ap.parse_args()

    f = pd.read_csv(args.features, encoding="utf-8")
    r = pd.read_excel(args.record)
    if "source" not in f.columns:
        sys.exit("features.csv 缺少 source 列，请先运行 preprocess.py")
    if "样本编号" not in r.columns:
        sys.exit(f"record.xlsx 缺少「样本编号」列，现有列：{list(r.columns)}")

    f["_src"] = f["source"].str.replace(r"\.csv$", "", regex=True)
    r["_src"] = r["样本编号"].astype(str).str.strip()
    merged = f.merge(r[["_src", args.time_col]].rename(columns={args.time_col: "minutes"}),
                     on="_src", how="left")
    merged = merged.drop(columns=["_src"])
    n_missing = int(merged["minutes"].isna().sum())
    if n_missing:
        print(f"[warn] {n_missing} 个样本在记录表中找不到时间点，minutes 为空")
    merged.to_csv(args.features, index=False, encoding="utf-8")
    print(f"完成：minutes 已并入 {args.features}（缺失 {n_missing} 个）")


if __name__ == "__main__":
    main()