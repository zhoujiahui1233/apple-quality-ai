# -*- coding: utf-8 -*-
"""
build_multimodal_table.py - 四觉数据汇总成一张总表

把 multimodal_data 下的视觉/嗅觉/触觉/人工感官数据按 sample_id 对齐，
输出一张宽表供多模态融合与相关性分析使用。

用法：
  python scripts/build_multimodal_table.py
  python scripts/build_multimodal_table.py --root multimodal_data --out output/multimodal_table.csv

输出列：
  sample_id, label, remark,
  visual_file, visual_exists,
  odor_file, odor_feats_json,（若已跑过 preprocess，可并入 features.csv 的 60 维列）
  hardness, weight_g, brix, ph,
  qda_color, qda_odor, qda_taste, qda_texture, qda_overall
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="multimodal_data")
    ap.add_argument("--out", default="output/multimodal_table.csv")
    args = ap.parse_args()

    base = ROOT / args.root
    visual_dir = base / "visual"
    odor_raw = base / "odor" / "raw"
    odor_feat = base / "odor" / "features.csv"
    touch_csv = base / "touch" / "touch.csv"
    qda_csv = base / "qda" / "qda_scores.csv"
    manifest_csv = base / "manifest.csv"

    # 收集全部 sample_id（并集）
    ids = set()
    if visual_dir.exists():
        ids |= {p.stem for p in visual_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    if odor_raw.exists():
        ids |= {p.stem for p in odor_raw.glob("*.csv")}
    for csv in (touch_csv, qda_csv, manifest_csv):
        if csv.exists():
            ids |= set(pd.read_csv(csv, encoding="utf-8")["sample_id"].astype(str))

    rows = []
    odor_feat_df = pd.read_csv(odor_feat, encoding="utf-8") if odor_feat.exists() else None
    for sid in sorted(ids):
        row = {"sample_id": sid, "visual_exists": 0, "odor_exists": 0}
        # 视觉
        for ext in (".jpg", ".jpeg", ".png"):
            vp = visual_dir / f"{sid}{ext}"
            if vp.exists():
                row["visual_exists"] = 1
                row["visual_file"] = str(vp.relative_to(ROOT))
                break
        # 嗅觉：原始曲线存在性 + 特征表合并
        if (odor_raw / f"{sid}.csv").exists():
            row["odor_exists"] = 1
            row["odor_file"] = f"multimodal_data/odor/raw/{sid}.csv"
        if odor_feat_df is not None and sid in odor_feat_df["source"].astype(str).values:
            sub = odor_feat_df[odor_feat_df["source"].astype(str) == sid]
            for c in odor_feat_df.columns:
                if c not in ("source", "label"):
                    row[f"odor_{c}"] = float(sub.iloc[0][c]) if sub.iloc[0][c] is not None else None
        # 触觉/理化（动态读取全部列，新增指标自动包含）
        if touch_csv.exists():
            t = pd.read_csv(touch_csv, encoding="utf-8")
            trow = t[t["sample_id"].astype(str) == sid]
            if len(trow):
                for c in t.columns:
                    if c != "sample_id":
                        row[c] = trow.iloc[0][c]
        # QDA
        if qda_csv.exists():
            q = pd.read_csv(qda_csv, encoding="utf-8")
            qrow = q[q["sample_id"].astype(str) == sid]
            if len(qrow):
                for c in q.columns:
                    if c != "sample_id":
                        row[c] = qrow.iloc[0][c]
        # manifest
        if manifest_csv.exists():
            m = pd.read_csv(manifest_csv, encoding="utf-8")
            mrow = m[m["sample_id"].astype(str) == sid]
            if len(mrow):
                for c in ("label", "remark"):
                    if c in m.columns:
                        row[c] = mrow.iloc[0][c]
        rows.append(row)

    out_df = pd.DataFrame(rows)
    os.makedirs(Path(args.out).parent, exist_ok=True)
    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"总表已写出: {args.out}（{len(out_df)} 个样本）")
    print("模态覆盖情况：")
    for c in ("visual_exists", "odor_exists"):
        if c in out_df.columns:
            print(f"  {c}: {int(out_df[c].sum())}/{len(out_df)}")


if __name__ == "__main__":
    main()
