# -*- coding: utf-8 -*-
"""多模态融合服务测试：数据聚合 / 训练 / 预测"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from services.fusion_service import FusionService, infer_label

HEADER = ["ts_ms", "MQ1", "MQ2", "MQ3", "MQ4", "tempC", "humPct",
          "pressPa", "gasOhm", "eco2", "tvoc", "label"]


def _mock_sample(md, sid, label, color=(30, 30, 200)):
    import cv2
    vdir = md / "visual"; odir = md / "odor" / "raw"
    vdir.mkdir(parents=True, exist_ok=True); odir.mkdir(parents=True, exist_ok=True)
    img = np.zeros((200, 260, 3), dtype=np.uint8)
    cv2.circle(img, (130, 100), 70, color, -1)
    ok, buf = cv2.imencode(".jpg", img)
    buf.tofile(str(vdir / f"{sid}.jpg"))
    with open(odir / f"{sid}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(HEADER)
        for i in range(40):
            w.writerow([i * 1000, 100 + i, 200, 150, 80, 25, 60, 101325, 1000, 400, 300, label])
    touch = md / "touch"; touch.mkdir(parents=True, exist_ok=True)
    csvp = touch / "touch.csv"
    new = not csvp.exists()
    with open(csvp, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["sample_id", "hardness", "brix", "ph"])
        w.writerow([sid, 6.0, 12.0, 3.9])


def test_infer_label_from_name():
    assert infer_label("apple_fresh_001") == "fresh"
    assert infer_label("apple_rotten_01") == "rotten"
    assert infer_label("abc_xyz") is None


def test_collect_and_predict(tmp_path):
    md = tmp_path / "multimodal_data"
    for i in range(4):
        _mock_sample(md, f"fresh_00{i}", "fresh")
        _mock_sample(md, f"rotten_00{i}", "rotten", color=(30, 30, 120))
    svc = FusionService(root=tmp_path)
    samples, phys_df, odor_df, manifest = svc.collect()
    assert len(samples) == 8
    report = svc.train(cv=3)
    assert report["visual"]["n"] >= 8
    assert report["odor"]["n"] >= 8
    assert report["phys"]["n"] >= 8
    # 每类明细 + 混淆矩阵
    assert "per_class" in report["visual"]
    assert "confusion" in report["visual"]
    assert len(report["visual"]["confusion"]) == 2
    assert "fresh" in report["visual"]["per_class"]
    assert "rotten" in report["visual"]["per_class"]
    out = svc.predict("fresh_000")
    assert out["modalities"]["visual"]["available"] is True
    assert out["fusion"] is None or out["fusion"]["prediction"] in ("fresh", "rotten")


def test_predict_missing_data(tmp_path):
    svc = FusionService(root=tmp_path)
    out = svc.predict("no_such")
    assert out["ok"] is True
    for m in out["modalities"].values():
        assert m["available"] is False
