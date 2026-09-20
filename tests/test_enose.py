# -*- coding: utf-8 -*-
"""????????S-G ???? / 60 ??? / ????"""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "enose"))
import preprocess as PP
import predict_freshness as PF

HEADER = ["ts_ms", "MQ1", "MQ2", "MQ3", "MQ4", "tempC", "humPct",
          "pressPa", "gasOhm", "eco2", "tvoc", "label"]


def _mock_csv(path, n=40, label="fresh"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i in range(n):
            w.writerow([i * 1000, 100 + i * 0.1, 200 + i * 0.05, 150, 80,
                        25, 60, 101325, 1000, 400, 300, label])


def test_savgol_filter_returns_df(tmp_path):
    """???savgol_filter_df ???? df????? None ?? 500?"""
    p = tmp_path / "s.csv"
    _mock_csv(str(p))
    df = PP.savgol_filter_df(_read(p))
    assert df is not None
    assert "MQ1" in df.columns


def _read(p):
    import pandas as pd
    return pd.read_csv(p, encoding="utf-8")


def test_savgol_accepts_df_object(tmp_path):
    p = tmp_path / "s2.csv"
    _mock_csv(str(p))
    df = _read(p)
    out = PP.savgol_filter_df(df)
    assert out is df  # ???????????
    assert np.isfinite(df["MQ1"].values).all()


def test_extract_single_60_dim(tmp_path):
    p = tmp_path / "s3.csv"
    _mock_csv(str(p))
    feats = PF.extract_single(str(p))
    assert feats.shape == (60,)
    assert np.isfinite(feats).all()


def test_preprocess_loads_samples(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _mock_csv(str(raw / "s001_fresh_day0.csv"))
    samples = PP.load_samples(str(raw))
    assert len(samples) == 1
    name, label, df = samples[0]
    assert label == "fresh"
    assert len(df) == 40


def test_enose_predict_route(tmp_path, monkeypatch):
    """/api/enose/predict ?? CSV ??? 60 ??????"""
    import config as app_config
    monkeypatch.setattr(app_config, "ROOT_DIR", tmp_path)
    import app as appmod
    client = appmod.app.test_client()
    p = tmp_path / "up.csv"
    _mock_csv(str(p))
    with open(p, "rb") as f:
        data = {"file": (f, "up.csv")}
        r = client.post("/api/enose/predict", data=data,
                        content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["feature_dim"] == 60
    assert len(body["curves"]) == 10
    assert len(body["channels"]) == 10
