# -*- coding: utf-8 -*-
"""??????????????? / ?? / ????? CSV ??"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import config as app_config
import models


def test_sensory_record_columns():
    cols = {c.name for c in models.SensoryRecord.__table__.columns}
    for c in ("sample_id", "category", "visual_filename", "odor_csv",
              "hardness", "weight_g", "brix", "ph",
              "lab_l", "lab_a", "lab_b", "color_coverage",
              "fruit_diameter_mm", "shape_index",
              "ta", "sugar_acid_ratio", "vitamin_c",
              "tpa_springiness", "tpa_cohesiveness", "tpa_chewiness", "juice_pct",
              "qda_color", "qda_odor", "qda_taste", "qda_texture", "qda_overall",
              "qda_sweet", "qda_sour", "qda_astringent", "qda_aroma",
              "qda_crispness", "qda_juiciness",
              "label", "remark"):
        assert c in cols


def test_sensory_record_unique_sample_id():
    assert models.SensoryRecord.__table__.c.sample_id.unique is True


def test_qda_entry_columns():
    cols = {c.name for c in models.QDAEntry.__table__.columns}
    for c in ("sample_id", "panelist", "qda_color", "qda_odor", "qda_overall",
              "qda_sweet", "qda_crispness", "qda_juiciness"):
        assert c in cols
    # sample_id + panelist 联合唯一
    uniq = [c for c in models.QDAEntry.__table__.constraints
            if c.__class__.__name__ == "UniqueConstraint" and c.columns.keys() == ["sample_id", "panelist"]]
    assert len(uniq) == 1


def test_sensory_routes_registered():
    import app as appmod
    rules = {str(r.rule) for r in appmod.app.url_map.iter_rules()}
    assert "/sensory" in rules
    assert "/api/sensory/list" in rules
    assert "/api/sensory/save" in rules
    assert "/api/sensory/delete" in rules
    assert "/api/sensory/export" in rules
    assert "/enose" in rules
    assert "/api/enose/predict" in rules
    assert "/api/defect_block/save" in rules
    assert "/api/sensory/qda-detail" in rules
    assert "/api/sensory/delete-qda" in rules
    assert "/api/sensory/train" in rules
    assert "/api/sensory/fusion-predict" in rules


def test_defect_block_save_writes_csv(tmp_path, monkeypatch):
    """/api/defect_block/save ??????? multimodal_data/defect_blocks.csv"""
    import config as app_config
    monkeypatch.setattr(app_config, "ROOT_DIR", tmp_path)
    import app as appmod
    client = appmod.app.test_client()
    payload = {"filename": "a.png", "block_index": 1, "type": "??",
               "area": 99.5, "circularity": 0.8, "aspect": 1.1,
               "hsv": [10, 20, 30], "glcm": [0.1, 0.2, 0.3, 0.4], "grade": "???"}
    r = client.post("/api/defect_block/save", json=payload)
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
    csv_path = tmp_path / "multimodal_data" / "defect_blocks.csv"
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8-sig")
    assert "??" in text
    assert "a.png" in text
