# -*- coding: utf-8 -*-
"""实验汇总服务测试"""
from services.experiment_service import summarize_category, build_result, _safe_auc


def _row(preds=None, y_true="good", y3="premium", pipeline3=None, scores=None, latency=None):
    r = {"y_true": y_true, "y3": y3, "pipeline3": pipeline3,
         "scores": scores or {}, "latency": latency or {}}
    for m in ("svm", "rf", "dl", "yolo"):
        r[m] = None
    for k, v in (preds or {}).items():
        r[k] = v
    return r


def test_safe_auc_binary():
    yt = ["good"] * 4 + ["bad"] * 4
    sc = [0.9, 0.8, 0.7, 0.6, 0.2, 0.3, 0.1, 0.4]
    auc, roc = _safe_auc(yt, sc)
    assert auc is not None and auc > 0.5
    assert roc and len(roc["fpr"]) == len(roc["tpr"])


def test_safe_auc_single_class():
    auc, roc = _safe_auc(["good"] * 5, [0.5] * 5)
    assert auc is None and roc is None


def test_summarize_category_basic():
    rows = [
        _row(preds={"svm": "good", "rf": "good", "dl": "good", "yolo": "good"},
             y_true="good", y3="premium", pipeline3="premium",
             scores={"svm": 0.9, "rf": 0.8, "dl": 0.7, "yolo": 0.6},
             latency={"svm": 10.0, "rf": 20.0, "dl": 30.0, "yolo": 40.0}),
        _row(preds={"svm": "good", "rf": "bad", "dl": "bad", "yolo": "bad"},
             y_true="good", y3="standard", pipeline3="standard",
             scores={"svm": 0.8, "rf": 0.3, "dl": 0.2, "yolo": 0.1},
             latency={"svm": 10.0, "rf": 20.0, "dl": 30.0, "yolo": 40.0}),
        _row(preds={"svm": "bad", "rf": "bad", "dl": "bad", "yolo": "bad"},
             y_true="bad", y3="reject", pipeline3="reject",
             scores={"svm": 0.2, "rf": 0.1, "dl": 0.3, "yolo": 0.2},
             latency={"svm": 10.0, "rf": 20.0, "dl": 30.0, "yolo": 40.0}),
        _row(preds={"svm": "good", "rf": "bad", "dl": "bad", "yolo": "good"},
             y_true="bad", y3="reject", pipeline3="reject",
             scores={"svm": 0.1, "rf": 0.2, "dl": 0.4, "yolo": 0.7},
             latency={"svm": 10.0, "rf": 20.0, "dl": 30.0, "yolo": 40.0}),
    ]
    cat = summarize_category(rows)
    assert cat["n"] == 4
    assert cat["grade_accuracy"] == 100.0
    names = [m["name"] for m in cat["models"]]
    assert names == ["svm", "rf", "dl", "yolo"]
    svm = next(m for m in cat["models"] if m["name"] == "svm")
    assert svm["accuracy"] == 75.0
    assert svm["cm"] == [[2, 0], [1, 1]]
    assert svm["auc"] is not None
    assert svm["latency_ms"] == 10.0
    assert cat["pipeline_cm3"] and cat["pipeline_cm3"]["cm"]
    assert len(cat["mcnemar"]) == 4


def test_summarize_category_no_models():
    """面包/饮料：只有管线结果"""
    rows = [
        _row(preds={}, y_true="good", y3="standard", pipeline3="standard"),
        _row(preds={}, y_true="bad", y3="reject", pipeline3="reject"),
    ]
    cat = summarize_category(rows)
    assert cat["n"] == 2
    assert cat["models"] == []
    assert cat["grade_accuracy"] == 100.0
    assert cat["pipeline_cm3"]["cm"]


def test_build_result():
    cat = summarize_category([
        _row(preds={"svm": "good", "rf": "bad"}, y_true="good", y3="premium",
             pipeline3="premium", scores={"svm": 0.9, "rf": 0.1},
             latency={"svm": 5.0, "rf": 6.0}),
    ])
    res = build_result({"apple": cat})
    assert res["models"] == ["svm", "rf", "pipeline"]
    assert res["accuracy"]["apple"]["svm"] == 100.0
    assert res["accuracy"]["apple"]["pipeline"] == 100.0
    assert res["details"]["apple"]["n"] == 1