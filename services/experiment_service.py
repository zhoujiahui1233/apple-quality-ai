# -*- coding: utf-8 -*-
"""实验数据汇总服务：把逐样本预测结果转为显著性分析页面所需的完整结构。

包含：准确率/精确率/召回率/F1、二分类混淆矩阵、三级混淆矩阵、ROC/AUC、
McNemar 检验、模型平均推理耗时。
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from services.stats_service import model_comparison_table

MODEL_NAMES = ["svm", "rf", "dl", "yolo"]


def _safe_auc(y_true, scores):
    """由模型置信分数计算 AUC 与 ROC 曲线点；AUC<0.5 时自动反转分数"""
    y_true = np.asarray(list(y_true))
    scores = np.asarray(list(scores), dtype=float)
    if len(y_true) < 2 or len(set(y_true.tolist())) < 2:
        return None, None
    # 标签转 0/1（good=1），兼容不同版本 sklearn（部分版本不支持 pos_label 参数）
    y_bin = np.where(y_true == "good", 1, 0).astype(int)
    try:
        auc = float(roc_auc_score(y_bin, scores))
        if auc < 0.5:
            scores = 1.0 - scores
            auc = float(roc_auc_score(y_bin, scores))
        fpr, tpr, _ = roc_curve(y_bin, scores)
        return round(auc, 4), {
            "fpr": [round(float(x), 4) for x in fpr],
            "tpr": [round(float(x), 4) for x in tpr],
        }
    except Exception:
        return None, None


def summarize_category(rows):
    """汇总一个品类的逐样本结果。

    rows: list[dict]
      必填: y_true (good/bad), y3 (premium/standard/reject)
      苹果: svm/rf/dl/yolo 预测(good/bad)、scores.{m}、latency.{m}
      面包/饮料: pipeline3 (premium/standard/reject)
    返回该品类完整汇总 dict。
    """
    n = len(rows)
    cat = {"n": n, "models": [], "mcnemar": None,
           "grade_accuracy": None, "pipeline_cm3": None}

    # 管线三级分级准确率（premium/standard -> good）
    g3_ok = [1 for r in rows
             if (r.get("pipeline3") in ("premium", "standard")) == (r.get("y_true") == "good")]
    if g3_ok and n:
        cat["grade_accuracy"] = round(len(g3_ok) / n * 100, 2)

    # 三级混淆矩阵（管线）
    p3 = [(r.get("y3"), r.get("pipeline3")) for r in rows if r.get("pipeline3")]
    if p3:
        labels3 = ["premium", "standard", "reject"]
        try:
            cat["pipeline_cm3"] = {
                "labels": labels3,
                "cm": confusion_matrix([a for a, _ in p3], [b for _, b in p3],
                                       labels=labels3).tolist(),
            }
        except Exception:
            pass

    preds = []
    for m in MODEL_NAMES:
        pairs = [(r.get(m), r.get("y_true")) for r in rows if r.get(m) is not None]
        if pairs:
            preds.append({"name": m,
                          "y_pred": [p[0] for p in pairs],
                          "y_true": [p[1] for p in pairs]})
    if not preds:
        return cat

    table = model_comparison_table(preds)
    model_out = []
    for mm in table["models"]:
        name = mm["name"]
        valid = [r for r in rows if r.get(name) is not None]
        entry = {
            "name": name,
            "accuracy": round(mm["accuracy"] * 100, 2),
            "precision": [round(x, 3) for x in mm["precision"]],
            "recall": [round(x, 3) for x in mm["recall"]],
            "f1": [round(x, 3) for x in mm["f1"]],
            "n": len(valid),
            "auc": None, "roc": None, "latency_ms": None,
            "cm": confusion_matrix([r.get("y_true") for r in valid],
                                   [r.get(name) for r in valid],
                                   labels=["good", "bad"]).tolist(),
            "cm_labels": ["good", "bad"],
        }
        sc_pairs = [( (r.get("scores") or {}).get(name), r.get("y_true") )
                    for r in valid
                    if (r.get("scores") or {}).get(name) is not None]
        if sc_pairs:
            auc, roc = _safe_auc([p[1] for p in sc_pairs], [p[0] for p in sc_pairs])
            entry["auc"] = auc
            entry["roc"] = roc
        lats = [(r.get("latency") or {}).get(name) for r in valid
                if (r.get("latency") or {}).get(name) is not None]
        if lats:
            entry["latency_ms"] = round(float(np.mean(lats)), 2)
        model_out.append(entry)

    cat["models"] = model_out
    cat["mcnemar"] = table["mcnemar"]
    return cat


def build_result(summary_by_cat):
    """把各品类汇总合并为 /api/stats/models 需要的顶层结构"""
    models_union = []
    accuracy = {}
    n_map = {}
    mcnemar = {}
    for cat, cs in summary_by_cat.items():
        n_map[cat] = cs["n"]
        accuracy[cat] = {}
        for m in cs.get("models", []):
            accuracy[cat][m["name"]] = m["accuracy"]
            if m["name"] not in models_union:
                models_union.append(m["name"])
        if cs.get("mcnemar"):
            mcnemar[cat] = cs["mcnemar"]
        if cs.get("grade_accuracy") is not None:
            accuracy[cat]["pipeline"] = cs["grade_accuracy"]
            if "pipeline" not in models_union:
                models_union.append("pipeline")
    return {
        "categories": list(summary_by_cat.keys()),
        "models": models_union,
        "accuracy": accuracy,
        "n": n_map,
        "mcnemar": mcnemar,
        "grade_accuracy": {c: s["grade_accuracy"] for c, s in summary_by_cat.items()},
        "details": summary_by_cat,
    }