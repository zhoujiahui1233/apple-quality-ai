import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def mcnemar_test(y_pred1, y_pred2, y_true):
    """McNemar 检验: 两模型预测差异显著性"""
    e1 = (y_pred1 != y_true)
    e2 = (y_pred2 != y_true)
    b = np.sum(e1 & ~e2)   # 模型1错、模型2对
    c = np.sum(~e1 & e2)   # 模型1对、模型2错
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p)


def kappa_score(y1, y2):
    """Cohen Kappa 一致性"""
    return float(cohen_kappa_score(y1, y2)), None


def confusion_matrix_stats(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = np.asarray(cm)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    acc = tp.sum() / max(cm.sum(), 1)
    return {"accuracy": float(acc),
            "precision": [float(x) for x in precision],
            "recall": [float(x) for x in recall],
            "f1": [float(x) for x in f1],
            "cm": cm.tolist()}


def model_comparison_table(predictions):
    """多模型对比表:
    predictions = [{"name": "svm", "y_pred": [...], "y_true": [...]}, ...]
    返回各模型指标 + 两两 McNemar p 值
    """
    models = []
    for p in predictions:
        y_true = np.asarray(p["y_true"])
        y_pred = np.asarray(p["y_pred"])
        s = confusion_matrix_stats(y_true, y_pred)
        models.append({"name": p["name"], **s})

    # 两两 McNemar
    n = len(models)
    mcnemar_matrix = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            _, pval = mcnemar_test(
                np.asarray(predictions[i]["y_pred"]),
                np.asarray(predictions[j]["y_pred"]),
                np.asarray(predictions[i]["y_true"])
            )
            mcnemar_matrix[i][j] = pval
    return {"models": models, "mcnemar": mcnemar_matrix}
