import numpy as np
from services.stats_service import mcnemar_test, kappa_score, confusion_matrix_stats


def test_kappa_perfect():
    y1 = np.array([1, 0, 1, 0, 1])
    y2 = np.array([1, 0, 1, 0, 1])
    k, _ = kappa_score(y1, y2)
    assert k == 1.0


def test_mcnemar_significant():
    # 20 个样本: 模型1错 10个，模型2错 2个
    y_true = np.ones(20, dtype=int)
    y1 = np.concatenate([np.zeros(10, dtype=int), np.ones(10, dtype=int)])
    y2 = np.concatenate([np.zeros(2, dtype=int), np.ones(18, dtype=int)])
    stat, p = mcnemar_test(y1, y2, y_true)
    assert p < 0.05


def test_confusion_stats():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    s = confusion_matrix_stats(y_true, y_pred)
    assert 0.5 <= s["accuracy"] <= 1.0
    assert len(s["cm"]) == 2
