# -*- coding: utf-8 -*-
"""消融实验 / 鲁棒性测试脚本的核心逻辑测试"""
import numpy as np

import scripts.ablation_feature_dims as afd
import scripts.robustness_test as rbt
from scripts.ablation_feature_dims import rank_features, _safe_n_jobs
from scripts.robustness_test import perturb, rotate


def test_rank_features_returns_all_indices():
    rng = np.random.RandomState(0)
    X = rng.rand(20, 170)
    y = np.array([0] * 10 + [1] * 10)
    order = rank_features(X, y, seed=42)
    assert len(order) == 170
    assert sorted(order) == list(range(170))


def test_safe_n_jobs_positive():
    assert _safe_n_jobs() >= 1


def test_ablation_result_keys():
    rng = np.random.RandomState(1)
    X = rng.rand(30, 170).astype(np.float32)
    y = np.array([0] * 15 + [1] * 15)
    rng.shuffle(y)
    res = afd.run_ablation(X, y, [15, 50], 0.3, 42)
    assert res["n_train"] + res["n_test"] == 30
    assert set(res["results"]["svm"].keys()) == {"15", "50"}
    assert set(res["results"]["rf"].keys()) == {"15", "50"}
    assert len(res["order"]) == 20
    for m in ("svm", "rf"):
        for k in ("15", "50"):
            v = res["results"][m][k]
            assert 0.0 <= v <= 100.0


def test_perturb_shapes():
    img = (np.random.RandomState(2).rand(120, 160, 3) * 255).astype(np.uint8)
    for kind in ("blur2", "blur4", "bright+30", "bright-30", "rot5", "rot15"):
        out = perturb(img, kind)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


def test_perturb_changes_pixels():
    img = (np.random.RandomState(3).rand(120, 160, 3) * 255).astype(np.uint8)
    assert not np.array_equal(perturb(img, "blur2"), img)
    assert not np.array_equal(perturb(img, "bright+30"), img)
    assert not np.array_equal(perturb(img, "rot5"), img)


def test_rotate_keeps_size():
    img = (np.random.RandomState(4).rand(100, 100, 3) * 255).astype(np.uint8)
    assert rotate(img, 15).shape == img.shape