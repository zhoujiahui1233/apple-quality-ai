# -*- coding: utf-8 -*-
"""论文吸收功能测试：白平衡 / 融合判定 / 果径估计"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import apple_defect_detector as D


def _test_img():
    # 生成一张简单的"苹果"图：红色圆 + 深色缺陷斑
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.circle(img, (200, 150), 90, (30, 30, 200), -1)  # 红色苹果(BGR)
    cv2.circle(img, (220, 160), 20, (30, 30, 60), -1)   # 暗缺陷
    return img


def test_auto_white_balance_shape():
    img = _test_img()
    out = D.auto_white_balance(img, strength=0.6)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    blue = img.copy(); blue[:, :, 2] = 200
    out2 = D.auto_white_balance(blue, strength=1.0)
    assert np.mean(out2[:, :, 2]) < np.mean(blue[:, :, 2])


def test_auto_white_balance_off():
    img = _test_img()
    out = D.auto_white_balance(img, strength=0)
    assert np.array_equal(out, img)


def test_estimate_diameter():
    cnt = np.array([[0, 0], [100, 0], [100, 50], [50, 50], [0, 50]], dtype=np.float32)
    d = D.estimate_diameter_px(cnt, img_width=200)
    assert d["diameter_px"] > 100
    assert d["diameter_norm"] == pytest.approx(d["diameter_px"] / 200, rel=1e-3)
    assert d["aspect"] == pytest.approx(2.0, rel=0.1)


def test_ensemble_grade_agree():
    r = D.ensemble_score("优质 Premium", "优质 Premium", "优质 Premium", "优质 Premium",
                         0.9, 0.8, 0.7, 0.9, 0.01, 0.05, 0.15)
    assert "Premium" in r["grade"]
    assert r["conf"] > 0.5


def test_ensemble_grade_prior_flips_lowconf():
    # 模型低置信全合格 + 缺陷比例 40%（强先验不合格）→ 先验把融合拉向不合格
    r = D.ensemble_score("优质 Premium", "优质 Premium", "优质 Premium", "优质 Premium",
                         0.51, 0.51, 0.51, 0.51, 0.40, 0.05, 0.15)
    assert "Reject" in r["grade"]
    assert r["votes"]
    # 同场景缺陷很少时融合应判合格
    r2 = D.ensemble_score("优质 Premium", "优质 Premium", "优质 Premium", "优质 Premium",
                          0.51, 0.51, 0.51, 0.51, 0.02, 0.05, 0.15)
    assert "Premium" in r2["grade"]

def test_extract_features_v3_dim():
    img = _test_img()
    f = D.extract_features_v2(img)
    assert f is not None
    assert len(f) == 185  # 170 + 着色率4 + GLCM8 + 果径3
