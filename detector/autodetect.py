# -*- coding: utf-8 -*-
"""
品类自动识别：基于颜色/亮度启发式判断
用于 category=auto 时自动选择检测品类。
注意：这只是轻量级判断，大规模实验建议明确选择品类。
"""
import cv2
import numpy as np

CATEGORY_NAMES = {"apple": "\u82f9\u679c", "bread": "\u9762\u5305", "beverage": "\u996e\u6599"}


def _ratio(mask):
    return float(np.mean(mask > 0)) if mask.size else 0.0


def auto_detect_category(img_bgr: np.ndarray) -> tuple:
    """返回 (category_id, confidence)，confidence 为 0~1 的得分差。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    hh, ww = h.shape
    cy, cx = hh // 2, ww // 2
    rh = slice(max(0, cy - hh // 4), min(hh, cy + hh // 4))
    rw = slice(max(0, cx - ww // 4), min(ww, cx + ww // 4))

    def region(arr):
        return arr[rh, rw]

    hr, sr, vr = region(h), region(s), region(v)

    # 苹果：红色/绿色水果色彩
    red = ((hr >= 0) & (hr <= 10)) | (hr >= 170)
    green = (hr >= 30) & (hr <= 90)
    fruit = (red & (sr > 60) & (vr > 60)) | (green & (sr > 50) & (vr > 60))
    apple_score = _ratio(fruit)

    # 面包：棕色系（低饱和暖色）
    brown = (hr >= 5) & (hr <= 30) & (sr >= 30) & (sr <= 220) & (vr >= 70) & (vr <= 235)
    bread_score = _ratio(brown)

    # 饮料：高亮白/冷色标签（白色高亮+蓝色）
    bright_white = (vr > 190) & (sr < 70)
    blue = (hr >= 90) & (hr <= 140) & (sr > 50) & (vr > 60)
    beverage_score = 0.65 * _ratio(bright_white) + 0.35 * _ratio(blue)

    scores = {"apple": apple_score, "bread": bread_score, "beverage": beverage_score}
    best = max(scores, key=scores.get)
    second = sorted(scores.values(), reverse=True)[1]
    confidence = max(0.0, min(1.0, scores[best] - second + 0.15))
    # 得分过低时默认苹果
    if scores[best] < 0.10:
        best, confidence = "apple", 0.3
    return best, round(confidence, 2)
