import numpy as np
from detector.beverage import BeverageAdapter


def _make_bottle(liquid_level=0.5):
    img = np.full((200, 100, 3), (220, 220, 220), dtype=np.uint8)  # 瓶身透明
    y = int(200 * liquid_level)
    img[y:, :] = (120, 100, 40)          # 液面以下为饮料色
    return img


def test_beverage_level_ok():
    a = BeverageAdapter()
    img = _make_bottle(liquid_level=0.5)
    res = a.detect_surface(img, np.ones((200, 100), np.uint8) * 255, None)
    assert a.classify_grade(res).grade == "premium"


def test_beverage_level_low():
    a = BeverageAdapter()
    img = _make_bottle(liquid_level=0.2)
    res = a.detect_surface(img, np.ones((200, 100), np.uint8) * 255, None)
    assert any(d.kind == "level" for d in res.defects)
    assert a.classify_grade(res).grade == "reject"


def test_beverage_level_high():
    a = BeverageAdapter()
    img = _make_bottle(liquid_level=0.85)
    res = a.detect_surface(img, np.ones((200, 100), np.uint8) * 255, None)
    assert any(d.kind == "level" for d in res.defects)
