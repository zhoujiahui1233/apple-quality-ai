import numpy as np
from detector.bread import BreadAdapter


def _make_bread_image(with_mold=False, with_burn=False):
    img = np.full((200, 200, 3), (210, 180, 140), dtype=np.uint8)  # 面包色
    if with_mold:
        img[80:120, 80:120] = (90, 160, 90)  # 绿色霉斑
    if with_burn:
        img[30:60, 30:60] = (25, 45, 90)     # 深褐焦烘
    return img


def test_bread_clean():
    a = BreadAdapter()
    img = _make_bread_image()
    res = a.detect_surface(img, np.ones((200, 200), np.uint8) * 255, None)
    assert res.defect_count == 0
    assert a.classify_grade(res).grade == "premium"


def test_bread_mold_detected():
    a = BreadAdapter()
    img = _make_bread_image(with_mold=True)
    mask = np.ones((200, 200), np.uint8) * 255
    res = a.detect_surface(img, mask, None)
    assert res.defect_count >= 1
    assert any(d.kind == "mold" for d in res.defects)
    assert a.classify_grade(res).grade == "reject"


def test_bread_burn_detected():
    a = BreadAdapter()
    img = _make_bread_image(with_burn=True)
    mask = np.ones((200, 200), np.uint8) * 255
    res = a.detect_surface(img, mask, None)
    assert res.defect_count >= 1
    assert any(d.kind == "burn" for d in res.defects)
