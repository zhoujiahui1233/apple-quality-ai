# -*- coding: utf-8 -*-
"""YOLO 手动框选标注导出 API 测试"""
import io
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, r"E:\apple")
import app as appmod


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "YOLO_DETECT_DIR", tmp_path)
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), tmp_path


def _png_bytes(w=200, h=150):
    img = np.full((h, w, 3), 200, np.uint8)
    cv2.circle(img, (100, 75), 60, (60, 40, 200), -1)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_export_box_ok(client):
    c, tmp = client
    data = {"file": (io.BytesIO(_png_bytes()), "a.png"), "category": "apple",
            "bbox": "[0.5, 0.5, 0.4, 0.4]"}
    r = c.post("/api/yolo/export_box", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    d = r.get_json()
    assert d["success"] is True
    img_file = tmp / "images" / "train" / d["image"]
    lbl_file = tmp / "labels" / "train" / d["label"]
    assert img_file.exists()
    assert lbl_file.exists()
    line = lbl_file.read_text(encoding="utf-8").strip()
    parts = line.split()
    assert parts[0] == "0"
    xc, yc, w, h = map(float, parts[1:])
    assert abs(xc - 0.5) < 0.01 and abs(yc - 0.5) < 0.01
    assert abs(w - 0.4) < 0.01 and abs(h - 0.4) < 0.01
    # 图片被正确保存（可解码）
    img = cv2.imdecode(np.fromfile(str(img_file), np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[:2] == (150, 200)


def test_export_box_rejects_bad_bbox(client):
    c, _ = client
    cases = ["[1.5,0.5,0.4,0.4]", "[0.5,0.5,0,0.4]", "[0.5]", "null", "[0.5,0.5,0.4,0.4,0.1]"]
    for bbox in cases:
        data = {"file": (io.BytesIO(_png_bytes()), "a.png"), "category": "apple", "bbox": bbox}
        r = c.post("/api/yolo/export_box", data=data, content_type="multipart/form-data")
        assert r.status_code == 400, bbox


def test_export_box_requires_file(client):
    c, _ = client
    r = c.post("/api/yolo/export_box", data={"category": "apple", "bbox": "[0.5,0.5,0.4,0.4]"},
               content_type="multipart/form-data")
    assert r.status_code == 400


def test_export_box_rejects_bad_category(client):
    c, _ = client
    data = {"file": (io.BytesIO(_png_bytes()), "a.png"), "category": "../../etc",
            "bbox": "[0.5,0.5,0.4,0.4]"}
    r = c.post("/api/yolo/export_box", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
# -*- coding: utf-8 -*-
"""YOLO 批量导出测试（追加到现有 test_yolo_export.py）"""
import io


def test_export_batch_ok(client):
    c, tmp = client
    data = {
        "category": "apple",
        "save_good": "0",
        "files": [(io.BytesIO(_png_bytes()), "a.png"),
                  (io.BytesIO(_png_bytes(w=300, h=200)), "b.png")],
    }
    r = c.post("/api/yolo/export_batch", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    d = r.get_json()
    assert d["success"] is True
    assert d["exported"] == 2
    assert d["skipped"] == 0
    imgs = list((tmp / "images" / "train").glob("auto_*.jpg"))
    lbls = list((tmp / "labels" / "train").glob("auto_*.txt"))
    assert len(imgs) == 2 and len(lbls) == 2
    for lbl in lbls:
        line = lbl.read_text(encoding="utf-8").strip()
        parts = line.split()
        assert parts[0] == "0"
        xc, yc, w, h = map(float, parts[1:])
        assert 0 < xc < 1 and 0 < yc < 1 and 0 < w <= 1 and 0 < h <= 1


def test_export_batch_save_good(client, tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    c, _ = client
    data = {
        "category": "apple",
        "save_good": "1",
        "files": [(io.BytesIO(_png_bytes()), "a.png")],
    }
    r = c.post("/api/yolo/export_batch", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    d = r.get_json()
    assert d["exported"] == 1 and d["saved_good"] == 1
    good_files = list((tmp_path / "dataset_collected" / "apple" / "合格").glob("*.jpg"))
    assert len(good_files) == 1


def test_export_batch_skips_blank(client):
    import numpy as np
    import cv2
    blank = np.full((120, 120, 3), 255, np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    c, tmp = client
    data = {"category": "apple", "save_good": "0",
            "files": [(io.BytesIO(buf.tobytes()), "blank.png")]}
    r = c.post("/api/yolo/export_batch", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    d = r.get_json()
    assert d["exported"] == 0 and d["skipped"] == 1

def test_export_batch_names_unique_on_same_second(client):
    """同秒两次批量导出不应覆盖同名文件"""
    c, tmp = client
    for _ in range(2):
        data = {
            "category": "apple",
            "save_good": "0",
            "files": [(io.BytesIO(_png_bytes()), "a.png")],
        }
        r = c.post("/api/yolo/export_batch", data=data, content_type="multipart/form-data")
        assert r.status_code == 200
        d = r.get_json()
        assert d["exported"] == 1
    imgs = list((tmp / "images" / "train").glob("auto_*.jpg"))
    lbls = list((tmp / "labels" / "train").glob("auto_*.txt"))
    assert len(imgs) == 2 and len(lbls) == 2
    names = [p.stem for p in imgs]
    assert len(set(names)) == 2, "same-second export produced duplicate names"
