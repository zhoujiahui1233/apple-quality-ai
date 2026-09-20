# -*- coding: utf-8 -*-
"""数据集智能批量导入（自动分割 + 手动修正 + YOLO 贯通）测试"""
import io
import json
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, r"E:\apple")
import app as appmod


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "YOLO_DETECT_DIR", tmp_path / "yolo")
    appmod.app.config["TESTING"] = True
    return appmod.app.test_client(), tmp_path


def _apple_png(w=200, h=150):
    img = np.full((h, w, 3), 200, np.uint8)
    cv2.circle(img, (w // 2, h // 2), min(w, h) // 3, (60, 40, 200), -1)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_import_segment_ok(client):
    c, _ = client
    data = {"files": [(io.BytesIO(_apple_png()), "a.png"),
                      (io.BytesIO(_apple_png(w=300, h=200)), "b.png")]}
    r = c.post("/api/import/segment", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    d = r.get_json()
    assert d["success"] is True
    assert len(d["items"]) == 2
    for it in d["items"]:
        assert it["bbox"] is not None and len(it["bbox"]) == 4
        assert it["preview"].startswith("data:image/jpeg;base64,")
        xc, yc, w, h = it["bbox"]
        assert 0 < xc < 1 and 0 < yc < 1 and 0 < w <= 1 and 0 < h <= 1


def test_import_segment_blank_skipped(client):
    blank = np.full((120, 120, 3), 255, np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    c, _ = client
    r = c.post("/api/import/segment",
               data={"files": [(io.BytesIO(buf.tobytes()), "blank.png")]},
               content_type="multipart/form-data")
    d = r.get_json()
    assert d["items"][0]["bbox"] is None
    assert d["items"][0]["preview"] is not None and d["items"][0]["preview"].startswith("data:image/jpeg;base64,")


def test_batch_import_with_boxes_exports_yolo(client, monkeypatch, tmp_path):
    import config
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path / "root")
    c, _ = client
    data = {
        "category": "apple",
        "quality_check": "0", "dedup": "0", "force": "1",
        "files": [(io.BytesIO(_apple_png()), "a.png")],
        "labels": ["合格"], "defects": [""],
        "boxes": "[[0.5, 0.5, 0.4, 0.4]]",
    }
    r = c.post("/api/dataset/batch_import", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    d = r.get_json()
    assert d["success"] is True and d["ok"] == 1
    assert d["yolo_exported"] == 1
    # 分类数据集
    good = tmp_path / "root" / "dataset_collected" / "apple" / "合格"
    assert len(list(good.glob("*.png"))) == 1
    # YOLO 标注
    yimg = list((tmp_path / "yolo" / "images" / "train").glob("import_*.jpg"))
    ylbl = list((tmp_path / "yolo" / "labels" / "train").glob("import_*.txt"))
    assert len(yimg) == 1 and len(ylbl) == 1
    parts = ylbl[0].read_text(encoding="utf-8").strip().split()
    assert parts[0] == "0" and abs(float(parts[1]) - 0.5) < 0.01


def test_batch_import_without_boxes_no_yolo(client, monkeypatch, tmp_path):
    import config
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path / "root")
    c, _ = client
    data = {
        "category": "apple",
        "quality_check": "0", "dedup": "0", "force": "1",
        "files": [(io.BytesIO(_apple_png()), "a.png")],
        "labels": ["合格"], "defects": [""],
    }
    r = c.post("/api/dataset/batch_import", data=data, content_type="multipart/form-data")
    d = r.get_json()
    assert d["ok"] == 1 and d["yolo_exported"] == 0

def test_batch_import_null_box_skips_yolo(client, monkeypatch, tmp_path):
    """boxes 中为 null 的项跳过 YOLO 导出，但数据集仍保存"""
    import config
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path / "root")
    c, _ = client
    data = {
        "category": "apple",
        "quality_check": "0", "dedup": "0", "force": "1",
        "files": [(io.BytesIO(_apple_png()), "a.png")],
        "labels": ["合格"], "defects": [""],
        "boxes": "[[null]]",
    }
    r = c.post("/api/dataset/batch_import", data=data, content_type="multipart/form-data")
    d = r.get_json()
    assert d["success"] is True and d["ok"] == 1
    assert d["yolo_exported"] == 0
    good = tmp_path / "root" / "dataset_collected" / "apple" / "合格"
    assert len(list(good.glob("*.png"))) == 1


def test_batch_import_names_unique_on_same_second(client, monkeypatch, tmp_path):
    """同秒内重复批量导入不覆盖已保存文件"""
    import config
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path / "root")
    c, _ = client
    for _ in range(2):
        data = {
            "category": "apple",
            "quality_check": "0", "dedup": "0", "force": "1",
            "files": [(io.BytesIO(_apple_png()), "a.png")],
            "labels": ["合格"], "defects": [""],
            "boxes": "[[0.5, 0.5, 0.4, 0.4]]",
        }
        r = c.post("/api/dataset/batch_import", data=data, content_type="multipart/form-data")
        assert r.status_code == 200
        d = r.get_json()
        assert d["ok"] == 1 and d["yolo_exported"] == 1
    good = tmp_path / "root" / "dataset_collected" / "apple" / "合格"
    dataset_files = list(good.glob("*.png"))
    assert len(dataset_files) == 2, "same-second import overwrote dataset files"
    yimgs = list((tmp_path / "yolo" / "images" / "train").glob("import_*.jpg"))
    assert len(yimgs) == 2, "same-second import overwrote YOLO images"


def test_auto_bbox_helper(client):
    """_auto_bbox: 圆形主体出框，空白图返回 None"""
    import app as appmod
    img = np.full((150, 200, 3), 200, np.uint8)
    cv2.circle(img, (100, 75), 60, (60, 40, 200), -1)
    bb = appmod._auto_bbox(img)
    assert bb is not None and len(bb) == 4
    xc, yc, w, h = bb
    assert 0.35 < xc < 0.65 and 0.35 < yc < 0.65 and w > 0.4 and h > 0.4
    blank = np.full((120, 120, 3), 255, np.uint8)
    assert appmod._auto_bbox(blank) is None


def test_batch_delete_ok(client, monkeypatch, tmp_path):
    """批量删除：同步删除数据集文件夹中的对应文件"""
    import config
    import app as appmod
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path / "root")
    monkeypatch.setattr(appmod.dataset_service, "BASE", tmp_path / "root" / "dataset_collected")
    c, _ = client
    data = {
        "category": "apple",
        "quality_check": "0", "dedup": "0", "force": "1",
        "files": [(io.BytesIO(_apple_png()), "a.png"),
                  (io.BytesIO(_apple_png(w=300, h=200)), "b.png")],
        "labels": ["合格", "不合格"], "defects": ["", "腐烂"],
        "boxes": "[[0.5,0.5,0.4,0.4],[null]]",
    }
    r = c.post("/api/dataset/batch_import", data=data, content_type="multipart/form-data")
    d = r.get_json()
    assert d["ok"] == 2
    base = tmp_path / "root" / "dataset_collected" / "apple"
    good_files = list((base / "合格").glob("*.png"))
    bad_files = list((base / "不合格_腐烂").glob("*.png"))
    assert len(good_files) == 1 and len(bad_files) == 1
    items = json.dumps([
        {"category": "apple", "label": "合格", "filename": good_files[0].name},
        {"category": "apple", "label": "不合格_腐烂", "filename": bad_files[0].name},
    ])
    r2 = c.post("/api/dataset/batch_delete",
                data={"category": "apple", "items": items},
                content_type="multipart/form-data")
    d2 = r2.get_json()
    assert d2["success"] is True and d2["deleted"] == 2 and d2["failed"] == []
    assert not (base / "合格" / good_files[0].name).exists()
    assert not (base / "不合格_腐烂" / bad_files[0].name).exists()


def test_batch_delete_invalid_params(client):
    c, _ = client
    r = c.post("/api/dataset/batch_delete", data={"category": "apple", "items": "[]"},
               content_type="multipart/form-data")
    assert r.status_code == 400
    r2 = c.post("/api/dataset/batch_delete", data={"category": "../../etc", "items": "[{}]"},
                content_type="multipart/form-data")
    assert r2.status_code == 400


def test_batch_delete_missing_file(client, monkeypatch, tmp_path):
    import app as appmod
    monkeypatch.setattr(appmod.dataset_service, "BASE", tmp_path / "root" / "dataset_collected")
    c, _ = client
    r = c.post("/api/dataset/batch_delete",
               data={"category": "apple",
                     "items": '[{"label": "合格", "filename": "nonexist.png"}]'},
               content_type="multipart/form-data")
    d = r.get_json()
    assert d["success"] is True and d["deleted"] == 0
    assert len(d["failed"]) == 1
