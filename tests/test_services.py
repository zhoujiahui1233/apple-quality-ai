# -*- coding: utf-8 -*-
"""服务层单元测试：品类自动识别 / 数据集管理 / 模型版本"""
import cv2
import numpy as np


def test_autodetect_apple():
    from detector.autodetect import auto_detect_category
    img = np.full((200, 200, 3), 255, np.uint8)
    cv2.circle(img, (100, 100), 80, (60, 40, 200), -1)  # 红色苹果
    cat, conf = auto_detect_category(img)
    assert cat == "apple"
    assert 0.0 <= conf <= 1.0


def test_autodetect_bread():
    from detector.autodetect import auto_detect_category
    img = np.full((200, 200, 3), (60, 100, 150), np.uint8)  # 棕色面包
    cat, _ = auto_detect_category(img)
    assert cat == "bread"


def test_dataset_service_list_delete(tmp_path, monkeypatch):
    from services.dataset_service import DatasetService
    monkeypatch.setattr(DatasetService, "BASE", tmp_path)
    d = tmp_path / "apple" / "合格"
    d.mkdir(parents=True)
    (d / "a.png").write_bytes(b"x")
    svc = DatasetService()
    res = svc.list()
    assert res["total"] == 1
    assert "apple" in res["categories"]
    assert "合格" in res["categories"]["apple"]
    assert svc.delete("apple", "合格", "a.png") is True
    assert svc.delete("apple", "合格", "a.png") is False


def test_model_registry_register_rollback(tmp_path, monkeypatch):
    from services.model_registry import ModelRegistry
    monkeypatch.setattr(ModelRegistry, "FILE", tmp_path / "registry.json")
    reg = ModelRegistry()
    e1 = reg.register("apple", "svm", 99.0, 100)
    e2 = reg.register("apple", "svm", 99.5, 150)
    assert e1["version"] == "v1"
    assert e2["version"] == "v2"
    assert reg.active("apple", "svm")["version"] == "v2"
    assert reg.rollback("apple", "svm", "v1") is True
    assert reg.active("apple", "svm")["version"] == "v1"
    assert reg.rollback("apple", "svm", "v9") is False


def test_dataset_service_delete_rejects_path_traversal(tmp_path, monkeypatch):
    from services.dataset_service import DatasetService
    monkeypatch.setattr(DatasetService, "BASE", tmp_path)
    d = tmp_path / "apple" / "合格"
    d.mkdir(parents=True)
    (d / "a.png").write_bytes(b"x")
    # 路径穿越文件名应被拒绝，且不能删除外部文件
    outside = tmp_path.parent / "target.txt"
    outside.write_bytes(b"secret")
    svc = DatasetService()
    assert svc.delete("apple", "合格", "../合格/../../target.txt") is False
    assert svc.delete("apple", "..", "a.png") is False
    assert svc.delete("apple", "合格", "a.png") is True
    assert outside.exists() is True


def test_dataset_service_move_rejects_path_traversal(tmp_path, monkeypatch):
    from services.dataset_service import DatasetService
    monkeypatch.setattr(DatasetService, "BASE", tmp_path)
    good = tmp_path / "apple" / "合格"
    good.mkdir(parents=True)
    (good / "a.png").write_bytes(b"x")
    svc = DatasetService()
    ok, info = svc.move("apple", "../../target.png", "合格", "不合格")
    assert ok is False
    assert (tmp_path / "apple" / "不合格" / "a.png").exists() is False


def test_dataset_batch_import_route(tmp_path, monkeypatch):
    """批量导入：合格/不合格/不合格+缺陷种类 分别入库，坏图被拒"""
    import io
    import cv2
    import numpy as np
    import config as app_config
    monkeypatch.setattr(app_config, "ROOT_DIR", tmp_path)
    import app as appmod
    client = appmod.app.test_client()
    with client.session_transaction() as s:
        s["sb_user"] = {"id": "t", "email": "t@t.com", "nickname": "tester", "role": "user"}

    img = np.full((64, 64, 3), (60, 40, 200), np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    data = {
        "category": "apple",
        "files": [(io.BytesIO(buf.tobytes()), "a1.png"),
                  (io.BytesIO(buf.tobytes()), "a2.png"),
                  (io.BytesIO(b"not-an-image"), "bad.png")],
        "labels": ["合格", "不合格", "不合格"],
        "defects": ["", "腐烂", "碰伤"],
    }
    r = client.post("/api/dataset/batch_import", data=data,
                    content_type="multipart/form-data")
    d = r.get_json()
    assert d["success"] is True
    assert d["total"] == 3
    assert d["ok"] == 2
    base = tmp_path / "dataset_collected" / "apple"
    assert (base / "合格").is_dir()
    assert (base / "不合格_腐烂").is_dir()
    assert not (base / "不合格_碰伤").is_dir()
    n_ok = len(list((base / "合格").iterdir())) + len(list((base / "不合格_腐烂").iterdir()))
    assert n_ok == 2


def test_dataset_batch_import_rejects_bad_category(tmp_path, monkeypatch):
    import io
    import cv2
    import numpy as np
    import config as app_config
    monkeypatch.setattr(app_config, "ROOT_DIR", tmp_path)
    import app as appmod
    client = appmod.app.test_client()
    with client.session_transaction() as s:
        s["sb_user"] = {"id": "t", "email": "t@t.com", "nickname": "tester", "role": "user"}
    img = np.full((32, 32, 3), 200, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    r = client.post("/api/dataset/batch_import",
                    data={"category": "../etc",
                          "files": [(io.BytesIO(buf.tobytes()), "x.jpg")],
                          "labels": ["合格"], "defects": [""]},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_dataset_batch_import_quality_dedup(tmp_path, monkeypatch):
    """批量导入智能选项：去重拦截、质量拦截、强制导入跳过"""
    import io
    import cv2
    import numpy as np
    import config as app_config
    monkeypatch.setattr(app_config, "ROOT_DIR", tmp_path)
    import app as appmod
    client = appmod.app.test_client()
    with client.session_transaction() as s:
        s["sb_user"] = {"id": "t", "email": "t@t.com", "nickname": "tester", "role": "user"}

    img = np.full((64, 64, 3), (60, 40, 200), np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok

    def post(fname, payload_img, qc="1", dup="1", force="0", label="合格"):
        o, b = cv2.imencode(".png", payload_img)
        return client.post("/api/dataset/batch_import",
                           data={"category": "apple",
                                 "files": [(io.BytesIO(b.tobytes()), fname)],
                                 "labels": [label], "defects": [""],
                                 "quality_check": qc, "dedup": dup, "force": force},
                           content_type="multipart/form-data").get_json()

    # 正常图首次导入成功
    assert post("n.png", img)["ok"] == 1
    # 相同图片再次导入 -> 去重拦截
    d = post("dup.png", img)
    assert d["ok"] == 0 and any("重复" in x["reason"] for x in d["results"])
    # 勾选强制导入 -> 跳过去重
    assert post("dup2.png", img, force="1")["ok"] == 1
    # 高亮度（近乎过曝）图片 -> 质量拦截
    bright = np.full((64, 64, 3), 250, np.uint8)
    d = post("bright.png", bright, dup="0")
    assert d["ok"] == 0 and any("质量" in x["reason"] for x in d["results"])
    # 强制导入 -> 跳过质量拦截
    assert post("bright2.png", bright, dup="0", force="1")["ok"] == 1


def test_severity_grading():
    """缺陷严重度分级：面积占比 <10% 轻度，10%~30% 中度，>=30% 重度"""
    from services.detection_service import severity_of
    assert severity_of(0.0) == "轻度"
    assert severity_of(0.05) == "轻度"
    assert severity_of(0.10) == "中度"
    assert severity_of(0.29) == "中度"
    assert severity_of(0.30) == "重度"
    assert severity_of(0.85) == "重度"
    assert severity_of(None) == "轻度"


def test_review_reasons():
    """待复核判定：置信度灰区 / 模型不一致 / 缺陷占比临界"""
    from services.detection_service import review_reasons
    # 置信度灰区：YOLO 0.6 命中，DL 0.95 不命中
    r1 = review_reasons(yolo_conf=0.6, dl_conf=0.95,
                        grades=("优等 Premium",), defect_ratio=0.02,
                        premium_th=0.05, standard_th=0.15)
    assert "YOLO置信度偏低" in r1
    assert "深度学习置信度偏低" not in r1
    # 模型结论不一致
    r2 = review_reasons(yolo_conf=0.95, dl_conf=0.95,
                        grades=("优等 Premium", "不合格 Reject"), defect_ratio=0.02,
                        premium_th=0.05, standard_th=0.15)
    assert "模型结论不一致" in r2
    # 缺陷占比临界（接近优质/合格阈值）
    r3 = review_reasons(yolo_conf=0.95, dl_conf=0.95,
                        grades=("优等 Premium",), defect_ratio=0.048,
                        premium_th=0.05, standard_th=0.15)
    assert any("阈值" in x for x in r3)
    # 高置信度且一致、占比远离阈值 → 不标记
    r4 = review_reasons(yolo_conf=0.95, dl_conf=0.95,
                        grades=("优等 Premium",), defect_ratio=0.02,
                        premium_th=0.05, standard_th=0.15)
    assert r4 == []
    # 置信度过低（模型未加载为 0）不应误报灰区
    r5 = review_reasons(yolo_conf=0.0, dl_conf=0.0, grades=(), defect_ratio=0.02,
                        premium_th=0.05, standard_th=0.15)
    assert "YOLO置信度偏低" not in r5
