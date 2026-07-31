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
