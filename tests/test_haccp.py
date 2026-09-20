# -*- coding: utf-8 -*-
"""HACCP 品控视图服务单元测试"""
import pytest
from flask import Flask

from models import db, DetectionRecord, QualityRecord


@pytest.fixture()
def haccp_ctx(tmp_path):
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + str(tmp_path / "haccp.db")
    app.config["TESTING"] = True
    db.init_app(app)
    with app.app_context():
        db.create_all()
        yield
        db.session.remove()
        db.drop_all()


def _add_record(category="apple", grade="合格 Standard", severity="轻度",
                needs_review=False, days_ago=0):
    from datetime import datetime, timedelta
    r = DetectionRecord(
        category=category,
        filename="t.jpg",
        grade=grade,
        grade_color="(0,200,0)",
        defect_ratio=0.1,
        defect_count=1,
        apple_area=1000,
        defect_area=100,
        severity=severity,
        needs_review=needs_review,
    )
    r.created_at = datetime.now() - timedelta(days=days_ago)
    db.session.add(r)
    db.session.commit()
    return r


def test_flow_config_complete():
    from services.haccp_service import HACCP_FLOWS, get_flow
    for cat in ("apple", "bread", "beverage"):
        flow = get_flow(cat)
        assert flow["name"]
        assert len(flow["steps"]) == 3
        for step in flow["steps"]:
            for key in ("ccp", "name", "hazard", "limit", "monitor", "action"):
                assert step[key]
    assert get_flow("unknown") is HACCP_FLOWS["apple"]


def test_grade_class():
    from services.haccp_service import grade_class
    assert grade_class("优等 Premium") == "premium"
    assert grade_class("合格 Standard") == "standard"
    assert grade_class("不合格 Reject") == "reject"
    assert grade_class("") == "standard"


def test_ccp_statuses():
    from services.haccp_service import ccp_statuses, get_flow
    flow = get_flow("apple")
    empty = {"total": 0, "premium": 0, "standard": 0, "reject": 0, "review": 0, "pass_rate": 0.0}
    st = ccp_statuses(flow, empty)
    assert [s["status"] for s in st] == ["ok", "ok", "ok"]

    bad = {"total": 100, "premium": 90, "standard": 5, "reject": 5, "review": 0, "pass_rate": 0.95}
    st = ccp_statuses(flow, bad)
    assert st[0]["status"] == "warning"     # 5% 处于限值边界 → 关注

    ok2 = {"total": 100, "premium": 93, "standard": 5, "reject": 2, "review": 0, "pass_rate": 0.98}
    st = ccp_statuses(flow, ok2)
    assert st[0]["status"] == "ok"          # 2% 以内正常

    worse = {"total": 100, "premium": 88, "standard": 6, "reject": 6, "review": 0, "pass_rate": 0.94}
    st = ccp_statuses(flow, worse)
    assert st[0]["status"] == "danger"      # 6% 超标

    review = {"total": 100, "premium": 90, "standard": 5, "reject": 5, "review": 8, "pass_rate": 0.95}
    st = ccp_statuses(flow, review)
    assert st[1]["status"] == "warning"     # 有待复核 → 分选分级关注
    assert st[2]["status"] == "pending"     # 包装复检待处理


def test_overview(haccp_ctx):
    from services.haccp_service import overview
    _add_record("apple", "优等 Premium", "轻度")
    _add_record("apple", "不合格 Reject", "重度")
    _add_record("apple", "合格 Standard", "中度", needs_review=True)
    _add_record("bread", "合格 Standard", "轻度")

    d = overview("apple", "2099-01-01")
    # 2099-01-01 无记录
    assert d["stats"]["total"] == 0
    assert d["ccp"][0]["status"] == "ok"

    from datetime import datetime
    period = datetime.now().strftime("%Y-%m-%d")
    d2 = overview("apple", period)
    assert d2["stats"]["total"] == 3
    assert d2["stats"]["premium"] == 1
    assert d2["stats"]["standard"] == 1
    assert d2["stats"]["reject"] == 1
    assert d2["stats"]["review"] == 1
    assert d2["hazards"][0]["count"] == 1  # 重度缺陷分布


def test_generate_record(haccp_ctx):
    from services.haccp_service import generate_record, overview
    _add_record("apple", "优等 Premium", "轻度")
    _add_record("apple", "不合格 Reject", "重度")
    from datetime import datetime
    period = datetime.now().strftime("%Y-%m-%d")

    rec = generate_record("apple", period, "测试员")
    assert rec.record_no.startswith("QR-")
    assert rec.total == 2
    assert rec.reject == 1
    assert "超标" in rec.conclusion
    assert QualityRecord.query.count() == 1

    rec2 = generate_record("apple", period, "测试员")
    assert rec2.record_no.endswith("002")
    assert QualityRecord.query.count() == 2

    # 导出 Excel
    buf, filename = __import__("services.haccp_service", fromlist=["export_record_excel"]).export_record_excel(rec)
    assert buf.getvalue().startswith(b"PK")
    assert filename.endswith(".xlsx")


def test_generate_empty_day(haccp_ctx):
    from services.haccp_service import generate_record
    rec = generate_record("apple", "2099-01-01", "测试员")
    assert rec.total == 0
    assert "无检测数据" in rec.conclusion
    assert rec.pass_rate == 0.0

def test_generate_record_invalid_period_falls_back(haccp_ctx):
    """invalid period should fall back to today without raising"""
    from services.haccp_service import generate_record
    rec = generate_record("apple", "not-a-date", "测试员")
    assert rec.total == 0
    assert rec.record_no.startswith("QR-")
    assert "无检测数据" in rec.conclusion
    from datetime import datetime
    assert rec.period == datetime.now().strftime("%Y-%m-%d")
