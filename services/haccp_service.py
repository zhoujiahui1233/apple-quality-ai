# -*- coding: utf-8 -*-
"""HACCP 品控视图服务。

把 AI 表面检测结果映射到食品安全关键控制点（CCP），按品类生成
质检记录单，支撑 HACCP 体系的监控程序与记录保持程序。
"""

import json
from datetime import datetime, timedelta

from models import db, DetectionRecord, QualityRecord

# ==================== HACCP 流程配置 ====================
HACCP_FLOWS = {
    "apple": {
        "name": "苹果",
        "icon": "🍎",
        "steps": [
            {
                "ccp": "CCP1",
                "name": "原料验收",
                "hazard": "腐烂、病虫害、机械伤（物理性危害）",
                "limit": "外观缺陷占比 ≤ 5%",
                "monitor": "AI 表面缺陷检测（SVM / RF / DL / YOLO）",
                "action": "超标批次隔离复检，不合格原料拒收",
            },
            {
                "ccp": "CCP2",
                "name": "分选分级",
                "hazard": "等级混装、缺陷漏检",
                "limit": "分级准确率 ≥ 95%",
                "monitor": "四模型交叉验证 + 置信度评估",
                "action": "低置信度样本转入人工复核",
            },
            {
                "ccp": "CCP3",
                "name": "包装复检",
                "hazard": "待复核缺陷遗漏出厂",
                "limit": "待复核记录清零",
                "monitor": "待复核队列追踪",
                "action": "复核不合格品剔除并修正等级",
            },
        ],
    },
    "apple_slice": {
        "name": "苹果切片",
        "icon": "🍎",
        "steps": [
            {
                "ccp": "CCP1",
                "name": "切片原料验收",
                "hazard": "原料褐变、机械损伤、微生物污染（物理/生物性危害）",
                "limit": "原料果外观缺陷占比 ≤ 5%，无腐烂",
                "monitor": "整果外观检测（SVM / RF / DL / YOLO）+ 人工目视",
                "action": "超标原料拒收，冷藏运输链温度核验",
            },
            {
                "ccp": "CCP2",
                "name": "切片褐变控制",
                "hazard": "酶促褐变（PPO）、切面失水",
                "limit": "褐变面积占比 ≤ 15%（2 级以内）",
                "monitor": "切片视觉褐变检测（BI / 褐变面积占比）+ 电子鼻异味监测",
                "action": "超限样本立即隔离，评估护色处理（VC/柠檬酸）或缩短货架期",
            },
            {
                "ccp": "CCP3",
                "name": "冷链与货架期复核",
                "hazard": "微生物增殖、异味发酵",
                "limit": "4℃ 冷链，出货前褐变与气味双指标复核",
                "monitor": "电子鼻传感器（乙醇/挥发性物质）趋势 + 切片视觉复核",
                "action": "异常批次下架，记录剩余货架期并追溯",
            },
        ],
    },
    "bread": {
        "name": "面包",
        "icon": "🍞",
        "steps": [
            {
                "ccp": "CCP1",
                "name": "原料验收",
                "hazard": "异物混入、霉变原料",
                "limit": "外观异常占比 ≤ 5%",
                "monitor": "AI 表面缺陷检测",
                "action": "异常批次隔离，原料拒收",
            },
            {
                "ccp": "CCP2",
                "name": "烘焙外观检验",
                "hazard": "焦糊、色泽不均、塌陷",
                "limit": "色泽异常占比 ≤ 8%",
                "monitor": "AI 表面缺陷检测（色泽 / 纹理）",
                "action": "调整烘焙参数，异常品降级",
            },
            {
                "ccp": "CCP3",
                "name": "冷却包装复检",
                "hazard": "表面污染、异物附着",
                "limit": "待复核记录清零",
                "monitor": "待复核队列追踪",
                "action": "污染品剔除，包装线清洁",
            },
        ],
    },
    "beverage": {
        "name": "饮料",
        "icon": "🥤",
        "steps": [
            {
                "ccp": "CCP1",
                "name": "原料验收",
                "hazard": "异物混入、包装破损",
                "limit": "外观异常占比 ≤ 5%",
                "monitor": "AI 表面缺陷检测",
                "action": "异常批次隔离，原料拒收",
            },
            {
                "ccp": "CCP2",
                "name": "灌装密封检验",
                "hazard": "漏液、瓶身变形",
                "limit": "外观异常占比 ≤ 8%",
                "monitor": "AI 表面缺陷检测（轮廓 / 反光）",
                "action": "调整灌装线，异常品剔除",
            },
            {
                "ccp": "CCP3",
                "name": "成品外观检验",
                "hazard": "标签错贴、异物附着",
                "limit": "待复核记录清零",
                "monitor": "待复核队列追踪",
                "action": "标签返工，异物品剔除",
            },
        ],
    },
}

STATUS_META = {
    "ok": ("正常", "#00ff88"),
    "warning": ("关注", "#ffc107"),
    "danger": ("超标", "#ff4757"),
    "pending": ("待处理", "#ffc107"),
}


def get_flow(category):
    """获取品类流程配置（缺省回退苹果）。"""
    return HACCP_FLOWS.get(category or "apple") or HACCP_FLOWS["apple"]


def grade_class(grade):
    g = grade or ""
    if "Premium" in g or "优等" in g:
        return "premium"
    if "Reject" in g or "不合格" in g:
        return "reject"
    return "standard"


def _period_range(period):
    try:
        start = datetime.strptime(period + " 00:00:00", "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end


def _stats_for(category, start, end):
    query = DetectionRecord.query.filter(
        DetectionRecord.created_at >= start, DetectionRecord.created_at < end)
    if category:
        query = query.filter(DetectionRecord.category == category)
    records = query.all()
    total = len(records)
    counts = {"premium": 0, "standard": 0, "reject": 0}
    review = 0
    for r in records:
        counts[grade_class(r.grade)] += 1
        if r.needs_review:
            review += 1
    pass_rate = (counts["premium"] + counts["standard"]) / total if total else 0.0
    return {
        "total": total,
        "premium": counts["premium"],
        "standard": counts["standard"],
        "reject": counts["reject"],
        "review": review,
        "pass_rate": round(pass_rate, 4),
    }


def _ccp_status(idx, stats):
    total = stats.get("total") or 0
    reject = stats.get("reject") or 0
    review = stats.get("review") or 0
    reject_rate = reject / total if total else 0.0
    review_rate = review / total if total else 0.0
    if idx == 0:
        if reject_rate > 0.05:
            return "danger"
        if reject_rate > 0.02:
            return "warning"
        return "ok"
    if idx == 1:
        if review_rate > 0.10:
            return "danger"
        if review_rate > 0:
            return "warning"
        return "ok"
    return "pending" if review > 0 else "ok"


def ccp_statuses(flow, stats):
    out = []
    for i, step in enumerate(flow["steps"]):
        status = _ccp_status(i, stats)
        label, color = STATUS_META[status]
        out.append({
            "ccp": step["ccp"],
            "name": step["name"],
            "hazard": step["hazard"],
            "limit": step["limit"],
            "monitor": step["monitor"],
            "action": step["action"],
            "status": status,
            "status_label": label,
            "status_color": color,
        })
    return out


def _hazard_breakdown(category, start, end):
    query = DetectionRecord.query.filter(
        DetectionRecord.created_at >= start, DetectionRecord.created_at < end)
    if category:
        query = query.filter(DetectionRecord.category == category)
    records = [r for r in query.all() if grade_class(r.grade) == "reject"]
    sev_map = {"重度": 0, "中度": 0, "轻度": 0}
    for r in records:
        sev = r.severity or "轻度"
        if sev in sev_map:
            sev_map[sev] += 1
        else:
            sev_map["轻度"] += 1
    return [
        {"name": "重度缺陷（疑似腐烂 / 破损）", "count": sev_map["重度"], "color": "#ff4757"},
        {"name": "中度缺陷", "count": sev_map["中度"], "color": "#ffc107"},
        {"name": "轻度缺陷（疑似污渍 / 斑点）", "count": sev_map["轻度"], "color": "#00d4ff"},
    ]


def overview(category="apple", period=None):
    """HACCP 总览数据：流程、CCP 状态、当日统计、危害分布。"""
    period = period or datetime.now().strftime("%Y-%m-%d")
    try:
        datetime.strptime(period, "%Y-%m-%d")
    except (TypeError, ValueError):
        period = datetime.now().strftime("%Y-%m-%d")
    start, end = _period_range(period)
    stats = _stats_for(category, start, end)
    flow = get_flow(category)
    ccp = ccp_statuses(flow, stats)
    return {
        "category": category,
        "period": period,
        "flow": flow,
        "stats": stats,
        "ccp": ccp,
        "hazards": _hazard_breakdown(category, start, end),
    }


def generate_record(category="apple", period=None, operator="本地用户"):
    """生成某品类某日的质检记录单并入库，返回记录对象。"""
    period = period or datetime.now().strftime("%Y-%m-%d")
    try:
        datetime.strptime(period, "%Y-%m-%d")
    except (TypeError, ValueError):
        period = datetime.now().strftime("%Y-%m-%d")
    start, end = _period_range(period)
    stats = _stats_for(category, start, end)
    flow = get_flow(category)
    ccp = ccp_statuses(flow, stats)
    total = stats["total"]

    prefix = "QR-" + period.replace("-", "") + "-"
    seq = QualityRecord.query.filter(QualityRecord.record_no.like(prefix + "%")).count() + 1
    record_no = "{}{:03d}".format(prefix, seq)

    danger = [c for c in ccp if c["status"] == "danger"]
    warning = [c for c in ccp if c["status"] == "warning"]
    if total == 0:
        conclusion = "当日无检测数据，未形成品控结论"
    elif danger:
        names = "、".join(c["ccp"] for c in danger)
        if any(c["ccp"] == "CCP1" for c in danger):
            conclusion = "不合格率超标（{}），建议该批次隔离复检、暂停放行".format(names)
        else:
            conclusion = "待复核/异常比例超标（{}），建议加强复核把关并追溯原因".format(names)
    elif warning:
        conclusion = "存在关注项，建议加强人工复核与过程监控"
    else:
        conclusion = "各项关键控制点正常，本批次可放行"

    rec = QualityRecord(
        record_no=record_no,
        category=category,
        period=period,
        total=stats["total"],
        premium=stats["premium"],
        standard=stats["standard"],
        reject=stats["reject"],
        review_count=stats["review"],
        pass_rate=stats["pass_rate"],
        ccp_status=json.dumps(ccp, ensure_ascii=False),
        conclusion=conclusion,
        operator=operator or "本地用户",
    )
    db.session.add(rec)
    db.session.commit()
    return rec


def export_record_excel(record):
    """导出质检记录单为 Excel，返回 (BytesIO, 文件名)。"""
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from io import BytesIO

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "质检记录单"

    title_font = Font(name="Microsoft YaHei", bold=True, size=14, color="1a237e")
    header_font = Font(name="Microsoft YaHei", bold=True, color="FFFFFF", size=10)
    header_fill = PatternFill(start_color="1a237e", end_color="1a237e", fill_type="solid")
    data_font = Font(name="Microsoft YaHei", size=10)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin = Border(*[Side(style="thin")] * 4)

    ws.merge_cells("A1:F1")
    c = ws.cell(row=1, column=1, value="HACCP 质检记录单")
    c.font = title_font
    c.alignment = center

    flow = get_flow(record.category)
    info = [
        ("记录编号", record.record_no),
        ("品类", flow["name"]),
        ("检测日期", record.period),
        ("操作员", record.operator),
    ]
    for i, (k, v) in enumerate(info):
        ws.cell(row=3, column=i * 2 + 1, value=k).font = header_font
        ws.cell(row=3, column=i * 2 + 1).fill = header_fill
        ws.cell(row=3, column=i * 2 + 2, value=v).font = data_font

    stats = [
        ("检测数量", record.total),
        ("优质", record.premium),
        ("标准", record.standard),
        ("不合格", record.reject),
        ("待复核", record.review_count),
        ("合格率", "{:.1f}%".format(record.pass_rate * 100)),
    ]
    for i, (k, v) in enumerate(stats):
        ws.cell(row=4, column=i + 1, value="{}: {}".format(k, v)).font = data_font
        ws.cell(row=4, column=i + 1).alignment = center

    headers = ["CCP", "控制点", "危害分析", "关键限值", "监控方式", "状态"]
    r = 6
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=r, column=col, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        cell.border = thin
    try:
        ccp = json.loads(record.ccp_status or "[]")
    except Exception:
        ccp = []
    status_color = {"ok": "00ff88", "warning": "ffc107", "danger": "ff4757", "pending": "ffc107"}
    for item in ccp:
        r += 1
        values = [item.get("ccp", ""), item.get("name", ""), item.get("hazard", ""),
                  item.get("limit", ""), item.get("monitor", ""), item.get("status_label", "")]
        for col, v in enumerate(values, 1):
            cell = ws.cell(row=r, column=col, value=v)
            cell.font = data_font
            cell.alignment = center
            cell.border = thin
        color = status_color.get(item.get("status", "ok"), "00ff88")
        fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
        ws.cell(row=r, column=6).fill = fill

    r += 2
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=6)
    ws.cell(row=r, column=1, value="结论：" + record.conclusion).font = data_font
    ws.cell(row=r, column=1).alignment = Alignment(wrap_text=True, vertical="center")

    widths = [12, 16, 30, 20, 34, 10]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    filename = "HACCP_{}_{}.xlsx".format(record.record_no, record.category)
    return buf, filename