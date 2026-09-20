# -*- coding: utf-8 -*-
"""建立电子鼻实验数据处理表：record.xlsx（记录表 + 填写说明）"""
import os
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

OUT = r"E:\apple\enose\data\record.xlsx"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

HEADERS = ["样本编号", "轮次", "苹果编号", "片号", "时间点(min)", "拍照时刻", "入箱时刻",
           "采集开始", "采集结束", "室温(℃)", "湿度(%)", "硬度(kgf)", "糖度(°Brix)",
           "视觉褐变面积(%)", "等级", "备注"]

EXAMPLE = ["s001_slice_0min_a1", "R01", "A01", "A01-1", 0, "08:12:30", "08:13:10",
           "08:16:10", "08:18:10", 26.5, 52.0, 6.2, 12.5, 8.0, "fresh", "示例行，可删除"]

wb = Workbook()

# ---------- Sheet1: 记录表 ----------
ws = wb.active
ws.title = "记录表"

header_fill = PatternFill("solid", fgColor="1F4E79")
header_font = Font(name="微软雅黑", size=10, bold=True, color="FFFFFF")
body_font = Font(name="微软雅黑", size=10)
example_font = Font(name="微软雅黑", size=10, italic=True, color="808080")
thin = Side(style="thin", color="BFBFBF")
border = Border(left=thin, right=thin, top=thin, bottom=thin)
center = Alignment(horizontal="center", vertical="center", wrap_text=True)

for c, h in enumerate(HEADERS, 1):
    cell = ws.cell(row=1, column=c, value=h)
    cell.fill = header_fill; cell.font = header_font; cell.alignment = center; cell.border = border

for c, v in enumerate(EXAMPLE, 1):
    cell = ws.cell(row=2, column=c, value=v)
    cell.font = example_font; cell.alignment = center; cell.border = border

# 示例行填充浅灰，提示可删
for c in range(1, len(HEADERS) + 1):
    ws.cell(row=2, column=c).fill = PatternFill("solid", fgColor="F2F2F2")

# 预留 200 行格式（边框 + 下拉验证）
GRADE_DV = DataValidation(type="list", formula1='"fresh,mid,rotten"', allow_blank=True,
                          showErrorMessage=True, errorTitle="等级只能选", error="请输入 fresh / mid / rotten")
ws.add_data_validation(GRADE_DV)
GRADE_DV.add(f"O2:O201")

TIME_DV = DataValidation(type="list", formula1='0,30,60,120,240,360,1440', allow_blank=True,
                         showErrorMessage=True, errorTitle="时间点只能选", error="请输入方案中的时间点")
ws.add_data_validation(TIME_DV)
TIME_DV.add(f"E2:E201")

for r in range(3, 202):
    for c in range(1, len(HEADERS) + 1):
        cell = ws.cell(row=r, column=c)
        cell.font = body_font; cell.border = border; cell.alignment = center

# 列宽
widths = [24, 7, 10, 9, 12, 10, 10, 10, 10, 8, 8, 9, 10, 15, 9, 16]
for i, w in enumerate(widths, 1):
    ws.column_dimensions[get_column_letter(i)].width = w
ws.row_dimensions[1].height = 28
ws.freeze_panes = "A2"
ws.auto_filter.ref = "A1:P201"

# ---------- Sheet2: 填写说明 ----------
ws2 = wb.create_sheet("填写说明")
notes = [
    ["电子鼻实验记录表 · 填写说明（泡沫箱法 V1.1）"],
    [""],
    ["1. 样本编号：s{流水号}_slice_{时间点}_{片号}，如 s001_slice_0min_a1。流水号按测量顺序连续编（每轮 s001~s015；次日过夜档 s016 起）。必须与 CSV 文件名一致（不带 .csv），merge_minutes.py 靠它自动合并。"],
    ["2. 轮次 / 苹果编号 / 片号：如 R01 / A01 / A01-1。同一苹果的多片作平行样本。"],
    ["3. 时间点(min)：0 / 30 / 60 / 120 / 240 / 360 / 1440（下拉选择）。1440 = 过夜 24 h 档。"],
    ["4. 拍照时刻 / 入箱时刻：填 HH:MM:SS。入箱 − 拍照 ≤ 2 分钟，超时的样本在备注写「间隔超标」，融合时剔除。"],
    ["5. 采集开始 / 结束：对应 CSV 起止时间，用于核对采集时长 ≥ 2 分钟。"],
    ["6. 室温 / 湿度：BME680 已自动写入 CSV，这里记环境参考值即可。"],
    ["7. 硬度 / 糖度：可选。硬度计去皮赤道部两处取均值（kgf）；糖度计滴汁测（°Brix）。"],
    ["8. 视觉褐变面积(%)：用系统「切片检测」页上传照片自动计算，这是定级主指标。"],
    ["9. 等级：fresh / mid / rotten（下拉选择）。定级规则：褐变 <15% 且硬度≥5.5 = fresh；15~40% = mid；≥40% 或有酒味/酸味/渗液 = rotten。等级就是训练标签，务必与 CSV 的 label 列一致。"],
    ["10. 备注：异常、重测、剔除标记（如：剔除-999、重测、间隔超标）。"],
    [""],
    ["每轮结束必做：运行 check_raw.py 核对 raw 目录与记录表是否一一对应。"],
]
for r, row in enumerate(notes, 1):
    for c, v in enumerate(row, 1):
        cell = ws2.cell(row=r, column=c, value=v)
        if r == 1:
            cell.font = Font(name="微软雅黑", size=13, bold=True, color="1F4E79")
        else:
            cell.font = Font(name="微软雅黑", size=10.5)
        cell.alignment = Alignment(vertical="top", wrap_text=True)
ws2.column_dimensions["A"].width = 110

wb.save(OUT)
print("saved:", OUT)