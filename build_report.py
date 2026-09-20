# -*- coding: utf-8 -*-
"""论文用《实验数据与图表汇总》生成器"""
import sys
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
import openpyxl.utils
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

ROOT = Path(r"E:\apple")
OUT = ROOT / "论文素材"
CH = OUT / "charts"
CH.mkdir(parents=True, exist_ok=True)

for fp in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"):
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

def save_fig(fig, name):
    p = CH / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p

DATA = {}
DATA["datasets"] = [
    ["外观分级-合格", "dataset_collected/apple/合格", "6,572 张(去重)", "自己拍摄+视频抽帧, 256px"],
    ["外观分级-不合格", "dataset_collected/apple/不合格", "6,405 张(去重)", "同上(腐烂/碰伤/伤口等)"],
    ["缺陷人工精标", "defect_annotate", "105 张 / 451 框", "碰伤/腐烂/疮痂/伤口/软烂/日灼"],
    ["种子人工复核", "vlm_seed_400", "400 张(335 正+65 负)", "Qwen 草稿+人工修正, 1,150 框"],
    ["教师自动标注", "teacher_autolabel_out", "6,065 张", "高置信 2,026 / 低置信 1,759 / 未检出 2,280"],
    ["缺陷训练集 v2", "defect_dataset_v2", "2,426 张(2,062 训/364 验)", "400 人工 + 2,026 高置信自动框"],
    ["分类历史集(Kaggle)", "kaggle_data/extracted", "21,971 训 / 1,779 测", "fresh/rotten 苹果"],
]
DATA["defect_eval"] = {
    "sets": ["105张人工真值\n(外部风格)", "80张自己风格\n(人工种子)"],
    "models": ["旧 YOLO11n", "v2 YOLO11n", "v2 YOLO11s", "旧+v2 融合"],
    "center_hit": [[78.0, 30.1], [49.0, 60.8], [45.5, 62.5], [82.7, 68.8]],
    "img_hit": [[103, 33], [72, 51], [72, 53], [104, 55]],
    "img_total": [105, 65],
}
DATA["by_type"] = {
    "types": ["碰伤", "腐烂", "疮痂", "伤口", "软烂", "日灼"],
    "old": [44.4, 87.3, 93.8, 84.5, 82.8, 93.8],
    "v2n": [17.8, 65.5, 46.9, 76.4, 59.5, 2.1],
    "img_old": [8, 15, 10, 25, 25, 20],
    "img_v2n": [3, 11, 7, 25, 25, 1],
    "img_n": [10, 15, 10, 25, 25, 20],
}
DATA["threshold"] = {
    "t": [0.30, 0.35, 0.40, 0.45, 0.50],
    "recall_miss": [8, 4, 3, 2, 2],
    "fp_clean": [11, 9, 6, 5, 3],
}
DATA["e2e"] = {
    "rows": [
        ["修复前(分类器覆盖 AI 证据 bug)", "29/30", "96.7", "15/30", "50.0"],
        ["修复后(缺陷证据优先)", "30/30", "100.0", "14/30", "46.7"],
        ["门槛降到 0.40 实验", "28/30", "93.3", "15/30", "50.0"],
    ],
}
DATA["vlm"] = {
    "metrics": [29.5, 50.4, 68.1, 92.4],
    "by_type_types": ["腐烂", "伤口", "软烂", "碰伤", "日灼", "疮痂"],
    "by_type_hit": [15, 25, 25, 10, 13, 9],
    "by_type_n": [15, 25, 25, 10, 20, 10],
}
DATA["apple_detect"] = [["苹果检测 YOLO11n", "0.885", "0.885", "0.933", "0.730", "验证 1,945 图 / 2,452 实例"]]
DATA["teacher"] = [["教师模型 YOLO11n(400 种子)", "0.532", "0.335", "0.309", "0.129", "80 验证图 / 176 实例"]]

# ---------- 图 ----------
x = np.arange(2); w = 0.2
fig, ax = plt.subplots(figsize=(8, 4.6))
for i, m in enumerate(DATA["defect_eval"]["models"]):
    ax.bar(x + (i - 1.5) * w, DATA["defect_eval"]["center_hit"][i], w, label=m)
ax.set_xticks(x); ax.set_xticklabels(DATA["defect_eval"]["sets"], fontsize=9)
ax.set_ylabel("中心命中率 (%)"); ax.set_title("缺陷检测模型：中心命中率对比")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
for i in range(2):
    for j in range(4):
        v = DATA["defect_eval"]["center_hit"][j][i]
        ax.text(x[i] + (j - 1.5) * w, v + 1, "%.1f" % v, ha="center", fontsize=7)
save_fig(fig, "fig1_中心命中.png")

fig, ax = plt.subplots(figsize=(8, 4.6))
for i, m in enumerate(DATA["defect_eval"]["models"]):
    vals = [DATA["defect_eval"]["img_hit"][i][k] / DATA["defect_eval"]["img_total"][k] * 100 for k in range(2)]
    ax.bar(x + (i - 1.5) * w, vals, w, label=m)
ax.set_xticks(x); ax.set_xticklabels(DATA["defect_eval"]["sets"], fontsize=9)
ax.set_ylabel("图片级命中率 (%)"); ax.set_title("至少命中一处缺陷的图片占比")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
for i in range(2):
    for j in range(4):
        v = DATA["defect_eval"]["img_hit"][j][i] / DATA["defect_eval"]["img_total"][i] * 100
        ax.text(x[i] + (j - 1.5) * w, v + 1, "%.1f" % v, ha="center", fontsize=7)
save_fig(fig, "fig2_图片级命中.png")

xx = np.arange(len(DATA["by_type"]["types"])); ww = 0.38
fig, ax = plt.subplots(figsize=(8.6, 4.6))
ax.bar(xx - ww/2, DATA["by_type"]["old"], ww, label="旧 YOLO11n")
ax.bar(xx + ww/2, DATA["by_type"]["v2n"], ww, label="v2 YOLO11n")
ax.set_xticks(xx); ax.set_xticklabels(DATA["by_type"]["types"])
ax.set_ylabel("中心命中率 (%)"); ax.set_title("不同缺陷类型的检出表现(105 张人工真值)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
save_fig(fig, "fig3_类型分解.png")

t = DATA["threshold"]["t"]
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(t, DATA["threshold"]["recall_miss"], "o-", label="明显缺陷图被救回(共14张)")
ax.plot(t, DATA["threshold"]["fp_clean"], "s--", label="干净果误出框(共30张)")
ax.set_xlabel("AI 缺陷置信度门槛"); ax.set_ylabel("张数"); ax.set_title("置信度门槛扫描")
ax.legend(); ax.grid(alpha=0.3)
save_fig(fig, "fig4_阈值扫描.png")

fig, ax = plt.subplots(figsize=(8, 4.4))
labels = ["修复前", "修复后"]
g = [96.7, 100.0]; b = [50.0, 46.7]
xx2 = np.arange(2); ww2 = 0.35
ax.bar(xx2 - ww2/2, g, ww2, label="合格池(应判合格)", color="#4ade80")
ax.bar(xx2 + ww2/2, b, ww2, label="不合格池(应判不合格)", color="#ff5d5d")
for i in range(2):
    ax.text(xx2[i]-ww2/2, g[i]+1, "%.1f" % g[i], ha="center")
    ax.text(xx2[i]+ww2/2, b[i]+1, "%.1f" % b[i], ha="center")
ax.set_xticks(xx2); ax.set_xticklabels(labels)
ax.set_ylabel("判定正确率 (%)"); ax.set_ylim(0, 115)
ax.set_title("系统端到端判定(各 30 张实测)"); ax.legend()
save_fig(fig, "fig5_端到端.png")

fig, ax = plt.subplots(figsize=(8, 4.8))
names = ["人工精标", "种子复核", "高置信自动\n(并入v2)", "低置信\n(待复核)", "未检出\n(待补)", "v2 训练集"]
vals = [105, 400, 2026, 1759, 2280, 2426]
colors = ["#8b98a5", "#60a5fa", "#4ade80", "#fbbf24", "#f87171", "#a78bfa"]
bars = ax.barh(names, vals, color=colors)
for b2, v in zip(bars, vals):
    ax.text(v + 30, b2.get_y() + b2.get_height()/2, "{:,}".format(v), va="center", fontsize=9)
ax.set_xlabel("图片数"); ax.set_title("缺陷数据积累与训练集构成")
ax.grid(axis="x", alpha=0.3)
save_fig(fig, "fig6_数据流转.png")

fig, ax = plt.subplots(figsize=(8, 4.4))
mm = ["Recall", "Precision", "中心命中", "图片级命中"]
vv = DATA["vlm"]["metrics"]
bars = ax.bar(mm, vv, color=["#60a5fa", "#fbbf24", "#4ade80", "#a78bfa"])
for b2, v in zip(bars, vv):
    ax.text(b2.get_x()+b2.get_width()/2, v+1, "%.1f" % v, ha="center")
ax.set_ylabel("%"); ax.set_ylim(0, 110)
ax.set_title("Qwen3-VL-Flash 自动预标注质量(105 张人工真值)")
ax.grid(axis="y", alpha=0.3)
save_fig(fig, "fig7_VLM.png")

figures = ["fig1_中心命中.png", "fig2_图片级命中.png", "fig3_类型分解.png",
           "fig4_阈值扫描.png", "fig5_端到端.png", "fig6_数据流转.png", "fig7_VLM.png"]

# ---------- Excel ----------
def style_sheet(ws, headers, rows, widths=None):
    head_fill = PatternFill("solid", fgColor="305496")
    for c, h in enumerate(headers, 1):
        cell = ws.cell(1, c, h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = head_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for r, row in enumerate(rows, 2):
        for c, v in enumerate(row, 1):
            cell = ws.cell(r, c, v)
            cell.alignment = Alignment(horizontal="center", vertical="center")
    if widths:
        for c, wd in enumerate(widths, 1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(c)].width = wd
    ws.freeze_panes = "A2"

wb = openpyxl.Workbook()
ws = wb.active; ws.title = "数据集统计"
style_sheet(ws, ["名称", "路径/来源", "数量", "说明"], DATA["datasets"], [18, 36, 26, 40])
ws = wb.create_sheet("缺陷模型评估")
rows = []
for j, m in enumerate(DATA["defect_eval"]["models"]):
    for k in range(2):
        rows.append([DATA["defect_eval"]["sets"][k].replace("\n", "/"), m,
                     DATA["defect_eval"]["center_hit"][j][k],
                     DATA["defect_eval"]["img_hit"][j][k], DATA["defect_eval"]["img_total"][k]])
style_sheet(ws, ["测试集", "模型", "中心命中%", "图片级命中(张)", "有缺陷图总数"], rows, [28, 16, 12, 16, 14])
ws = wb.create_sheet("类型分解")
rows = [[tp, o, v, io, iv, n] for tp, o, v, io, iv, n in zip(
    DATA["by_type"]["types"], DATA["by_type"]["old"], DATA["by_type"]["v2n"],
    DATA["by_type"]["img_old"], DATA["by_type"]["img_v2n"], DATA["by_type"]["img_n"])]
style_sheet(ws, ["类型", "旧中心命中%", "v2中心命中%", "旧图片级", "v2图片级", "总图"], rows, [12, 14, 14, 10, 10, 8])
ws = wb.create_sheet("阈值实验")
rows = list(zip(DATA["threshold"]["t"], DATA["threshold"]["recall_miss"], DATA["threshold"]["fp_clean"]))
style_sheet(ws, ["置信度门槛", "明显缺陷图救回(14中)", "干净果误出框(30中)"], rows, [14, 24, 24])
ws = wb.create_sheet("端到端判定")
style_sheet(ws, ["配置", "合格池正确", "合格池%", "不合格池正确", "不合格池%"], DATA["e2e"]["rows"], [30, 14, 12, 16, 12])
ws = wb.create_sheet("VLM预标注")
rows = [["Qwen3-VL-Flash(105人工集)", 29.5, 50.4, 68.1, 92.4, "约3~4秒/张"],
        ["GLM-4.6V-Flash", "不可用", "不可用", "不可用", "不可用", "约40秒/张"]]
style_sheet(ws, ["模型", "Recall%", "Precision%", "中心命中%", "图片级命中%", "速度"], rows, [26, 10, 12, 12, 14, 12])
ws = wb.create_sheet("其它模型")
style_sheet(ws, ["模型", "Precision", "Recall", "mAP50", "mAP50-95", "说明"],
            DATA["apple_detect"] + DATA["teacher"], [26, 12, 10, 12, 12, 32])
wb.save(OUT / "实验数据表.xlsx")

# ---------- Word ----------
doc = Document()
style = doc.styles["Normal"]
style.font.name = "微软雅黑"; style.font.size = Pt(10.5)
style.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

def P(t, bold=False, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(t); r.bold = bold; r.italic = italic
    return p

def TBL(headers, rows):
    tb = doc.add_table(rows=1, cols=len(headers)); tb.style = "Table Grid"
    for c, h in enumerate(headers):
        cell = tb.rows[0].cells[c]; cell.text = str(h)
        for par in cell.paragraphs:
            for run in par.runs:
                run.bold = True
    for row in rows:
        cells = tb.add_row().cells
        for c, v in enumerate(row):
            cells[c].text = str(v)
    return tb

def FIG(fname, cap):
    doc.add_picture(str(CH / fname), width=Inches(5.8))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P(cap, italic=True).alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_heading("苹果 AI 品控系统：实验数据与图表汇总", 0)
P("数据采集于 2026-09-03~09-04 系统实测；供毕业论文实验章节使用。")

doc.add_heading("一、数据集构成", 1)
TBL(["名称", "路径/来源", "数量", "说明"], DATA["datasets"])
FIG("fig6_数据流转.png", "图6 缺陷数据积累与训练集构成")

doc.add_heading("二、缺陷检测模型评估", 1)
rows = []
for j, m in enumerate(DATA["defect_eval"]["models"]):
    row = [m]
    for k in range(2):
        row += ["%.1f%%" % DATA["defect_eval"]["center_hit"][j][k],
                "%d/%d" % (DATA["defect_eval"]["img_hit"][j][k], DATA["defect_eval"]["img_total"][k])]
    rows.append(row)
TBL(["模型", "105人工集·中心命中", "105人工集·图片级", "自己风格80·中心命中", "自己风格80·图片级"], rows)
FIG("fig1_中心命中.png", "图1 缺陷检测模型中心命中率对比")
FIG("fig2_图片级命中.png", "图2 图片级命中率对比")
P("说明：105 张人工真值图属旧模型训练域(对其有利)；80 张自己风格验证集更接近系统真实输入。"
  "“旧+v2 融合”在两组均最优，系统因此采用双模型融合。")

doc.add_heading("三、缺陷类型表现", 1)
rows = [[tp, "%.1f%%" % o, "%.1f%%" % v, "%d/%d" % (io, n), "%d/%d" % (iv, n)]
        for tp, o, v, io, iv, n in zip(DATA["by_type"]["types"], DATA["by_type"]["old"],
                                       DATA["by_type"]["v2n"], DATA["by_type"]["img_old"],
                                       DATA["by_type"]["img_v2n"], DATA["by_type"]["img_n"])]
TBL(["类型", "旧中心命中", "v2中心命中", "旧·图片级", "v2·图片级"], rows)
FIG("fig3_类型分解.png", "图3 不同缺陷类型的检出表现")
P("v2 在自己训练域(自己风格)全面占优，但对未见过的日灼(quema)类几乎失效——训练域缺口是主要瓶颈；"
  "旧模型在外部风格图上有泛化优势，最终采用融合策略。")

doc.add_heading("四、置信度门槛分析", 1)
TBL(["置信度门槛", "明显缺陷图救回(14 张中)", "干净果误出框(30 张中)"],
    list(zip(DATA["threshold"]["t"], DATA["threshold"]["recall_miss"], DATA["threshold"]["fp_clean"])))
FIG("fig4_阈值扫描.png", "图4 置信度门槛与漏检/误报权衡")
P("缺陷模型置信度在真实缺陷与干净果上存在重叠(0.3~0.5)，降门槛会同步引入干净果误报。"
  "端到端验证表明 0.50 门槛下干净果 100% 不误报，故维持 0.50；召回提升依赖补数据重训。")

doc.add_heading("五、系统端到端判定", 1)
TBL(["配置", "合格池正确", "合格池%", "不合格池正确", "不合格池%"], DATA["e2e"]["rows"])
FIG("fig5_端到端.png", "图5 系统端到端判定准确率")
P("修复“分类器覆盖 AI 缺陷证据”逻辑后，干净合格果误报降为 0；不合格池召回受限于中/小缺陷检出能力。")

doc.add_heading("六、大模型自动预标注评估", 1)
TBL(["模型", "Recall%", "Precision%", "中心命中%", "图片级命中%", "速度"],
    [["Qwen3-VL-Flash", 29.5, 50.4, 68.1, 92.4, "约3~4秒/张"],
     ["GLM-4.6V-Flash", "不可用", "不可用", "不可用", "不可用", "约40秒/张"]])
FIG("fig7_VLM.png", "图7 大模型自动预标注质量")
P("大模型直接输出精框不可靠(Precision 50%、Recall 30%)，但作为“预标注草稿”图片级命中 92%，"
  "配合人工复核可将标注效率提升约一倍。")

doc.add_heading("七、苹果检测器与教师模型", 1)
TBL(["模型", "Precision", "Recall", "mAP50", "mAP50-95", "说明"],
    DATA["apple_detect"] + DATA["teacher"])

doc.add_heading("八、指标口径", 1)
P("· 中心命中率：人工缺陷框中心落在任一检出框内记为命中，衡量“检出位置对不对”。")
P("· 图片级命中率：有缺陷图片中至少一个缺陷被命中的比例，衡量“漏不漏”。")
P("· Recall/Precision(IoU≥0.25)：与人工框重合匹配；对“区域级粗框 vs 微型人工框”标准差异较严苛。")
P("· 端到端判定：调用系统完整流程(白平衡→苹果分割/YOLO→缺陷融合→多模型定级)，"
  "以系统最终“优质/不合格”判定与文件夹标签比对。")

doc.save(OUT / "数据图表汇总.docx")
print("已生成:")
print(" ", OUT / "数据图表汇总.docx")
print(" ", OUT / "实验数据表.xlsx")
print(" ", OUT / "charts")
