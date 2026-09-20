# -*- coding: utf-8 -*-
"""生成《智果AI品控系统学习地图（精炼版）》docx"""
import sys
from docx import Document
from docx.shared import Pt, Twips, Cm, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = r"E:\apple\docs\智果AI品控系统学习地图.docx"
EA = "微软雅黑"
H1C = "2E74B5"; H2C = "2E74B5"; H3C = "1F4D78"
HDR_FILL = "E8EEF5"

doc = Document()

# ---- page: A4, 1 inch margins ----
sec = doc.sections[0]
sec.page_width = Twips(11906); sec.page_height = Twips(16838)
sec.left_margin = sec.right_margin = Twips(1440)
sec.top_margin = sec.bottom_margin = Twips(1440)

# ---- Normal style ----
n = doc.styles["Normal"]
n.font.name = "Calibri"; n.font.size = Pt(11)
n.element.rPr.rFonts.set(qn("w:eastAsia"), EA)
n.paragraph_format.space_before = Pt(0); n.paragraph_format.space_after = Pt(6)
n.paragraph_format.line_spacing = 1.25

def style_heading(name, size, color, before, after):
    s = doc.styles[name]
    s.font.name = "Calibri"; s.font.size = Pt(size); s.font.bold = True
    s.font.color.rgb = RGBColor.from_string(color)
    s.element.rPr.rFonts.set(qn("w:eastAsia"), EA)
    s.paragraph_format.space_before = Pt(before); s.paragraph_format.space_after = Pt(after)

style_heading("Heading 1", 16, H1C, 18, 10)
style_heading("Heading 2", 13, H2C, 14, 7)
style_heading("Heading 3", 12, H3C, 10, 5)

def para(text, bold=False, size=None, color=None, after=6):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    if size: r.font.size = Pt(size)
    if color: r.font.color.rgb = RGBColor.from_string(color)
    p.paragraph_format.space_after = Pt(after)
    return p

def bullet(text, bold_head=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_head:
        r = p.add_run(bold_head); r.bold = True
        p.add_run(text)
    else:
        p.add_run(text)
    p.paragraph_format.space_after = Pt(4)
    return p

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def fix_layout(t, widths_cm):
    tblPr = t._tbl.tblPr
    layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed"); tblPr.append(layout)
    grid = t._tbl.find(qn("w:tblGrid"))
    if grid is not None:
        for gc, w in zip(grid.findall(qn("w:gridCol")), widths_cm):
            gc.set(qn("w:w"), str(int(w * 566.93)))
def add_table(headers, rows, widths_cm):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    total_dxa = int(sum(widths_cm) * 566.93)
    tblPr = t._tbl.tblPr
    tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), str(total_dxa)); tw.set(qn("w:type"), "dxa"); tblPr.append(tw)
    ind = OxmlElement("w:tblInd"); ind.set(qn("w:w"), "120"); ind.set(qn("w:type"), "dxa"); tblPr.append(ind)
    mar = OxmlElement("w:tblCellMar")
    for tag, val in (("top", 80), ("start", 120), ("bottom", 80), ("end", 120)):
        el = OxmlElement("w:" + tag); el.set(qn("w:w"), str(val)); el.set(qn("w:type"), "dxa"); mar.append(el)
    tblPr.append(mar)
    for j, h in enumerate(headers):
        c = t.rows[0].cells[j]
        c.width = Cm(widths_cm[j])
        shade(c, HDR_FILL)
        p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h); r.bold = True; r.font.size = Pt(10.5)
    for row in rows:
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].width = Cm(widths_cm[j])
            p = cells[j].paragraphs[0]; p.paragraph_format.space_after = Pt(2)
            r = p.add_run(val); r.font.size = Pt(10.5)
    fix_layout(t, widths_cm)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

# ================= 内容 =================
para("智果·AI 品控系统 — 学习地图（精炼版）", bold=True, size=22, color="1F4D78", after=4)
para("只列该学的：能讲懂自己的系统、能答辩、能写论文、能打比赛。其余一律不学。", size=11, color="595959", after=12)

doc.add_heading("一、判断标准（先记住这一句）", level=1)
para("凡是不能帮你\u201c讲懂系统、写好论文、应付答辩\u201d的知识，一律不学。下面所有内容都按这个标准筛过。")

doc.add_heading("二、层次说明（学到什么程度）", level=1)
add_table(
    ["层次", "含义", "本科目标"],
    [
        ["L1 了解", "能用自己的话讲清\u201c是什么、有什么用\u201d（答辩能说）", "第二档知识到此即可"],
        ["L2 会用", "能照着操作、会改参数、能跑通", "第一档知识的目标"],
        ["L3 理解", "懂原理、能独立排错、能改代码", "个别重点项到L3即可"],
        ["L4 掌握", "能独立实现、从零写出来", "不追求，时间不划算"],
    ],
    [2.6, 9.6, 3.7],
)

doc.add_heading("三、必须学（第一档，学完就能看懂系统）", level=1)
para("目标：L2（会讲+会操作）。这 6 块是你理解系统的地基，暑假就能搞定。")
add_table(
    ["知识点", "一句话解释（大白话）", "程度", "对应系统位置"],
    [
        ["Python：读CSV、列表循环、画图", "用代码处理数据表、把数据画成图（matplotlib）", "L2 会用", "collect.py、训练脚本、output图表"],
        ["图像三件套：像素、RGB/HSV、阈值分割", "图片=像素格子；HSV把颜色拆成\u201c色相/深浅/亮暗\u201d；阈值分割=按颜色范围圈出苹果", "L2 会讲会调参", "apple_defect_detector.py 的 segment_apple"],
        ["机器学习四张表：训练集/测试集/准确率/混淆矩阵", "训练=让模型看数据学规律；测试=拿新数据考它；混淆矩阵=看它错在哪一类", "L2 会讲", "scripts/evaluate_models.py"],
        ["传感器三句话：MQ原理/ADC/为什么要预热和基线", "MQ=加热丝+敏感层，遇气体电阻变；ADC=模拟电压转数字；预热+基线=读数稳定可比", "L2 会讲", "enose/enose_sketch/enose_sketch.ino"],
        ["电子鼻数据流", "ESP32每秒采一行→串口→collect.py存CSV→训练→预测", "L2 会跑通", "enose/ 整个目录"],
        ["统计三件套：均值/标准差/p值", "p<0.05=差异不是碰巧，是统计学\u201c及格线\u201d", "L2 会用", "显著性分析页"],
    ],
    [3.2, 7.0, 2.3, 3.4],
)

doc.add_heading("四、重点学（第二档，论文答辩的\u201c技术含量\u201d）", level=1)
para("目标：L1~L2（会讲、会看结果）。答辩时老师主要问这些。")
add_table(
    ["知识点", "一句话解释", "程度"],
    [
        ["SVM / 随机森林怎么理解", "SVM=画一条分界线把好果坏果分开；RF=一群\u201c小决策树\u201d投票", "L1 会讲"],
        ["CNN 直觉", "卷积=小窗口扫图找特征：边缘→纹理→局部形状→整体，层越多越抽象", "L1 会讲"],
        ["YOLO 是干什么的", "分类=整张图打一个标签；检测=还要把每个苹果\u201c框\u201d出来", "L1 会讲"],
        ["交叉验证 + AUC", "CV=把数据分几份轮流当考卷，评估更公平；AUC=模型区分好坏的能力值", "L1~L2 会用"],
        ["实验设计三原则", "对照（有空白组）、重复（每组多测几次）、随机（不挑着测）", "L2 写方案会用"],
        ["PCA 降维（了解）", "十几列数据压成两三个\u201c综合指标\u201d，方便画图看分组", "L1 知道干嘛"],
    ],
    [4.4, 8.6, 2.9],
)

doc.add_heading("五、按需学（用到再补，别提前学）", level=1)
add_table(
    ["知识点", "什么时候补", "怎么补"],
    [
        ["Grad-CAM 热力图原理", "答辩被问\u201c可视化怎么做的\u201d", "讲清\u201c看模型盯着图片哪里判断\u201d即可"],
        ["Flask 前后端怎么通信", "想讲平台架构（前端点按钮→后端处理→返回）", "了解概念即可"],
        ["显著性分析实操", "写论文做数据表时", "用 Python/SPSS 算出p值"],
        ["嵌入式改引脚/采样率", "要调整电子鼻设备时", "会改 enose_sketch.ino 对应几行"],
    ],
    [4.2, 6.2, 5.5],
)

doc.add_heading("六、不要学（省时间的清单）", level=1)
add_table(
    ["不要学", "为什么"],
    [
        ["Python 高级：面向对象/装饰器/多线程", "你不写库、不做大规模开发，用不上"],
        ["算法与数据结构、刷题", "除非你想转码当程序员"],
        ["深度学习数学推导（反向传播公式等）", "会讲概念就够答辩，推导是研究生的事"],
        ["前端框架、CSS 美化", "系统界面已经做好，不用改样式"],
        ["数据库设计/索引优化", "系统用 SQLite 一个文件，够用"],
        ["嵌入式底层：RTOS/驱动/PCB/电路", "你是使用者不是开发者"],
        ["高数/线代/概率", "除非考研要考——那是考试需要，跟系统无关，两码事"],
    ],
    [6.2, 9.7],
)

doc.add_heading("七、学习顺序（照着做，别贪多）", level=1)
add_table(
    ["阶段", "学什么"],
    [
        ["第1步：暑假 2~3 周（每天1~2小时）", "Python读CSV+画图 → 图像三件套 → 机器学习四张表 → 传感器三句话。学完就能对着代码讲系统"],
        ["第2步：大三上（跟课程走）", "统计、实验设计课上认真听；同时把系统包装去参赛/进实验室，边用边补"],
        ["第3步：大三下~大四（写论文时）", "按需补显著性分析、Grad-CAM；遇到新概念先问\u201c答辩会不会被问、论文用不用得上\u201d，用不上就不学"],
    ],
    [4.6, 11.3],
)

para("")
para("一句话收尾：你的目标是\u201c懂原理、会操作、能答辩\u201d，不是\u201c全都会\u201d。按这张地图走，大四答辩时你就是最了解自己系统的那个。", bold=True, color="1F4D78")

doc.core_properties.title = "智果AI品控系统学习地图（精炼版）"
doc.save(OUT)
print("SAVED:", OUT)