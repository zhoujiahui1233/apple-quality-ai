# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm
from docx.oxml.ns import qn
sys.stdout.reconfigure(encoding="utf-8")

def ps(p, size=12):
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(5)

def mk(anchor, text, bold=False, size=12, indent=True, font="宋体"):
    p = anchor.insert_paragraph_before()
    ps(p)
    if indent: p.paragraph_format.first_line_indent = Cm(0.74)
    r = p.add_run(text); r.bold = bold; r.font.size = Pt(size)
    r.font.name = "Times New Roman"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    return p

def h3(anchor, text):
    return mk(anchor, text, bold=True, size=12, indent=False, font="黑体")

def tbl(doc, anchor, headers, rows, fs=10, caption=None):
    if caption:
        mk(anchor, caption, bold=True, size=fs, indent=False, font="黑体")
    tb = doc.add_table(rows=1, cols=len(headers)); tb.style = "Table Grid"
    for c, hh in enumerate(headers):
        cell = tb.rows[0].cells[c]; cell.text = str(hh)
        for par in cell.paragraphs:
            for run in par.runs:
                run.bold = True; run.font.size = Pt(fs)
                run.font.name = "Times New Roman"
                run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    for row in rows:
        cells = tb.add_row().cells
        for c, v in enumerate(row):
            cells[c].text = str(v)
            for par in cells[c].paragraphs:
                for run in par.runs:
                    run.font.size = Pt(fs)
                    run.font.name = "Times New Roman"
                    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    anchor._p.addprevious(tb._tbl)
    return tb

def find_start(doc, prefix):
    for p in doc.paragraphs:
        if p.text.startswith(prefix):
            return p
    raise KeyError(prefix)

# ========= 论文（食品方向版） =========
tp = Path(r"E:\apple\论文初稿_食品方向版.docx")
doc = Document(str(tp))
a = find_start(doc, "4.7　多模态大模型辅助标注评估")
h3(a, "4.6.2　模型对比与评估口径的说明")
mk(a, "本文涉及两类“模型对比”：一是同一任务上的算法对比，二是不同原理分类器之间的横向比较。为避免产生误导，"
     "对各模型采用与其训练数据、验证方式相匹配的评估口径，并在口径一致的前提下进行解释：苹果检测器与缺陷检测模型以检测指标"
     "（mAP、中心命中率、图片级命中率）评估；SVM 与随机森林以 5 折交叉验证评估（表 4-6a）；DL 与 YOLO 分类模型以独立验证集评估。"
     "各口径的差异已在表 4-6a 注明，读者不应直接比较不同行的绝对值。")
mk(a, "需要说明的是，直接在同一批图像上比较四类分级模型存在口径障碍：SVM/RF 采用手工特征与整图分类，DL 与 YOLO 分类模型的"
     "训练数据与标签口径并不一致，逐模型在异质样本上的同图输出会放大各自训练域的偏差，难以得到有统计意义的横向结论。"
     "因此本文采取“分口径评估 + 端到端系统验证 + 案例核查”的组合策略，既保证每个数字可复现，又用系统整体判定检验融合逻辑"
     "是否优于单一模型。")
h3(a, "4.6.3　模型判定与感官/理化指标的融合评价框架（设计）")
mk(a, "为使“系统判定”与食品品质语言衔接，本文设计了如下融合评价框架（正式实验待补充，不构成已完成结果的宣称）：")
tbl(doc, a, ["系统/仪器指标", "对应食品指标", "融合分析"],
    [["图像缺陷占比、着色/颜色统计", "褐变指数 BI、色差 L*a*b*、失重率", "相关性 + 回归预测"],
     ["电子鼻气体响应（乙醇/TVOC 等）", "TSS、可滴定酸、pH、发酵相关指标", "响应特征 → 品质状态判别"],
     ["系统“合格/标准/不合格”判定", "感官等级（规范量表评定）", "Kappa 一致性检验"],
     ["综合仪器指标", "感官接受度/货架期分组", "多指标联合判别/排序"]],
    caption="表4-6b 模型-感官-理化融合评价指标对应关系")
mk(a, "采样与统计流程：同一编号样品先完成图像与电子鼻无损采集，再按破坏性测定顺序完成理化与感官记录，保证样本严格配对；"
     "采用描述统计与正态性检验、Pearson/Spearman 相关、回归预测（报告 R²/RMSE）、ANOVA 与 Tukey 多重比较、以及 Kappa 一致性"
     "检验。该框架用于回答三个问题：系统指标能否反映真实品质变化；系统分级与人工感官是否一致；仪器能否作为感官/理化的"
     "快速替代手段。")
doc.save(str(tp))
print("论文已补 4.6.2/4.6.3")

# ========= 答辩手册 =========
hp = Path(r"E:\apple\答辩准备手册.docx")
hd = Document(str(hp))
a2 = find_start(hd, "第七部分　送命题应对表")
mk(a2, "问：你的模型之间到底谁好？为什么没有一张同图对比表？", bold=True, size=12, indent=False, font="黑体")
mk(a2, "答：模型各有适用的评估口径——SVM/随机森林用 5 折交叉验证（74.1%/79.3%），DL 与 YOLO 分类用独立验证集（DL 98.5%），"
     "苹果检测用 mAP。它们训练数据和标签口径不同，直接同图横向比较会放大各自偏差、反而误导，所以我采用“分口径评估+端到端验证+案例核查”，"
     "并明确在论文表注里说明不能直接比绝对值。")
mk(a2, "问：系统判定和感官评价怎么挂钩？", bold=True, size=12, indent=False, font="黑体")
mk(a2, "答：我设计了融合评价框架：图像/电子鼻指标对应褐变指数、色差、失重、TSS 等理化指标与感官等级，同一编号样品无损检测后做破坏性"
     "测定，用相关、回归与 Kappa 检验一致性，回答“仪器能否作为感官/理化的快速替代”。该实验按方案执行中，属后续工作，我不宣称已完成。")
hd.save(str(hp))
print("答辩手册已补两问")
