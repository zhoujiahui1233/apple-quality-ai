# -*- coding: utf-8 -*-
"""生成《智果AI品控系统学习手册（详细版）》docx"""
import sys
from docx import Document
from docx.shared import Pt, Twips, Cm, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = r"E:\apple\docs\智果AI品控系统学习手册.docx"
EA = "微软雅黑"
H1C = "2E74B5"; H2C = "2E74B5"; H3C = "1F4D78"
HDR_FILL = "E8EEF5"

doc = Document()
sec = doc.sections[0]
sec.page_width = Twips(11906); sec.page_height = Twips(16838)
sec.left_margin = sec.right_margin = Twips(1440)
sec.top_margin = sec.bottom_margin = Twips(1440)

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

def para(text, bold=False, size=None, color=None, after=6, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold; r.italic = italic
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

def add_table(headers, rows, widths_cm, font_size=10.5):
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
        c = t.rows[0].cells[j]; c.width = Cm(widths_cm[j]); shade(c, HDR_FILL)
        p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(2)
        r = p.add_run(h); r.bold = True; r.font.size = Pt(font_size)
    for row in rows:
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].width = Cm(widths_cm[j])
            p = cells[j].paragraphs[0]; p.paragraph_format.space_after = Pt(2)
            r = p.add_run(val); r.font.size = Pt(font_size)
    fix_layout(t, widths_cm)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

def block(num, title, what, where, exercises, check):
    doc.add_heading("%s %s" % (num, title), level=2)
    doc.add_heading("这是什么（大白话）", level=3)
    for line in what:
        para(line)
    doc.add_heading("在系统里对应哪里", level=3)
    for w in where:
        bullet(w)
    doc.add_heading("要做的练习", level=3)
    add_table(["练习", "具体做什么", "约用时"], exercises, [3.2, 9.4, 3.3])
    doc.add_heading("学会的标准（自测）", level=3)
    for c in check:
        bullet(c)

# ================= 封面 =================
para("智果·AI 品控系统 — 学习手册", bold=True, size=22, color="1F4D78", after=4)
para("把系统里的知识讲透：是什么、在哪、怎么练、算学会的标准。只讲有用的。", size=11, color="595959", after=12)

doc.add_heading("怎么用这份手册", level=1)
para("每个知识点都按同一套结构写：这是什么（大白话）→ 在系统里对应哪里 → 要做的练习 → 学会的标准。")
para("推荐的四步学习循环（每次都这么学，效率最高）：", bold=True)
bullet("读概念：先读本文档的\u201c这是什么\u201d，不懂的术语标个问号跳过，别死磕", bold_head="第1步 ")
bullet("找实物：去\u201c对应哪里\u201d打开真代码/真数据看一眼，对着实物学比看书快十倍", bold_head="第2步 ")
bullet("动手练：做\u201c要做的练习\u201d，出错=学到东西", bold_head="第3步 ")
bullet("复述自测：能对着\u201c学会的标准\u201d用自己的话讲清楚，才算真会（费曼技巧）", bold_head="第4步 ")

doc.add_heading("一、层次说明（学到什么程度）", level=1)
add_table(["层次", "含义", "本科目标"],
    [["L1 了解", "能用自己的话讲清\u201c是什么、有什么用\u201d（答辩能说）", "第二档知识到此即可"],
     ["L2 会用", "能照着操作、会改参数、能跑通", "第一档知识的目标"],
     ["L3 理解", "懂原理、能独立排错、能改代码", "个别重点项到L3即可"],
     ["L4 掌握", "能独立实现、从零写出来", "不追求，时间不划算"]],
    [2.6, 9.6, 3.7])

doc.add_heading("二、必须学（第一档：学完就能看懂系统）", level=1)
para("目标 L2。这 6 块是地基，第 1~4 周完成。")

block("1", "Python：读CSV、列表、循环、画图",
    ["CSV 就是最常见的数据表格文件：每行一个采样点，逗号分隔。你的电子鼻数据、历史数据都是这种格式。",
     "Python 你只需要会 4 件事：读文件（打开CSV）、列表（装数据的箱子）、for循环（把列表里每个东西依次处理一遍）、画图（matplotlib 把数据变成曲线图）。",
     "记住最常用的句式：for 变量 in 列表: 意思是\u201c把列表里每个东西依次拿出来处理一遍\u201d。",
     "类、面向对象、装饰器、多线程——全都不用学，那不是你需要的 Python。"],
    ["collect.py：从串口读数据写 CSV（看它是怎么逐行写入的）",
     "enose\\data\\raw\\ 里你采过的数据文件（真实 CSV）",
     "scripts\\evaluate_models.py：读 CSV 算准确率的地方",
     "output\\ 目录里画出来的图表"],
    [["练习1", "用 Python 打开一个电子鼻 CSV，打印前 5 行", "30分钟"],
     ["练习2", "把 MQ3 那一列画成折线图（横轴=时间）", "1小时"],
     ["练习3", "算某一列的均值、最大值、最小值", "30分钟"],
     ["练习4", "分别打印\u201c好果\u201d和\u201c坏果\u201d两组数据的均值", "1小时"]],
    ["能指着 collect.py 说出\u201c这一行在干嘛\u201d",
     "能独立写出\u201c读CSV→画图\u201d的 20 行脚本",
     "能解释 for 循环为什么能处理一整列数据"])

block("2", "图像三件套：像素、RGB/HSV、阈值分割",
    ["一张图片 = 很多\u201c小格子\u201d（像素），每个格子存一个颜色。图越清晰=格子越多。",
     "RGB = 红绿蓝三个通道，每个 0~255，混合出所有颜色。",
     "HSV = 把颜色拆成三维：色相（到底是什么颜色）、饱和度（颜色多浓）、明度（多亮）。",
     "为什么苹果分割用 HSV 不用 RGB？因为\u201c苹果的红\u201d用 RGB 很难圈（受光照影响大），但用 HSV 的色相范围一划就出来，而且光照变化主要在明度那一维，不影响色相判断。",
     "阈值分割 = 按颜色范围把图分成\u201c是苹果/不是苹果\u201d两块。你的系统流程：先用 HSV 范围圈出苹果区域，再在苹果区域内找颜色异常（缺陷）。"],
    ["apple_defect_detector.py 的 segment_apple()（找苹果）和 detect_defects()（找缺陷）",
     "config.py 的 APPLE_HSV_RANGES（颜色范围就调这里）",
     "网页上传图片后的分割可视化"],
    [["练习1", "上传一张苹果图，观察分割结果", "10分钟"],
     ["练习2", "打开 config.py，把红色范围改小，再测一张，看分割少了什么", "30分钟"],
     ["练习3", "用 Python 读一张图转成 HSV，单独显示 H（色相）通道", "1小时"]],
    ["能解释\u201c为什么苹果分割用HSV而不是RGB\u201d",
     "能说出\u201c阈值分割=按颜色范围圈区域\u201d",
     "调过 config.py 并看懂分割前后的变化"])

block("3", "机器学习四张表：训练集/测试集/准确率/混淆矩阵",
    ["机器学习 = 让程序从数据里自己找规律，而不是人写规则。你给它一堆\u201c图片+好坏标签\u201d，它自己总结\u201c什么样算好果\u201d。",
     "训练集 = 用来学规律的样本；测试集 = 学完后拿来考试的\u201c没见过的新样本\u201d。两者绝对不能混用，否则等于考试作弊，准确率虚高——这是答辩必考点。",
     "准确率 = 答对的百分比。",
     "混淆矩阵 = 一张表：行是真实类别，列是预测类别。它比一个准确率数字信息量大得多——能看清模型\u201c把哪类错认成了哪类\u201d。",
     "重点看一个格子：\u201c把坏果判成好果\u201d（漏检）。漏检比误报严重得多：坏果混进好果里直接卖出去。答辩时能说出这句，很加分。"],
    ["scripts\\evaluate_models.py：评估工具，跑一次就出准确率+混淆矩阵",
     "output\\evaluation\\ 里生成的混淆矩阵图",
     "网页\u201c模型管理/评估\u201d相关页面"],
    [["练习1", "运行 python scripts/evaluate_models.py --mode test，看输出", "30分钟"],
     ["练习2", "找到混淆矩阵里\u201c漏检\u201d那一格，说出为什么它最危险", "20分钟"],
     ["练习3", "用自己的话写三句话：训练集、测试集、准确率各是什么", "20分钟"]],
    ["能说出训练/测试/准确率/混淆矩阵分别是什么",
     "能看懂评估报告里的每个数字",
     "能解释\u201c为什么测试集必须是模型没见过的\u201d和\u201c漏检比误报严重\u201d"])

block("4", "传感器三句话：MQ原理/ADC/预热与基线",
    ["MQ 传感器内部 = 一根加热丝 + 一层敏感材料。敏感材料遇到目标气体会\u201c改变电阻\u201d，气体越多电阻变化越大。所以它测的不是浓度绝对值，而是\u201c电阻变化\u201d——必须跟基线比才有意义。",
     "ADC（模数转换器）= 把\u201c电压这种连续量\u201d变成\u201c数字\u201d，电脑才读得懂。ESP32 自带的 ADC 精度差，所以外接了 ADS1115（16位，读数 0~32767）。",
     "预热 = MQ 的加热丝要烧到工作温度读数才稳定（方案要求 30 分钟）。基线 = 没放样品时的读数；\u201c样品读数 − 基线\u201d才是传感器对样品的真实响应。",
     "湿度也会影响 MQ 读数，所以你的 CSV 里专门记了 humPct（湿度）——分析时用湿度列做校正，这是论文里\u201c环境因素校正\u201d的来源。"],
    ["enose\\enose_sketch\\enose_sketch.ino：固件（读ADS1115那几行，返回0~32767）",
     "enose\\collect.py：电脑端采集",
     "protocol_box.md：预热/基线流程就在方案里"],
    [["练习1", "打开固件，找到读 ADS1115 的代码，说出读数范围", "20分钟"],
     ["练习2", "打开你采过的 CSV，把 MQ3 列画出来，看放样品后有没有明显变化", "1小时"],
     ["练习3", "对比一次空箱基线和一次放样品的曲线，说出差在哪", "30分钟"]],
    ["能说清 MQ 为什么预热、为什么测基线",
     "能解释 ADS1115 是干嘛的、为什么不用 ESP32 自带 ADC",
     "知道湿度列为什么要记录"])

block("5", "电子鼻数据流：从传感器到预测",
    ["整条链路四步：ESP32 每秒把 11 个读数发到串口 → collect.py 收下存成 CSV（并打标签）→ train.py 用这批 CSV 训练分类模型 → predict_freshness.py 拿新数据预测新鲜度。",
     "你的\u201c电子鼻\u201d不是一个传感器，而是一套\u201c采集+存储+训练+预测\u201d的流程。数据是核心资产，模型只是从数据里学出来的结果。",
     "标签（label）很关键：模型靠\u201c这个气味=新鲜\u201d这样的标签才能学。采集时 --label 参数就是干这个的。"],
    ["enose\\collect.py（采集）→ enose\\data\\raw\\（CSV）→ enose\\train.py（训练）→ enose\\models\\（模型）→ enose\\predict_freshness.py（预测）",
     "enose\\README.md：整套流程说明"],
    [["练习1", "把装置接上，采 2 分钟数据（--label test1）", "30分钟"],
     ["练习2", "用 train.py 对已有数据训练一次", "1小时"],
     ["练习3", "用 predict_freshness.py 对一条新数据做预测，看输出", "30分钟"]],
    ["能不看文档说出\u201c采集→存CSV→训练→预测\u201d四步",
     "能解释标签为什么重要"])

block("6", "统计三件套：均值/标准差/p值",
    ["均值 = 一组数的平均水平。标准差 = 这组数\u201c有多散\u201d（传感器读数抖不抖，看它）。",
     "p 值 = 判断\u201c两组数据的差异是不是碰巧\u201d的数值。p<0.05 一般认为差异显著（不是巧合）；p>0.05 则不能下结论。",
     "论文里\u201c腐败组与新鲜组差异显著（p<0.05）\u201d这句话就是这么来的——这是食品论文的标配说法。"],
    ["系统\u201c显著性分析\u201d页面",
     "你实验数据里的新鲜组/腐败组 CSV"],
    [["练习1", "用 Python 对新鲜组和腐败组的 MQ3 数据做 t 检验，输出 p 值", "1小时"],
     ["练习2", "把结果和系统显著性页面对比，理解 p<0.05 的含义", "30分钟"]],
    ["能解释标准差和 p 值各回答什么问题",
     "能自己跑一次 t 检验并读懂输出"])

doc.add_heading("三、重点学（第二档：论文答辩的技术含量）", level=1)
para("目标 L1~L2（会讲、会看结果）。第 5~7 周完成。")

block("7", "SVM 和随机森林怎么理解",
    ["SVM = 在\u201c好果/坏果\u201d两类点之间画一条最宽的分界线，新样本落在哪边就归哪类。想象两类人中间划一条线，线离两边都尽量远，最不容易误判。",
     "随机森林 = 种很多棵\u201c决策树\u201d（每棵都是\u201c如果这个特征>某值就…\u201d的判断链），大家投票决定，少数服从多数，比单棵树稳。",
     "你的系统里 SVM 用 170 维特征、RF 会做特征筛选——这些细节不用深究，答辩时能讲\u201c分界线\u201d和\u201c投票\u201d两个比喻就够。"],
    ["train_ml_v2.py / train_ml_v3.py：训练脚本",
     "apple_svm_model.dat / apple_rf_model.pkl：训练好的模型",
     "网页四个模型按钮：SVM / 随机森林"],
    [["练习1", "跑评估工具，对比 SVM 和 RF 的准确率、混淆矩阵", "1小时"],
     ["练习2", "说出 RF 比 SVM 好在哪（或差在哪），用数据说话", "30分钟"]],
    ["能用\u201c画分界线\u201d解释SVM、用\u201c决策树投票\u201d解释RF"])

block("8", "CNN（深度学习）直觉",
    ["CNN = 用很多\u201c小窗口\u201d在图片上滑动，一层层学特征：第一层学边缘、第二层学纹理、第三层学局部形状、更深层学整体轮廓。",
     "所以 CNN 能自动学特征，不用人手工设计——这正是它和 SVM 的最大区别（SVM 需要人事先算好特征）。",
     "你的 DL 模型就是 CNN（apple_dl_model.pth）。Grad-CAM 热力图是\u201c回看模型到底盯着图片哪里判断\u201d的工具，答辩被问可视化时用得上。"],
    ["train_dl_model.py：训练脚本；apple_dl_model.pth：模型文件",
     "网页 DL 模型按钮 + 热力图可视化"],
    [["练习1", "上传一张图切到 DL 模型，看热力图，说出它主要看苹果哪个区域", "30分钟"]],
    ["能解释\u201cCNN逐层学特征\u201d和\u201c为什么比手工特征省事\u201d"])

block("9", "YOLO：分类 vs 检测",
    ["分类 = 整张图给一个结论（\u201c这是好苹果\u201d）。检测 = 还要在图上把每个苹果\u201c框\u201d出来（\u201c这里有3个苹果，位置在这里\u201d）。",
     "YOLO = \u201c看一眼就同时给出框和类别\u201d的检测算法，速度快，适合实时场景。",
     "你系统里：四个模型按钮中的\u201cYOLO\u201d是检测（框苹果+框缺陷）；另外还有一个 YOLO 分类模型（判等级）。答辩时能分清楚这两者。"],
    ["yolo_detect_best.pt：检测模型",
     "网页\u201cYOLO\u201d按钮（检测框）"],
    [["练习1", "上传一张有多个苹果的图，用 YOLO 检测，观察框的数量和位置", "20分钟"]],
    ["能说清\u201c分类和检测的区别\u201d，能指出系统里哪些是分类、哪些是检测"])

block("10", "交叉验证 + AUC",
    ["交叉验证 = 把数据分成几份，轮流拿一份当考卷、其余当课本，来回考几次取平均——比\u201c只考一次\u201d的分数可信得多，也能看出模型稳不稳。",
     "AUC = 模型区分好坏的\u201c综合能力分\u201d：0.5 = 瞎猜，越接近 1 越强。比较两个模型，AUC 比单个准确率更公平。",
     "评估输出里的\u201c均值±标准差\u201d（如 0.52±0.13）就是交叉验证的结果：均值看水平，标准差看稳定。"],
    ["scripts\\evaluate_models.py 的 --mode cv（交叉验证）",
     "output\\evaluation\\ 里的 ROC/AUC 曲线图"],
    [["练习1", "运行 python scripts/evaluate_models.py --mode cv --folds 5，看输出", "1小时"]],
    ["能解释\u201c为什么交叉验证比单次划分公平\u201d，能看懂 ROC/AUC 图"])

block("11", "实验设计三原则：对照/重复/随机",
    ["对照：必须有\u201c没处理的空白组\u201d，才能证明变化是样品引起的。你的\u201c空箱基线\u201d就是对照。",
     "重复：每组多测几遍，抵消偶然误差（测3个苹果不如测15个可信）。",
     "随机：测试顺序不能\u201c先测新买的\u201d，要随机，否则结果被时间因素污染（比如放着放着苹果自己变熟了）。",
     "这三个词是论文\u201c实验设计\u201d章节的骨架，必须会讲。"],
    ["protocol_box.md / protocol_slice.md：你的实验方案"],
    [["练习1", "对照你的实验方案，找出\u201c对照/重复/随机\u201d分别体现在哪一步", "30分钟"]],
    ["能指出自己方案里三原则各在哪里"])

block("12", "PCA 降维（了解即可）",
    ["你的电子鼻一次记 11 列数据，太多、看不清。PCA 把多列压缩成 2~3 个\u201c综合指标\u201d，损失一点信息，但能画成散点图，一眼看出新鲜组/腐败组是否分得开。",
     "电子鼻论文里那张\u201cPCA得分图\u201d就是这么来的。看到 PCA 图能说\u201c这是把多维数据压缩后看分组\u201d就够了。"],
    ["enose\\ 里的分析脚本、系统感官数据页面"],
    [["练习1", "找到一张 PCA 图（或自己画一张），说出图中两组分没分开", "30分钟"]],
    ["知道 PCA 是\u201c降维看分组\u201d的工具"])

doc.add_heading("四、按需学（用到再补，别提前学）", level=1)
add_table(["知识点", "什么时候补", "怎么补"],
    [["Grad-CAM 热力图原理", "答辩被问\u201c可视化怎么做的\u201d", "讲清\u201c看模型盯着图片哪里判断\u201d即可"],
     ["Flask 前后端通信", "想讲平台架构时", "理解\u201c前端点按钮→后端处理→返回结果\u201d即可"],
     ["显著性分析实操", "写论文做数据表时", "用 Python/SPSS 算出 p 值并写进论文"],
     ["嵌入式改引脚/采样率", "要调整电子鼻设备时", "会改 enose_sketch.ino 对应几行"]],
    [4.2, 6.2, 5.5])

doc.add_heading("五、不要学（省时间的清单）", level=1)
add_table(["不要学", "为什么"],
    [["Python 高级：面向对象/装饰器/多线程", "你不写库、不做大规模开发，用不上"],
     ["算法与数据结构、刷题", "除非你想转码当程序员"],
     ["深度学习数学推导（反向传播公式等）", "会讲概念就够答辩，推导是研究生的事"],
     ["前端框架、CSS 美化", "系统界面已经做好，不用改样式"],
     ["数据库设计/索引优化", "系统用 SQLite 一个文件，够用"],
     ["嵌入式底层：RTOS/驱动/PCB/电路", "你是使用者不是开发者"],
     ["高数/线代/概率", "除非考研要考——那是考试需要，跟系统无关"]],
    [6.2, 9.7])

doc.add_heading("六、学习计划（8周版，每周约7~10小时）", level=1)
add_table(["周次", "学什么（每天1~2小时）", "验收标准"],
    [["第1周", "Python：读CSV→列表/循环→画图；图像：像素/RGB/HSV/阈值分割", "能写\u201c读CSV画图\u201d脚本；能解释HSV分割"],
     ["第2周", "机器学习四张表；跑评估工具看懂准确率和混淆矩阵", "能看懂评估报告，能解释漏检"],
     ["第3周", "传感器三句话；电子鼻数据流四步全部跑通", "能独立采一次数据并画成曲线"],
     ["第4周", "统计三件套；t检验出p值；综合自测", "能不看文档完整复述一遍系统"],
     ["第5周", "SVM/RF + CNN（用比喻讲清楚）", "能用\u201c分界线\u201d\u201c投票\u201d\u201c小窗口找特征\u201d讲模型"],
     ["第6周", "YOLO分类vs检测 + 交叉验证/AUC", "能分清分类与检测，看懂ROC图"],
     ["第7周", "实验设计三原则 + PCA + 按需补漏", "能指出方案里三原则各在哪"],
     ["第8周", "整理成果：把学习内容做成答辩一页纸/PPT", "能对任何人讲懂自己的系统"]],
    [2.2, 9.2, 5.9], font_size=10)
para("注：8周是按\u201c每天1~2小时\u201d排的，可压缩可拉长；大三上的重心转到课程和比赛，这份手册随时回查。", size=10, color="595959")

para("")
para("收尾：你的目标是\u201c懂原理、会操作、能答辩\u201d。每学完一个知识点，用\u201c学会的标准\u201d自测一遍，能讲出来才算数。", bold=True, color="1F4D78")

doc.core_properties.title = "智果AI品控系统学习手册"
doc.save(OUT)
print("SAVED:", OUT)