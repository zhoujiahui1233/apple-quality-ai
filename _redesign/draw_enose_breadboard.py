# -*- coding: utf-8 -*-
"""电子鼻面包板布局图 (MB-102 830孔 顶视图)"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

for _fp in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\msyhbd.ttc']:
    try:
        font_manager.fontManager.addfont(_fp)
    except Exception:
        pass
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(19, 11.6), dpi=150)
ax.set_xlim(0, 19); ax.set_ylim(0, 11.6); ax.axis('off')
fig.patch.set_facecolor('#0a1120')

C = {
    'board': '#101c38', 'hole': '#2c3d63', 'border': '#3f5a8a', 'glow': '#1b3a6b',
    'red': '#ff5252', 'org': '#ffab40', 'gnd': '#9aa7bd',
    'sda': '#ffd740', 'scl': '#69f0ae', 'ana': '#e040fb',
    'txt': '#e8edf5', 'dim': '#8fa3c8', 'warn': '#ffc107', 'spare': '#4a5a7a',
    'mod': '#22304f',
}

COLW = 13.4 / 63.0
def colx(c): return 2.4 + (c - 0.5) * COLW
ROWS = {'a': 9.40, 'b': 8.85, 'c': 8.30, 'd': 7.75, 'e': 7.20,
        'f': 6.45, 'g': 5.90, 'h': 5.35, 'i': 4.80, 'j': 4.25}
RAIL = {'tplus': 10.35, 'tminus': 9.95, 'bminus': 3.70, 'bplus': 3.30}
BRK = 9.21          # 电源轨中点断口
BOARD = (2.4, 2.95, 13.4, 7.75)   # x, y, w, h

def wire(pts, color, lw=2.1, alpha=1.0, z=2):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, '-', color=color, lw=lw, alpha=alpha, zorder=z,
            solid_capstyle='round', solid_joinstyle='round')

def pin(x, y, color, ms=4.2, z=6, hollow=False):
    if hollow:
        ax.plot([x], [y], 'o', ms=ms, mfc='none', mec=color, mew=1.4, zorder=z)
    else:
        ax.plot([x], [y], 'o', ms=ms, color=color, zorder=z)

def label(x, y, s, fs=8, color=None, ha='center', va='center', bold=False, z=7, rot=0):
    ax.text(x, y, s, fontsize=fs, color=color or C['dim'], ha=ha, va=va,
            fontweight='bold' if bold else 'normal', zorder=z, rotation=rot)

def block(c0, c1, r_top, r_bot, text, fs=8, fc=None, alpha=0.5, title_color=None):
    x0, x1 = colx(c0 - 0.5), colx(c1 + 0.5)
    y0, y1 = ROWS[r_bot], ROWS[r_top]
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle='round,pad=0.02,rounding_size=0.05',
                       fc=fc or C['mod'], ec=C['border'], lw=1.2, alpha=alpha, zorder=3)
    ax.add_patch(p)
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha='center', va='center',
            fontsize=fs, color=title_color or C['txt'], fontweight='bold',
            zorder=4, linespacing=1.3)

# ================= 面包板底图 =================
for i in range(3, 0, -1):   # 发光边框
    gx = BOARD[0] - 0.10*i; gy = BOARD[1] - 0.10*i
    gw = BOARD[2] + 0.20*i; gh = BOARD[3] + 0.20*i
    ax.add_patch(FancyBboxPatch((gx, gy), gw, gh, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc='none', ec=C['glow'], lw=1.0, alpha=0.35 - 0.08*i, zorder=0))
ax.add_patch(FancyBboxPatch(BOARD[:2], BOARD[2], BOARD[3],
                            boxstyle='round,pad=0.02,rounding_size=0.12',
                            fc=C['board'], ec=C['border'], lw=1.6, zorder=1))
# 中央凹槽
ax.add_patch(Rectangle((BOARD[0], 6.55), BOARD[2], 0.5, fc='#0b1428', ec='none', zorder=1))
label(15.75, 6.80, '中央凹槽', fs=7, ha='right', color='#5a6f96')

# 终端孔
for c in range(1, 64):
    for r in ['a','b','c','d','e','f','g','h','i','j']:
        ax.plot([colx(c)], [ROWS[r]], 'o', ms=1.15, color=C['hole'], zorder=1.2)

# 电源轨（8 段）
rail_rows = [('tplus', 10.35, 'left', C['red'], '5V'),
             ('tminus', 9.95, 'left', C['gnd'], 'GND'),
             ('tplus', 10.35, 'right', C['org'], 'MQ 3.3V'),
             ('tminus', 9.95, 'right', C['gnd'], 'GND'),
             ('bplus', 3.30, 'left', C['spare'], '备用'),
             ('bminus', 3.70, 'left', C['spare'], '备用'),
             ('bplus', 3.30, 'right', C['org'], '3V3 数字'),
             ('bminus', 3.70, 'right', C['gnd'], 'GND')]
for key, y, side, col, name in rail_rows:
    if side == 'left':
        xs = [colx(c) for c in range(1, 33)]; x0, x1 = 2.45, BRK
        tx = 2.52
    else:
        xs = [colx(c) for c in range(33, 64)]; x0, x1 = BRK, 15.71
        tx = BRK + 0.10
    ax.add_patch(Rectangle((x0, y - 0.15), x1 - x0, 0.30, fc=col, ec='none', alpha=0.28, zorder=1.5))
    for x in xs:
        ax.plot([x], [y], 'o', ms=1.5, color=col, zorder=1.6)
    label(tx, y + 0.24, name, fs=7.5, ha='left', color=col if col != C['spare'] else '#6a7fa0')

label(9.31, 2.82, '电源轨中点断开（左/右半独立，需桥接）', fs=6.5, color='#5a6f96')

# 断口缺口
ax.plot([BRK], [10.35], marker='|', ms=1, color=C['board'])
ax.plot([BRK], [9.95], marker='|', ms=1, color=C['board'])
ax.plot([BRK], [3.30], marker='|', ms=1, color=C['board'])
ax.plot([BRK], [3.70], marker='|', ms=1, color=C['board'])

# ================= 模块 =================
# AMS1117（列29-31，引脚在 c 行）
block(29, 31, 'a', 'c', 'AMS1117-3.3\n5V→3.3V 稳压', fs=8.5)
for c, nm in [(29, 'IN'), (30, 'GND'), (31, 'OUT')]:
    pin(colx(c), ROWS['c'], {'IN': C['red'], 'GND': C['gnd'], 'OUT': C['org']}[nm])
    label(colx(c), ROWS['c'] - 0.17, nm, fs=6.5, color=C['dim'])

# 4 个 MQ（上区，引脚在 a 行）
mqs = [('MQ-3', '酒精/乙醇', 35, 38, 38, 44),
       ('MQ-135', '氨气/腐坏味', 40, 43, 43, 45),
       ('MQ-2', '可燃气体/烟', 45, 48, 48, 46),
       ('MQ-7', '一氧化碳', 50, 53, 53, 47)]
for name, note, c0, c1, aout, ain in mqs:
    block(c0, c1, 'a', 'c', name + '\n' + note, fs=8)
    pin(colx(c0), ROWS['a'], C['org']); label(colx(c0), ROWS['a'] - 0.16, 'VCC', fs=6.5, color=C['org'])
    pin(colx(c0 + 1), ROWS['a'], C['gnd']); label(colx(c0 + 1), ROWS['a'] - 0.16, 'GND', fs=6.5, color=C['gnd'])
    pin(colx(c0 + 2), ROWS['a'], C['spare'], hollow=True); label(colx(c0 + 2), ROWS['a'] - 0.16, 'DOUT\n不接', fs=5.5, color=C['spare'])
    pin(colx(aout), ROWS['a'], C['ana']); label(colx(aout), ROWS['a'] - 0.16, 'AOUT', fs=6.5, color=C['ana'])
    # 电源线
    wire([(colx(c0), ROWS['a']), (colx(c0), RAIL['tplus'])], C['org'], lw=1.8)
    wire([(colx(c0 + 1), ROWS['a']), (colx(c0 + 1), RAIL['tminus'])], C['gnd'], lw=1.8)
    # 模拟线 → AIN
    wire([(colx(aout), ROWS['a']), (colx(aout), 6.65), (colx(ain), 6.65), (colx(ain), ROWS['j'])], C['ana'], lw=1.7)
    label((colx(aout) + colx(ain)) / 2, 6.82, '→AIN%d' % (ain - 44), fs=6, color=C['ana'])

# 数字模块（下区，引脚在 j 行）
dmods = [('SGP30', 'eCO2/tVOC', 35, 36, 37, 38, 44),
         ('BME680', '温湿压/VOC', 40, 41, 42, 43, 45),
         ('OLED 0.96"', 'SSD1306', 58, 59, 60, 61, 46)]
for name, note, vdd, gnd, scl, sda, _ in dmods:
    block(vdd, sda, 'f', 'h', name + '\n' + note, fs=7.5)
    pin(colx(vdd), ROWS['j'], C['org']); label(colx(vdd), ROWS['j'] + 0.17, 'VDD', fs=6, color=C['org'])
    pin(colx(gnd), ROWS['j'], C['gnd']); label(colx(gnd), ROWS['j'] + 0.17, 'GND', fs=6, color=C['gnd'])
    pin(colx(scl), ROWS['j'], C['scl']); label(colx(scl), ROWS['j'] + 0.17, 'SCL', fs=6, color=C['scl'])
    pin(colx(sda), ROWS['j'], C['sda']); label(colx(sda), ROWS['j'] + 0.17, 'SDA', fs=6, color=C['sda'])
    wire([(colx(vdd), ROWS['j']), (colx(vdd), RAIL['bplus'])], C['org'], lw=1.7)
    wire([(colx(gnd), ROWS['j']), (colx(gnd), RAIL['bminus'])], C['gnd'], lw=1.7)
    wire([(colx(sda), ROWS['j']), (colx(sda), 5.20), (colx(55), 5.20)], C['sda'], lw=1.6)
    wire([(colx(scl), ROWS['j']), (colx(scl), 4.75), (colx(56), 4.75)], C['scl'], lw=1.6)

# ADS1115（10 引脚）
ads_pins = [('A0', 44, C['ana']), ('A1', 45, C['ana']), ('A2', 46, C['ana']), ('A3', 47, C['ana']),
            ('VDD', 48, C['org']), ('GND', 49, C['gnd']), ('SDA', 50, C['sda']), ('SCL', 51, C['scl']),
            ('ADDR', 52, C['gnd']), ('ALRT', 53, C['spare'])]
block(44, 53, 'f', 'h', 'ADS1115\n16位4通道ADC', fs=8)
for nm, c, col in ads_pins:
    pin(colx(c), ROWS['j'], col)
    label(colx(c), ROWS['j'] + 0.17, nm, fs=5.5, color=C['dim'] if nm == 'ALRT' else col)
wire([(colx(48), ROWS['j']), (colx(48), RAIL['bplus'])], C['org'], lw=1.7)
wire([(colx(49), ROWS['j']), (colx(49), RAIL['bminus'])], C['gnd'], lw=1.7)
wire([(colx(50), ROWS['j']), (colx(50), 5.20), (colx(55), 5.20)], C['sda'], lw=1.6)
wire([(colx(51), ROWS['j']), (colx(51), 4.75), (colx(56), 4.75)], C['scl'], lw=1.6)
wire([(colx(52), ROWS['j']), (colx(52), RAIL['bminus'])], C['gnd'], lw=1.6)
label(colx(52), ROWS['j'] - 0.17, '接GND=0x48', fs=5.5, color=C['gnd'])

# SDA/SCL 总线列高亮
ax.add_patch(Rectangle((colx(55) - 0.055, 4.25), 0.11, 2.2, fc=C['sda'], alpha=0.12, zorder=1.4))
ax.add_patch(Rectangle((colx(56) - 0.055, 4.25), 0.11, 2.2, fc=C['scl'], alpha=0.12, zorder=1.4))
label(colx(55), 6.72, 'SDA总线列55', fs=6.5, color=C['sda'])
label(colx(56), 6.72, 'SCL总线列56', fs=6.5, color=C['scl'])

# AMS1117 接线
wire([(colx(29), ROWS['c']), (colx(29), RAIL['tplus'])], C['red'], lw=1.9)
wire([(colx(30), ROWS['c']), (colx(30), RAIL['tminus'])], C['gnd'], lw=1.9)
wire([(colx(31), ROWS['c']), (colx(31), 10.75), (BRK, 10.75), (BRK, RAIL['tplus'])], C['org'], lw=2.0)

# GND 桥接
wire([(colx(32), RAIL['tminus']), (BRK + 0.24, RAIL['tminus'])], C['gnd'], lw=2.4)
wire([(BRK + 0.24, RAIL['tminus']), (BRK + 0.24, RAIL['bminus'])], C['gnd'], lw=2.4)
label(BRK + 0.24, 7.0, 'GND桥', fs=6.5, color=C['gnd'], rot=90)

# ================= ESP32（板外，左侧） =================
ax.add_patch(FancyBboxPatch((0.25, 4.55), 1.9, 4.5,
                            boxstyle='round,pad=0.05,rounding_size=0.12',
                            fc='#1b2a4a', ec=C['border'], lw=1.6, zorder=3))
label(1.2, 7.1, 'ESP32\nDevKitC', fs=12, color=C['txt'], bold=True, z=4)
esp = [('5V', 8.6, C['red'], '5V'),
       ('GND', 7.7, C['gnd'], 'GND'),
       ('3V3', 6.8, C['org'], '3V3'),
       ('GPIO21', 5.9, C['sda'], 'SDA'),
       ('GPIO22', 5.0, C['scl'], 'SCL')]
for nm, y, col, short in esp:
    pin(2.15, y, col, ms=4.5)
    label(2.0, y, nm + ('' if nm.startswith(('5','G','3')) else '·'+short), fs=7.5,
          ha='right', color=C['txt'] if nm in ('5V','GND','3V3') else col)

wire([(2.15, 8.6), (colx(2), 8.6), (colx(2), RAIL['tplus'])], C['red'], lw=2.2)
wire([(2.15, 7.7), (colx(2), 7.7), (colx(2), RAIL['tminus'])], C['gnd'], lw=2.2)
wire([(2.15, 6.8), (2.15, 2.55), (colx(34), 2.55), (colx(34), RAIL['bplus'])], C['org'], lw=2.2)
wire([(2.15, 5.9), (2.15, 2.25), (colx(55), 2.25), (colx(55), ROWS['j'])], C['sda'], lw=2.2)
wire([(2.15, 5.0), (2.15, 1.95), (colx(56), 1.95), (colx(56), ROWS['j'])], C['scl'], lw=2.2)
label(2.0, 2.55, '', fs=1)

# ================= 右侧说明面板 =================
def panel(x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03,rounding_size=0.08',
                                fc='#0e1a33', ec='#3f5a8a', lw=1.2, zorder=2))

def ml(x, y, lines, fs=8.5, color=None, lh=0.30, z=5):
    for i, ln in enumerate(lines):
        ax.text(x, y - i * lh, ln, fontsize=fs, color=color or C['dim'], ha='left', va='center', zorder=z)

# 线色说明
panel(16.25, 8.55, 2.65, 2.35)
label(16.4, 10.68, '线色说明', fs=11, color=C['txt'], bold=True, ha='left')
ml(16.4, 10.30, [
    '红 5V  ：ESP32 5V → AMS1117 IN',
    '橙 3.3V：AMS1117 OUT → MQ 供电轨',
    '         ESP32 3V3 → 数字模块',
    '蓝灰 GND：全部共地（含2处桥接）',
    '黄 SDA：GPIO21 → 总线列55 → SDA',
    '绿 SCL：GPIO22 → 总线列56 → SCL',
    '紫 AOUT：MQ → ADS1115 AIN0~3',
], fs=8, lh=0.27)

# 推荐接线顺序
panel(16.25, 4.15, 2.65, 4.25)
label(16.4, 8.20, '推荐接线顺序（新手照做）', fs=10.5, color=C['txt'], bold=True, ha='left')
ml(16.4, 7.80, [
    '1. 桥GND：顶部左-→顶部右-，再',
    '   垂直桥到底部右-（共地）',
    '2. AMS1117：IN←5V轨；OUT→MQ3.3V',
    '   轨；GND→GND轨',
    '3. 4×MQ：VCC→MQ3.3V轨；GND→',
    '   GND轨；DOUT悬空不接',
    '4. 数字模块：VDD→3V3轨；GND→GND',
    '5. SDA/SCL：各模块跳线到总线列',
    '   55/56（SDA列55、SCL列56）',
    '6. ADS1115：ADDR→GND(0x48)；',
    '   ALRT不接',
    '7. 紫线：MQ AOUT→AIN0~3 一一对应',
    '8. ESP32五根杜邦线：5V/GND/3V3/',
    '   SDA/SCL（公对母）',
], fs=8, lh=0.265)

# 注意事项
panel(16.25, 0.5, 2.65, 3.5)
label(16.4, 3.82, '⚠ 注意', fs=10.5, color=C['warn'], bold=True, ha='left')
ml(16.4, 3.42, [
    '· 电源轨左/右半不相通，必须桥接',
    '· 模块引脚顺序以丝印为准（图为',
    '  常见顺序，别死板对位）',
    '· SGP30 只能接 3.3V，严禁接 5V',
    '· 上电前万用表蜂鸣档查 5V/3.3V/',
    '  GND 不短路',
    '· ESP32 不插面包板，放板左侧用',
    '  公对母杜邦线连接',
], fs=8, lh=0.28)

# 标题
label(2.4, 11.30, '电子鼻面包板布局图（MB-102 · 830 孔 · 顶视图）', fs=20, color=C['txt'], bold=True, ha='left')
label(2.4, 11.00, '照着图把模块插进对应孔位，跳线颜色一一对应即可', fs=11, color=C['dim'], ha='left')

out = r'E:\apple\docs\enose_breadboard_layout.png'
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches='tight')
print('saved', out)
