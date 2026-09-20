# -*- coding: utf-8 -*-
"""Draw e-nose wiring diagram (ESP32 + 4xMQ + ADS1115 + BME680 + SGP30 + OLED)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _fp in [r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\msyhbd.ttc']:
    try:
        font_manager.fontManager.addfont(_fp)
    except Exception:
        pass
plt.rcParams['font.family'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import FancyBboxPatch, Circle

fig, ax = plt.subplots(figsize=(17, 11), dpi=150)
ax.set_xlim(0, 17); ax.set_ylim(0, 11); ax.axis('off')
fig.patch.set_facecolor('#0d1526')

C = {
    'esp': '#1b2a4a', 'mod': '#22304f', 'border': '#3f5a8a',
    'red': '#ff5252', 'org': '#ffab40', 'gnd': '#9aa7bd', 'sda': '#ffd740',
    'scl': '#69f0ae', 'ana': '#e040fb', 'txt': '#e8edf5', 'dim': '#8fa3c8',
    'ok': '#00ff88', 'warn': '#ffc107',
}

def module(x, y, w, h, title, fc=C['mod'], fs=13, title_color=C['txt']):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06,rounding_size=0.15',
                       fc=fc, ec=C['border'], lw=1.6, zorder=3)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=fs,
            color=title_color, fontweight='bold', zorder=4)
    return p

def pin(x, y, label, color, side='left', dy=0):
    ax.plot([x], [y], 'o', ms=5, color=color, zorder=5)
    if side == 'left':
        ax.text(x - 0.12, y + dy, label, ha='right', va='center', fontsize=11, color=C['dim'], zorder=5)
    elif side == 'right':
        ax.text(x + 0.12, y + dy, label, ha='left', va='center', fontsize=11, color=C['dim'], zorder=5)
    elif side == 'top':
        ax.text(x, y + 0.16 + dy, label, ha='center', va='bottom', fontsize=11, color=C['dim'], zorder=5)
    elif side == 'bottom':
        ax.text(x, y - 0.16 + dy, label, ha='center', va='top', fontsize=11, color=C['dim'], zorder=5)

def wire(p1, p2, color, lw=2.2, style='-'):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], style, color=color, lw=lw, zorder=2,
            solid_capstyle='round')

def elbow(p1, p2, color, lw=2.2):
    """L-shaped wire."""
    ax.plot([p1[0], p1[0], p2[0]], [p1[1], p2[1], p2[1]], '-', color=color, lw=lw,
            zorder=2, solid_capstyle='round')

def rail_label(x, y, text, color):
    ax.text(x, y, text, fontsize=10, color=color, fontweight='bold', ha='left', va='center', zorder=6)

# ============ ESP32 ============
module(0.4, 3.2, 3.0, 4.6, 'ESP32 DevKitC', fc=C['esp'], fs=13)
ax.text(1.9, 7.35, 'ESP32', ha='center', fontsize=10.5, color=C['dim'])
ex = 3.4
pin(ex, 7.3, '5V', C['red'], 'right')
pin(ex, 6.3, '3V3', C['org'], 'right')
pin(ex, 5.3, 'GPIO21 (SDA)', C['sda'], 'right')
pin(ex, 4.6, 'GPIO22 (SCL)', C['scl'], 'right')
pin(ex, 3.9, 'GND', C['gnd'], 'right')

# ============ AMS1117 ============
module(5.2, 6.9, 2.1, 1.4, 'AMS1117-3.3', fs=11)
pin(5.2, 7.6, 'IN', C['red'], 'left')
pin(7.3, 7.6, 'OUT 3.3V', C['org'], 'right')
pin(6.25, 6.9, 'GND', C['gnd'], 'bottom', dy=-0.18)

# 5V wire: ESP32 5V -> AMS1117 IN
elbow((ex, 7.3), (5.2, 7.6), C['red'])

# ============ MQ power rail ============
rail_x = 8.6
# AMS1117 OUT -> rail
elbow((7.3, 7.6), (rail_x, 7.6), C['org'])
ax.plot([rail_x, rail_x], [2.6, 7.6], '-', color=C['org'], lw=2.6, zorder=2)
rail_label(rail_x + 0.15, 7.85, '3.3V 供电轨（给 4 个 MQ）', C['org'])

# ============ MQ modules ============
mqs = [('MQ-3', 7.5, '酒精/乙醇'), ('MQ-135', 6.2, '氨气/腐坏味'), ('MQ-2', 4.9, '可燃气体/烟'), ('MQ-7', 3.6, '一氧化碳')]
mx0, mx1 = 9.9, 12.6
for i, (name, my, note) in enumerate(mqs):
    module(mx0, my, mx1 - mx0, 0.95, name, fs=12)
    ax.text(mx0 + (mx1-mx0)/2, my - 0.28, note, ha='center', va='center', fontsize=10, color=C['dim'])
    # VCC (left top), GND (left bottom), AOUT (bottom center)
    pin(mx0, my + 0.75, 'VCC', C['org'], 'left')
    pin(mx0, my + 0.2, 'GND', C['gnd'], 'left')
    pin(mx0 + (mx1-mx0)/2, my, 'AOUT', C['ana'], 'bottom', dy=-0.16)
    # power: rail -> VCC
    ax.plot([rail_x, mx0], [my + 0.75, my + 0.75], '-', color=C['org'], lw=2.0, zorder=2)
    # GND drop
    elbow((mx0, my + 0.2), (mx0, 1.15), C['gnd'], lw=1.8)

# ============ ADS1115 ============
ax_w = 2.4
ax0, ay0 = 6.4, 0.7
module(ax0, ay0, ax_w, 2.4, 'ADS1115\n16位 4通道ADC', fs=11)
for i in range(4):
    pin(ax0 + 0.5 + i * 0.47, ay0 + 2.4, 'AIN%d' % i, C['ana'], 'top', dy=0.04)
pin(ax0 + ax_w, ay0 + 1.9, 'VDD', C['org'], 'right')
pin(ax0 + ax_w, ay0 + 1.4, 'GND', C['gnd'], 'right')
pin(ax0 + ax_w, ay0 + 0.9, 'SDA', C['sda'], 'right')
pin(ax0 + ax_w, ay0 + 0.45, 'SCL', C['scl'], 'right')

# MQ AOUT -> AIN0..3 (L-shaped, avoid crossing too messily)
ain_xs = [ax0 + 0.5 + i * 0.47 for i in range(4)]
for i, (_, my, _) in enumerate(mqs):
    aout_x = mx0 + (mx1-mx0)/2
    elbow((aout_x, my), (ain_xs[i], ay0 + 2.4), C['ana'], lw=1.8)

# ============ I2C buses from ESP32 ============
# SDA bus at y=1.55, SCL at y=1.15, 3V3 at y=1.95, GND at y=0.75
sda_y, scl_y, v3_y, gnd_y = 1.55, 1.15, 1.95, 0.75
# from ESP32 pins down to bus level then right
elbow((ex, 5.3), (2.0, sda_y), C['sda'], lw=2.2); ax.plot([2.0, 15.6], [sda_y, sda_y], '-', color=C['sda'], lw=2.2, zorder=2)
elbow((ex, 4.6), (2.0, scl_y), C['scl'], lw=2.2); ax.plot([2.0, 15.6], [scl_y, scl_y], '-', color=C['scl'], lw=2.2, zorder=2)
elbow((ex, 6.3), (2.0, v3_y), C['org'], lw=2.2); ax.plot([2.0, 15.6], [v3_y, v3_y], '-', color=C['org'], lw=2.2, zorder=2)
elbow((ex, 3.9), (2.0, gnd_y), C['gnd'], lw=2.2); ax.plot([2.0, 15.6], [gnd_y, gnd_y], '-', color=C['gnd'], lw=2.2, zorder=2)
rail_label(2.1, v3_y + 0.22, '3V3 总线（数字模块供电）', C['org'])
rail_label(2.1, sda_y + 0.22, 'SDA 总线', C['sda'])
rail_label(2.1, scl_y + 0.22, 'SCL 总线', C['scl'])
rail_label(2.1, gnd_y - 0.22, 'GND 总线（全部共地）', C['gnd'])

# ADS1115 connects to buses (right edge pins)
for bus_y, label, col, pin_y in [(v3_y, 'VDD', C['org'], ay0 + 1.9), (gnd_y, 'GND', C['gnd'], ay0 + 1.4),
                                  (sda_y, 'SDA', C['sda'], ay0 + 0.9), (scl_y, 'SCL', C['scl'], ay0 + 0.45)]:
    elbow((ax0 + ax_w, pin_y), (13.0, bus_y), col, lw=1.8)

# ============ BME680 / SGP30 / OLED ============
bme = (9.5, 3.1, 2.0, 1.6); module(bme[0], bme[1], bme[2], bme[3], 'BME680\n温/湿/压/VOC')
sgp = (11.9, 3.1, 2.0, 1.6); module(sgp[0], sgp[1], sgp[2], sgp[3], 'SGP30\neCO2 / tVOC')
old = (14.1, 3.1, 2.4, 1.6); module(old[0], old[1], old[2], old[3], 'OLED 0.96"\nSSD1306')

# each digital module: bottom pins VIN/SDA/SCL/GND -> buses
for (bx, by, bw, bh), name in [(bme, 'BME680'), (sgp, 'SGP30'), (old, 'OLED')]:
    cx = bx + bw/2
    pins = [('VIN', cx - 0.45, v3_y), ('SDA', cx - 0.15, sda_y), ('SCL', cx + 0.15, scl_y), ('GND', cx + 0.45, gnd_y)]
    for label, px, bus_y in pins:
        # pin on bottom edge of module
        ax.plot([px], [by], 'o', ms=5, color={'VIN': C['org'], 'SDA': C['sda'], 'SCL': C['scl'], 'GND': C['gnd']}[label], zorder=5)
        ax.plot([px, px], [by, bus_y], '-', lw=1.8, color={'VIN': C['org'], 'SDA': C['sda'], 'SCL': C['scl'], 'GND': C['gnd']}[label], zorder=2)
    ax.text(bx + bw/2, by - 0.42, name, ha='center', fontsize=10, color=C['dim'])

# ============ legend ============
lg = [
    (C['red'], '5V 电源（ESP32 → AMS1117 IN）'),
    (C['org'], '3.3V（AMS1117 OUT → MQ 供电；3V3 总线 → 数字模块）'),
    (C['gnd'], 'GND 共地（所有模块）'),
    (C['sda'], 'SDA = GPIO21'),
    (C['scl'], 'SCL = GPIO22'),
    (C['ana'], 'MQ AOUT → ADS1115 AIN0~3'),
]
ax.text(0.4, 10.6, '电子鼻接线图（ESP32 免焊版）', fontsize=19, color=C['txt'], fontweight='bold')
for i, (col, t) in enumerate(lg):
    y = 10.0 - i * 0.34
    ax.plot([0.4, 0.9], [y, y], '-', color=col, lw=3, zorder=2)
    ax.text(1.05, y, t, fontsize=11.5, color=C['dim'], va='center')

out = r'E:\apple\docs\enose_wiring_diagram.png'
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches='tight')
print('saved', out)
