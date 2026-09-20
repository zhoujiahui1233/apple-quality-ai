# -*- coding: utf-8 -*-
"""
电子鼻采集控制台
把所有采集要用的命令集中到一个菜单里，双击「采集控制台.bat」即可使用。
采集数据自动存到 data/raw，文件名自动按流水号+标签生成，不会覆盖旧数据。
"""
import glob
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE, "data", "raw")
COLLECT_WIFI = os.path.join(BASE, "collect_wifi.py")
COLLECT_USB = os.path.join(BASE, "collect.py")


def next_serial():
    """扫描 data/raw 里的 sXXX 文件，返回下一个流水号（如 s005）。"""
    max_n = 0
    if os.path.isdir(RAW_DIR):
        for f in os.listdir(RAW_DIR):
            m = re.match(r"s(\d{3})", f)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def list_com_ports():
    """尽量列出串口，装不了 pyserial 就返回空列表。"""
    try:
        import serial.tools.list_ports as lp
        return [p.device for p in lp.comports()]
    except Exception:
        return []


def ask_label():
    """输入样品标签，直接回车默认 sample。"""
    label = input("  样品标签（回车默认 sample；如 fresh / mid / rotten / baseline）: ").strip()
    return label or "sample"


def make_out_path(label):
    """自动生成输出文件名：s005_rotten_2140.csv，放进 data/raw。"""
    stamp = time.strftime("%H%M")
    name = "s%03d_%s_%s.csv" % (next_serial(), label, stamp)
    return os.path.join("data", "raw", name)


def run(cmd):
    """运行采集脚本，结束后回到菜单。"""
    print()
    print(">>> 开始采集：%s" % " ".join(cmd))
    print(">>> 停止方法：按 Ctrl+C")
    print("-" * 46)
    try:
        subprocess.call(cmd, cwd=BASE)
    except KeyboardInterrupt:
        pass
    print("-" * 46)
    print(">>> 采集结束，已回到控制台。")


def wifi_collect():
    print()
    print("[WiFi 采集] 先运行接收端，再给 ESP32 上电，等 OLED 显示 W:OK C:OK")
    label = ask_label()
    out = make_out_path(label)
    run([sys.executable, COLLECT_WIFI, "--label", label, "--out", out])


def usb_collect():
    print()
    ports = list_com_ports()
    if ports:
        print("[USB 采集] 检测到串口：" + "、".join(ports))
    default_port = ports[0] if ports else "COM10"
    port = input("  串口（回车默认 %s）: " % default_port).strip() or default_port
    label = ask_label()
    out = make_out_path(label)
    run([sys.executable, COLLECT_USB, "--port", port, "--label", label, "--out", out])


def open_folder():
    if not os.path.isdir(RAW_DIR):
        os.makedirs(RAW_DIR)
    os.startfile(RAW_DIR)


def open_record():
    p = os.path.join(BASE, "data", "record.xlsx")
    if os.path.exists(p):
        os.startfile(p)
    else:
        print("没找到 data/record.xlsx，请先按操作卡建立记录表。")


def open_card():
    cands = glob.glob(os.path.join(BASE, "*操作卡*.docx"))
    if cands:
        os.startfile(cands[0])
    else:
        print("没找到傻瓜操作卡 docx。")


def main():
    print("=" * 46)
    print("   智果 · 电子鼻采集控制台")
    print("   所有采集操作都从这里进，选数字回车")
    print("=" * 46)
    while True:
        print()
        print("1) WiFi 采集（充电宝+无线，推荐）")
        print("2) USB 串口采集（插线）")
        print("3) 打开采集数据文件夹")
        print("4) 打开记录表 record.xlsx")
        print("5) 打开傻瓜操作卡")
        print("0) 退出")
        choice = input("请选择：").strip()
        if choice == "1":
            wifi_collect()
        elif choice == "2":
            usb_collect()
        elif choice == "3":
            open_folder()
        elif choice == "4":
            open_record()
        elif choice == "5":
            open_card()
        elif choice == "0":
            print("再见，测完记得整理数据~")
            break
        else:
            print("没这个选项，重新选。")


if __name__ == "__main__":
    main()