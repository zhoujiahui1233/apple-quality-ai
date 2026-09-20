# -*- coding: utf-8 -*-
"""
电子鼻数据采集上位机（ESP32 免焊版）

从 ESP32 串口读取传感器数据，实时写入 CSV。

依赖：pip install pyserial

用法：
    python collect.py --port COM3 --label apple_day1
    python collect.py --port COM3 --label apple_rotten_day5 --out data/apple_day5.csv

提示：
    - 每次实验用 --label 标记样品（如 apple_fresh / apple_day3 / apple_rotten），
      这个标签会写入 CSV 的 label 列，训练时作为分类目标。
    - 正式采样前先测一次"空罐基线"（--label baseline），训练时做差分校正。
    - 若用旧版 Arduino Nano 固件（MQ+DHT11），请用 git 历史中的旧版脚本。
    - Ctrl+C 停止采集。
"""
import argparse
import csv
import time

try:
    import serial
except ImportError:
    raise SystemExit("缺少 pyserial，请先运行: pip install pyserial")

HEADER = ["ts_ms", "MQ1", "MQ2", "MQ3", "MQ4", "tempC", "humPct",
          "pressPa", "gasOhm", "eco2", "tvoc", "label"]
N_COLS = len(HEADER) - 1  # 去掉 label


def main():
    ap = argparse.ArgumentParser(description="电子鼻串口采集")
    ap.add_argument("--port", required=True, help="串口名称，Windows 如 COM3，Linux 如 /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=9600, help="波特率，默认 9600")
    ap.add_argument("--out", default="enose_data.csv", help="输出 CSV 路径")
    ap.add_argument("--label", default="sample", help="样品标签，写入每行 label 列")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)  # 等待串口就绪
    print(f"开始采集 -> {args.out}  标签: {args.label}  (Ctrl+C 停止)")

    n = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        try:
            while True:
                raw = ser.readline().decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                if raw.startswith("ENOSE_READY") or raw.endswith("_ERROR"):
                    print("[Arduino]", raw)
                    continue
                parts = raw.split(",")
                if len(parts) < N_COLS:
                    continue  # 跳过表头/异常行
                row = parts[:N_COLS] + [args.label]
                writer.writerow(row)
                f.flush()
                n += 1
                print(f"[{n:>5}] " + raw)
        except KeyboardInterrupt:
            print(f"\n采集结束，共 {n} 行 -> {args.out}")
        finally:
            ser.close()


if __name__ == "__main__":
    main()
