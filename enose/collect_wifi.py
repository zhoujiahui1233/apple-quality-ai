# -*- coding: utf-8 -*-
"""
电子鼻数据采集上位机（WiFi 版，方式一：ESP32 连路由器）

ESP32 不再用 USB 线传数据，而是把传感器数据通过 WiFi（TCP）发到本脚本，
本脚本负责接收、实时写入 CSV。用法和 collect.py 基本一样。

依赖：无（只用 Python 自带库）

用法：
    python collect_wifi.py --label apple_day1
    python collect_wifi.py --label apple_rotten_day5 --out data/apple_day5.csv

步骤（每次实验）：
    1. 电脑连上 WiFi（和 ESP32 同一个路由器）。
    2. 先运行本脚本，它会打印本机局域网 IP 和监听端口。
    3. 把 IP 填进 enose_sketch.ino 的 PC_HOST_IP（改一次就行，除非 IP 变了）。
    4. 给 ESP32 上电（充电宝或 USB 都行），等 OLED 显示 W:OK C:OK。
    5. 本窗口开始滚动显示数据行，Ctrl+C 停止采集。

提示：
    - 首次运行如果 Windows 弹防火墙提示，选"允许访问"（专用网络）。
    - 电脑 IP 是路由器分配的，可能变化；连不上时用 ipconfig 重新查。
    - 每次实验用 --label 标记样品（如 apple_fresh / apple_day3 / apple_rotten），
      这个标签会写入 CSV 的 label 列，训练时作为分类目标。
"""
import argparse
import csv
import socket

HEADER = ["ts_ms", "MQ1", "MQ2", "MQ3", "MQ4", "tempC", "humPct",
          "pressPa", "gasOhm", "eco2", "tvoc", "label"]
N_COLS = len(HEADER) - 1  # 去掉 label


def local_ips():
    """列出本机局域网 IP，供填进固件的 PC_HOST_IP。"""
    ips = set()
    # 方法1：用 UDP"假装"连一个外网地址，让系统返回本机出口 IP（不会真发包）
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    # 方法2：按主机名解析
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return [ip for ip in ips if not ip.startswith("127.")]


def main():
    ap = argparse.ArgumentParser(description="电子鼻 WiFi 采集（电脑接收端）")
    ap.add_argument("--listen", type=int, default=9000,
                    help="监听端口，默认 9000，和固件 PC_PORT 保持一致")
    ap.add_argument("--out", default="enose_wifi_data.csv", help="输出 CSV 路径")
    ap.add_argument("--label", default="sample", help="样品标签，写入每行 label 列")
    args = ap.parse_args()

    ips = local_ips()
    print("本机局域网 IP（填进固件 PC_HOST_IP）：")
    if ips:
        for ip in ips:
            print("    ", ip)
    else:
        print("    未检测到，请先确认电脑已连上 WiFi，或用 ipconfig 查看。")
    print(f"监听端口: {args.listen}  标签: {args.label}  (Ctrl+C 停止)")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.listen))  # 监听所有网卡，WiFi/网线都能连
    srv.listen(1)

    n = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        print(f"等待 ESP32 连接... 数据将写入 -> {args.out}")
        try:
            while True:
                conn, addr = srv.accept()
                print(f"[WiFi] ESP32 已连接: {addr[0]}")
                try:
                    buf = b""
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break  # ESP32 断开
                        buf += data
                        # 按换行符切出完整数据行，处理粘包
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            raw = line.decode("utf-8", errors="replace").strip()
                            if not raw:
                                continue
                            if raw.startswith("ENOSE_READY") or raw.endswith("_ERROR"):
                                print("[Arduino]", raw)
                                continue
                            parts = raw.split(",")
                            if len(parts) < N_COLS:
                                continue  # 跳过表头/异常行
                            if parts[0] == "ts_ms":
                                continue  # 保险：再跳过一遍表头
                            row = parts[:N_COLS] + [args.label]
                            writer.writerow(row)
                            f.flush()
                            n += 1
                            print(f"[{n:>5}] " + raw)
                except OSError:
                    pass
                finally:
                    conn.close()
                    print("[WiFi] 连接断开，等待 ESP32 重新连接...")
        except KeyboardInterrupt:
            print(f"\n采集结束，共 {n} 行 -> {args.out}")
        finally:
            srv.close()


if __name__ == "__main__":
    main()