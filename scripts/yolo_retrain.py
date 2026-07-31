# -*- coding: utf-8 -*-
"""
基于 yolo_detect_dataset 重新训练 YOLO 检测模型（从当前 best.pt 继续微调）

用法:
  python scripts/yolo_retrain.py --epochs 30
"""
import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import config


def _log(log, msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line)
    if log is not None:
        log(line)


def train(category="apple", epochs=30, imgsz=640, batch=8, log=None):
    os.environ["YOLO_CONFIG_DIR"] = str(config.ROOT_DIR / ".ultralytics")
    from ultralytics import YOLO
    import torch

    data_yaml = config.ROOT_DIR / "yolo_detect_dataset" / "data.yaml"
    if not data_yaml.exists():
        raise RuntimeError("yolo_detect_dataset/data.yaml 不存在，请先导出 YOLO 训练集")

    start = config.MODEL_PATHS["yolo_detect"]
    if start.exists():
        model = YOLO(str(start))
        _log(log, f"从当前模型继续微调: {start.name}")
    else:
        pt = config.ROOT_DIR / "yolo11n.pt"
        model = YOLO(str(pt) if pt.exists() else "yolo11n.yaml")
        _log(log, "未找到当前模型，从预训练权重/从头开始")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(log, f"设备: {device} （CPU 训练较慢，建议先用小轮数测试）")
    _log(log, f"开始训练: epochs={epochs}, imgsz={imgsz}, batch={batch}")

    out_dir = config.ROOT_DIR / "yolo_detect_output" / "retrain"
    t0 = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=device,
        project=str(out_dir.parent),
        name="retrain",
        exist_ok=True,
        patience=10,
        workers=2,
    )
    best = out_dir / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(str(best), str(start))
        _log(log, f"新权重已写入: {start.name}")
    try:
        rd = results.results_dict
        acc = round(float(rd.get("metrics/mAP50-95(B)", rd.get("mAP50-95", 0))) * 100, 2)
    except Exception:
        acc = 0.0
    _log(log, f"YOLO 重训完成，耗时 {time.time() - t0:.0f}s")
    return acc


def main():
    parser = argparse.ArgumentParser(description="重新训练 YOLO 检测模型")
    parser.add_argument("--category", default="apple")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    train(args.category, epochs=args.epochs)


if __name__ == "__main__":
    main()
