# -*- coding: utf-8 -*-
"""
Webcam real-time apple detection using the current YOLO model.

Usage:
    python scripts/webcam_yolo.py

Keys:
    q / Esc  : quit
    s        : save current frame to E:/apple/webcam_snapshot.jpg

Notes:
    If no camera is found, it tries index 0 then 1.
"""
import os
import sys
import time

os.environ["YOLO_CONFIG_DIR"] = r"E:\apple\.ultralytics"
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from ultralytics import YOLO


def pick_camera():
    for idx in (0, 1, 2):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            print(f"摄像头已打开: index={idx}")
            return cap
        if cap is not None:
            cap.release()
    return None


def main():
    model_path = r"E:\apple\yolo_detect_best.pt"
    if not os.path.exists(model_path):
        print(f"[error] 找不到模型: {model_path}")
        return
    model = YOLO(model_path)
    print("YOLO 模型加载成功，开始摄像头检测...")
    print("按键: q/Esc 退出, s 保存截图")

    cap = pick_camera()
    if cap is None:
        print("[error] 没有找到可用摄像头")
        return

    win = "YOLO Apple Detection (q=quit, s=save)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)

    last_save = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取画面失败")
            break

        t0 = time.time()
        r = model.predict(frame, conf=0.85, imgsz=640, verbose=False)[0]
        dt_ms = (time.time() - t0) * 1000

        view = frame.copy()
        n_box = 0
        if r.boxes is not None and len(r.boxes):
            n_box = len(r.boxes)
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                color = (0, 255, 0) if c >= 0.5 else (0, 165, 255)
                cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"apple {c:.2f}"
                cv2.putText(view, label, (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(view, f"apples: {n_box}  |  infer: {dt_ms:.0f} ms",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(win, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s") and time.time() - last_save > 1.0:
            cv2.imwrite(r"E:\apple\webcam_snapshot.jpg", view,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            last_save = time.time()
            print("已保存截图 -> E:\\apple\\webcam_snapshot.jpg")

    cap.release()
    cv2.destroyAllWindows()
    print("已退出")


if __name__ == "__main__":
    main()
