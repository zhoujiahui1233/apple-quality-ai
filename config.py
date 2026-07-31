import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
STATIC_DIR = ROOT_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
TEMPLATE_DIR = ROOT_DIR / "templates"
INSTANCE_DIR = ROOT_DIR / "instance"
OUTPUT_DIR = ROOT_DIR / "output"

HOST = "0.0.0.0"
PORT = 5020
DEBUG = True

DATABASE_URI = "sqlite:///" + str(INSTANCE_DIR / "database.db")

MODEL_PATHS = {
    "scaler": ROOT_DIR / "apple_scaler.pkl",
    "svm": ROOT_DIR / "apple_svm_model.dat",
    "rf": ROOT_DIR / "apple_rf_model.pkl",
    "dl": ROOT_DIR / "apple_dl_model.pth",
    "yolo_cls": ROOT_DIR / "yolo_output" / "yolo_classify-2" / "weights" / "best.pt",
    "yolo_detect": ROOT_DIR / "yolo_detect_best.pt",
}

IMAGE_WIDTH = 500
PREMIUM_THRESHOLD = 0.05
STANDARD_THRESHOLD = 0.15
DEFECT_SENSITIVITY = 5
DEFECT_MIN_AREA = 200

APPLE_HSV_RANGES = [
    ((0, 20, 30), (10, 255, 255)),
    ((156, 20, 30), (180, 255, 255)),
    ((30, 20, 30), (85, 255, 255)),
    ((15, 20, 80), (35, 255, 255)),
]

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"