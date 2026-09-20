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
    # SVM / RF / scaler 已用「自有数据 + 按拍摄簇切分」重训（见
    # scripts/retrain_ml_clean.py）。干净验证集上 平衡准确率 95.44% / 94.46%，
    # 旧模型同口径只有 80.21% / 81.07%（旧模型文件仍在，改回上面三行即可回滚）。
    "scaler": ROOT_DIR / "apple_scaler_clean.pkl",
    "svm": ROOT_DIR / "apple_svm_model_clean.dat",
    "rf": ROOT_DIR / "apple_rf_model_clean.pkl",
    "dl": ROOT_DIR / "apple_dl_model.pth",
    "yolo_cls": ROOT_DIR / "yolo_output" / "yolo_classify-2" / "weights" / "best.pt",
    "yolo_detect": ROOT_DIR / "yolo_detect_best.pt",
}

IMAGE_WIDTH = 500
PREMIUM_THRESHOLD = 0.05
STANDARD_THRESHOLD = 0.15
DEFECT_SENSITIVITY = 5
DEFECT_MIN_AREA = 200

# AI 缺陷框保留门槛：只把置信度 >= 该值的 YOLO 缺陷框算作真实缺陷。
# 调低→召回更高但好果误报增多；调高→更保守。可用环境变量覆盖，便于实验对比。
AI_DEFECT_CONF_TH = float(os.environ.get("AI_DEFECT_CONF_TH", "0.50"))

# DL 热力图(Grad-CAM)需额外一次前向+反向，约占单图 ~250ms。
# 关闭后 DL 缩略图仍会生成，只是没有热力图叠加。
ENABLE_GRADCAM = os.environ.get("ENABLE_GRADCAM", "1").strip() not in ("0", "false", "False", "off")

APPLE_HSV_RANGES = [
    ((0, 20, 30), (10, 255, 255)),
    ((156, 20, 30), (180, 255, 255)),
    ((30, 20, 30), (85, 255, 255)),
    ((15, 20, 80), (35, 255, 255)),
]

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# ==================== 会话密钥 ====================
def _load_secret_key() -> str:
    """优先使用环境变量 APP_SECRET_KEY，否则持久化到 instance/secret_key.txt，保证会话重启不失效"""
    env_key = os.environ.get("APP_SECRET_KEY", "").strip()
    if env_key:
        return env_key
    key_file = INSTANCE_DIR / "secret_key.txt"
    try:
        if key_file.exists():
            key = key_file.read_text(encoding="utf-8").strip()
            if key:
                return key
        key = os.urandom(32).hex()
        INSTANCE_DIR.mkdir(exist_ok=True)
        key_file.write_text(key, encoding="utf-8")
        return key
    except Exception:
        return "zhi-guo-ai-qc-local-secret"


SECRET_KEY = _load_secret_key()

# ==================== Supabase（登录 / 用户数据） ====================
# 在 supabase.com 创建项目后，把项目地址和 anon key 填入环境变量：
#   SUPABASE_URL=https://xxxx.supabase.co
#   SUPABASE_ANON_KEY=eyJ...
# 未配置时系统自动进入「本地模式」，不启用登录。
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "").strip()

# ==================== 严重度与复核 ====================
SEVERITY_MILD_TH = 0.10        # 缺陷占比 <10% 为轻度
SEVERITY_MODERATE_TH = 0.30    # 缺陷占比 >=30% 为重度
REVIEW_CONF_LOW = 0.45         # 置信度灰区下限
REVIEW_CONF_HIGH = 0.85        # 置信度灰区上限
REVIEW_RATIO_MARGIN = 0.02     # 缺陷占比临界判定容差
AUTO_REVIEW_QUEUE = True       # 不确定结果默认自动进入复核队列
