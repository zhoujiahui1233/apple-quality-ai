import json
import os
import sys
import uuid
import logging
import threading
import logging.handlers
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, session, redirect
from flask_sqlalchemy import SQLAlchemy

import config
from apple_defect_detector import (
    preprocess, segment_apple, detect_defects, classify_quality,
    predict_with_svm, predict_with_rf, predict_with_dl, predict_with_yolo,
    visualize_result, save_result, visualize_svm, visualize_rf, visualize_dl,
    visualize_yolo, predict_with_yolo_detect, visualize_yolo_detect,
    load_scaler, load_svm_model, load_rf_model, load_dl_model,
    load_yolo_model, load_yolo_detect_model
)
import cv2
import numpy as np

from models import db, DetectionRecord, QualityRecord
from services import dataset_tools
from services import supabase_auth
from services import haccp_service

logger = logging.getLogger("app")

sys.path.insert(0, str(config.ROOT_DIR))
os.environ['YOLO_CONFIG_DIR'] = str(config.ROOT_DIR / '.ultralytics')

app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# ==================== 多果逐一定级 / 实时框选追踪 ====================
def _read_bgr_upload(file_storage):
    """读取上传图片为 BGR（兼容常见格式）"""
    data = file_storage.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _resize_long_side(img_bgr, max_side=1100):
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
    return img_bgr


def _ratio_severity(ratio):
    if ratio < config.SEVERITY_MILD_TH:
        return "轻度"
    if ratio < config.SEVERITY_MODERATE_TH:
        return "中度"
    return "重度"


def _clean_apple_boxes(dets, w, h):
    """过滤过小/疑似重复的苹果框，降低“多框不消失”的假阳性。
    规则：最小边长、最小面积占比、宽松 NMS 去重、数量上限。"""
    dets = dets or []
    out = []
    min_dim = max(22, int(min(w, h) * 0.06))
    min_area = (w * h) * 0.002
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        bw = x2 - x1; bh = y2 - y1
        if bw < min_dim or bh < min_dim:
            continue
        if bw * bh < min_area:
            continue
        out.append(d)
    if len(out) > 1:
        boxes = [[d["bbox"][0], d["bbox"][1],
                  d["bbox"][2] - d["bbox"][0],
                  d["bbox"][3] - d["bbox"][1]] for d in out]
        scores = [float(d.get("confidence", 0.5)) for d in out]
        keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0,
                                nms_threshold=0.35)
        if keep is not None and len(keep):
            keep = [k[0] if isinstance(k, (list, tuple)) else k for k in keep]
            out = [out[i] for i in keep]
    out.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    return out[:16]


def _multi_apple_analysis(img_bgr):
    """对一张图里的每个苹果逐一定级（多果模式）。
    返回: {w,h,apples:[{id,bbox,conf,grade,severity,defect_ratio}], worst}
    """
    from apple_defect_detector import predict_with_yolo_defect
    img = _resize_long_side(img_bgr)
    img_p, _ = preprocess(img)
    h, w = img_p.shape[:2]
    raw = predict_with_yolo_detect(img_p, conf_th=0.6)
    dets = _clean_apple_boxes(raw, w, h)
    apples = []
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        bw = max(1, x2 - x1); bh = max(1, y2 - y1)
        area = bw * bh
        conf = float(d.get("confidence", 0.0))
        ai = predict_with_yolo_defect(img_p, conf_th=0.25, region=(x1, y1, x2, y2))
        confirmed = [b for b in ai if b["confidence"] >= 0.50]
        defect_px = 0
        for b in confirmed:
            bx1, by1, bx2, by2 = b["bbox"]
            bx1 = max(x1, bx1); by1 = max(y1, by1)
            bx2 = min(x2, bx2); by2 = min(y2, by2)
            if bx2 > bx1 and by2 > by1:
                defect_px += (bx2 - bx1) * (by2 - by1)
        ratio = defect_px / area
        grade = "不合格" if ratio >= config.STANDARD_THRESHOLD else (
            "标准" if ratio >= config.PREMIUM_THRESHOLD else "合格")
        apples.append({
            "id": i,
            "bbox": [x1, y1, x2, y2],
            "conf": round(conf, 3),
            "grade": grade,
            "severity": _ratio_severity(ratio),
            "defect_ratio": round(ratio, 4),
        })
    bad = [a for a in apples if a["grade"] == "不合格"]
    worst = bad[0] if bad else (max(apples, key=lambda a: a["defect_ratio"])
                                if apples else None)
    return {"w": w, "h": h, "apples": apples, "worst": worst,
            "count": len(apples)}


def _frame_apple_boxes(img_bgr):
    """实时帧：只返回所有苹果框（快速），用于视频叠加与计数。"""
    img = _resize_long_side(img_bgr)
    img_p, _ = preprocess(img)
    h, w = img_p.shape[:2]
    raw = predict_with_yolo_detect(img_p, conf_th=0.6)
    dets = _clean_apple_boxes(raw, w, h)
    boxes = [{"bbox": [int(v) for v in d["bbox"]],
              "conf": round(float(d.get("confidence", 0.0)), 3)} for d in dets]
    return {"w": w, "h": h, "count": len(boxes), "boxes": boxes}


# ==================== 登录 / Supabase ====================
def _current_user():
    return session.get("sb_user") or {
        "id": None, "email": "", "nickname": "本地用户", "role": "guest",
    }


@app.context_processor
def _inject_current_user():
    return {"current_user": _current_user()}


def _auth_enabled():
    """登录是否启用：配置了 Supabase 且当前连得上。
    连不上（如免费项目被暂停）时自动退回本地模式，不影响使用。"""
    return supabase_auth.is_configured() and supabase_auth.is_online()


@app.before_request
def _auth_guard():
    """Supabase 未配置时进入本地模式（不启用登录）；配置后所有页面/接口需登录"""
    if not _auth_enabled():
        return
    p = request.path
    if p.startswith("/static") or p in ("/manifest.json", "/favicon.ico"):
        return
    if p in ("/login", "/register", "/logout"):
        if p != "/logout" and session.get("sb_user"):
            return redirect(url_for("index"))
        return
    if session.get("sb_user"):
        return
    if p.startswith("/api/"):
        return jsonify({"error": "未登录或会话已过期", "auth_required": True}), 401
    return redirect(url_for("login_page", next=p))


def _safe_next(nxt):
    if nxt and nxt.startswith("/") and not nxt.startswith("//"):
        return nxt
    return "/"


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if session.get("sb_user"):
        return redirect(url_for("index"))
    if not _auth_enabled():
        return render_template("login.html", mode="login", setup_required=True,
                               error=None, ok_msg=None)
    error = None
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        if not email or not password:
            error = "请输入邮箱和密码"
        else:
            ok, data = supabase_auth.sign_in(email, password)
            if ok:
                u = supabase_auth.user_view(data.get("user") or {})
                session["sb_user"] = u
                session["sb_token"] = data.get("access_token", "")
                _sync_profile(u)
                return redirect(_safe_next(request.args.get("next")))
            error = data.get("error") or "登录失败，请检查邮箱和密码"
    return render_template("login.html", mode="login", setup_required=False,
                           error=error, ok_msg=None)


@app.route('/register', methods=['GET', 'POST'])
def register_page():
    if not _auth_enabled():
        return redirect(url_for("login_page"))
    if session.get("sb_user"):
        return redirect(url_for("index"))
    error = None
    ok_msg = None
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        nickname = (request.form.get("nickname") or "").strip() or email.split("@")[0]
        if "@" not in email:
            error = "请输入有效邮箱"
        elif len(password) < 6:
            error = "密码至少需要 6 位"
        else:
            ok, data = supabase_auth.sign_up(email, password, nickname)
            if ok:
                if data.get("access_token"):
                    u = supabase_auth.user_view(data.get("user") or {})
                    session["sb_user"] = u
                    session["sb_token"] = data.get("access_token", "")
                    _sync_profile(u)
                    return redirect(url_for("index"))
                ok_msg = "注册成功！请前往邮箱点击验证链接完成激活，再返回登录。"
            else:
                error = data.get("error") or "注册失败"
    return render_template("login.html", mode="register", setup_required=False,
                           error=error, ok_msg=ok_msg)


@app.route('/logout')
def logout_page():
    token = session.pop("sb_token", None)
    session.pop("sb_user", None)
    if token and _auth_enabled():
        supabase_auth.sign_out(token)
    return redirect(url_for("login_page"))


def _sync_profile(u):
    """尽量同步 profiles 表（表未创建时自动忽略，不影响登录）"""
    try:
        okp, prof = supabase_auth.get_profile(u.get("id"))
        if okp and prof:
            session["sb_user"]["nickname"] = prof.get("nickname") or u.get("nickname")
            session["sb_user"]["role"] = prof.get("role") or u.get("role")
        else:
            supabase_auth.upsert_profile(u.get("id"), u.get("nickname"), u.get("role"))
    except Exception:
        pass


# ==================== 轻量数据库迁移 ====================
def _migrate_db():
    """为旧数据库补充新增字段（如 category）"""
    import sqlite3
    db_path = config.INSTANCE_DIR / "database.db"
    if not db_path.exists():
        return
    try:
        conn = sqlite3.connect(str(db_path))
        cols = [r[1] for r in conn.execute("PRAGMA table_info(detection_records)")]
        if "category" not in cols:
            conn.execute("ALTER TABLE detection_records ADD COLUMN category VARCHAR(20) DEFAULT 'apple'")
            conn.commit()
        for col, ddl in (("severity", "VARCHAR(10) DEFAULT '轻度'"),
                         ("needs_review", "BOOLEAN DEFAULT 0"),
                         ("review_reason", "VARCHAR(255) DEFAULT ''")):
            cols = [r[1] for r in conn.execute("PRAGMA table_info(detection_records)")]
            if col not in cols:
                conn.execute(f"ALTER TABLE detection_records ADD COLUMN {col} {ddl}")
                conn.commit()
        # sensory_records 新增列（感官系统扩展字段）
        scol = [r[1] for r in conn.execute("PRAGMA table_info(sensory_records)")]
        for col, ddl in (
            ("lab_l", "FLOAT"), ("lab_a", "FLOAT"), ("lab_b", "FLOAT"),
            ("color_coverage", "FLOAT"), ("fruit_diameter_mm", "FLOAT"), ("shape_index", "FLOAT"),
            ("ta", "FLOAT"), ("sugar_acid_ratio", "FLOAT"), ("vitamin_c", "FLOAT"),
            ("tpa_springiness", "FLOAT"), ("tpa_cohesiveness", "FLOAT"),
            ("tpa_chewiness", "FLOAT"), ("juice_pct", "FLOAT"),
            ("qda_sweet", "FLOAT"), ("qda_sour", "FLOAT"), ("qda_astringent", "FLOAT"),
            ("qda_aroma", "FLOAT"), ("qda_crispness", "FLOAT"), ("qda_juiciness", "FLOAT"),
        ):
            if col not in scol:
                conn.execute(f"ALTER TABLE sensory_records ADD COLUMN {col} {ddl}")
                conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"数据库迁移失败: {e}")


# ==================== 模型预热 ====================
def warmup_models():
    """启动时预加载所有模型，避免首次检测缓慢"""
    logger.info("正在加载模型...")
    try:
        load_scaler()
        logger.info("  [OK] Scaler ✓")
    except Exception as e:
        logger.warning(f"  [SKIP] Scaler: {e}")

    for name, loader in [
        ("SVM", load_svm_model),
        ("RF", load_rf_model),
        ("DL", load_dl_model),
        ("YOLO-CLS", load_yolo_model),
        ("YOLO-DET", load_yolo_detect_model),
    ]:
        try:
            loader()
            logger.info(f"  [OK] {name} \u2713")
        except Exception as e:
            logger.warning(f"  [SKIP] {name}: {e}")

    logger.info("模型预热完成")

app.json.ensure_ascii = False
app.config['SQLALCHEMY_DATABASE_URI'] = config.DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = str(config.UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB，支持批量导入照片

db.init_app(app)

from services.detection_service import DetectionService
from services.dataset_service import DatasetService
from services.model_registry import ModelRegistry
detection_service = DetectionService()
from services.industrial_service import IndustrialEngine
industrial_engine = IndustrialEngine(detection_service)
dataset_service = DatasetService()
model_registry = ModelRegistry()

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def _safe_component(name):
    """校验品类/标签/文件名，拒绝路径穿越与分隔符"""
    if not name or name in (".", ".."):
        return False
    if "/" in name or "\\" in name or os.path.isabs(name):
        return False
    return True


def _resolve_under(base: Path, rel: str):
    """把 rel 解析到 base 下，越界返回 None"""
    try:
        target = (base / rel).resolve()
        target.relative_to(base.resolve())
        return target
    except Exception:
        return None


def _module_context():
    """读取 URL 参数 module，返回 (module, default_category)。
    surface = 外观模块，slice = 切片模块（category 固定 apple_slice）"""
    module = request.args.get('module', 'surface')
    if module == 'slice':
        return 'slice', 'apple_slice'
    return 'surface', ''



@app.route('/')
def index():
    return render_template('index.html', active_tab='monitor', module='surface', default_category='')


@app.route('/realtime')
def realtime_page():
    """摄像头实时多果框选追踪 + 单图多果逐一定级页面"""
    return render_template("realtime.html", active_tab="realtime")


@app.route('/api/frame', methods=['POST'])
def api_frame():
    """实时帧快速苹果框选（视频叠加用）"""
    f = request.files.get('file')
    if f is None:
        return jsonify({"error": "缺少图片"}), 400
    img = _read_bgr_upload(f)
    if img is None:
        return jsonify({"error": "无法解析图片"}), 400
    return jsonify(_frame_apple_boxes(img))


@app.route('/api/multi_apple', methods=['POST'])
def api_multi_apple():
    """单图多果：逐个苹果定级"""
    f = request.files.get('file')
    if f is None:
        return jsonify({"error": "缺少图片"}), 400
    img = _read_bgr_upload(f)
    if img is None:
        return jsonify({"error": "无法解析图片"}), 400
    try:
        return jsonify(_multi_apple_analysis(img))
    except Exception as e:
        logger.warning("多果分析失败: %s", e)
        return jsonify({"error": "分析失败：" + str(e)[:120]}), 500


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': '未找到图片文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式，请上传 JPG/PNG/BMP 图片'}), 400

    category = request.form.get('category', 'apple')
    auto_review = request.form.get('auto_review', '1') == '1'
    tuning = {
        'sensitivity': request.form.get('sensitivity', 5),
        'min_area': request.form.get('min_area', 200),
        'premium_th': request.form.get('premium_th', 0.05),
        'standard_th': request.form.get('standard_th', 0.15),
        'white_balance': request.form.get('white_balance', '0'),
    }

    try:
        result = detection_service.process_upload(file, category, tuning, auto_review=auto_review)
        if 'error' in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        logger.error(f"检测失败: {e}")
        return jsonify({'error': f'检测失败: {str(e)}'}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '图片过大，请上传 64MB 以内的图片'}), 413


@app.route('/industrial')
def industrial():
    """产线模拟模式：看板 + 模拟控制台"""
    return render_template('industrial.html', active_tab='industrial', module='surface')


@app.route('/api/industrial/start', methods=['POST'])
def industrial_start():
    data = request.get_json(silent=True) or {}
    try:
        speed = float(data.get('speed', 2))
        faces = int(data.get('faces_per_apple', 3))
    except (TypeError, ValueError):
        return jsonify({'error': '参数无效'}), 400
    speed = max(0.5, min(10, speed))
    faces = max(2, min(8, faces))
    res = industrial_engine.start(speed=speed, faces_per_apple=faces)
    if 'error' in res:
        return jsonify(res), 400
    return jsonify(res)


@app.route('/api/industrial/stop', methods=['POST'])
def industrial_stop():
    return jsonify(industrial_engine.stop())


@app.route('/api/industrial/single', methods=['POST'])
def industrial_single():
    data = request.get_json(silent=True) or {}
    try:
        faces = int(data.get('faces_per_apple', 3))
    except (TypeError, ValueError):
        faces = 3
    log = industrial_engine.single_step(faces)
    return jsonify({'ok': True, 'log': log, 'stats': industrial_engine.stats()})


@app.route('/api/industrial/stats')
def industrial_stats():
    return jsonify(industrial_engine.stats())


@app.route('/api/multi_detect', methods=['POST'])
def multi_detect():
    """多角度检测：同一苹果多个面（2~8 张），逐面检测，取最差判定（防漏检）"""
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': '未找到图片文件'}), 400
    category = request.form.get('category', 'apple')
    auto_review = request.form.get('auto_review', '1') == '1'
    tuning = {
        'sensitivity': request.form.get('sensitivity', 5),
        'min_area': request.form.get('min_area', 200),
        'premium_th': request.form.get('premium_th', 0.05),
        'standard_th': request.form.get('standard_th', 0.15),
        'white_balance': request.form.get('white_balance', '0'),
    }
    try:
        result = detection_service.process_multi_upload(files, category, tuning, auto_review=auto_review)
        if 'error' in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        logger.error(f"多角度检测失败: {e}")
        return jsonify({'error': f'多角度检测失败: {str(e)}'}), 500


@app.route('/history')
def history():
    page = request.args.get('page', 1, type=int)
    per_page = 12
    grade_filter = request.args.get('grade', '')
    module, default_category = _module_context()
    if module == 'slice':
        category = default_category  # 切片模块品类固定 apple_slice，避免混入外观数据
    else:
        category = request.args.get('category', default_category)
    review_filter = request.args.get('review', '')

    query = DetectionRecord.query.order_by(DetectionRecord.created_at.desc())
    if review_filter == '1':
        query = query.filter(DetectionRecord.needs_review == True)
    if category:
        query = query.filter(DetectionRecord.category == category)
    if grade_filter:
        query = query.filter(DetectionRecord.grade.contains(grade_filter))

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    records = pagination.items

    return render_template('history.html', active_tab='history', records=records, pagination=pagination, grade_filter=grade_filter, category=category, review_filter=review_filter, module=module, default_category=default_category)


@app.route('/detail/<int:record_id>')
def detail(record_id):
    record = DetectionRecord.query.get_or_404(record_id)
    return render_template('detail.html', record=record)


@app.route('/delete/<int:record_id>', methods=['POST'])
def delete(record_id):
    record = DetectionRecord.query.get_or_404(record_id)
    try:
        _delete_record_files(record)
        db.session.delete(record)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _grade_class(grade: str) -> str:
    """等级字符串 -> premium/standard/reject（兼容中英文标签）"""
    g = grade or ""
    if "Premium" in g or "优等" in g:
        return "premium"
    if "Reject" in g or "不合格" in g:
        return "reject"
    return "standard"


def _review_original_name(name: str):
    """复核队列文件名 review_<hash>_<uuid>.<ext> -> 原记录文件名；非复核文件返回 None"""
    if name and name.startswith("review_"):
        idx = name.find("_", len("review_"))
        if idx != -1:
            return name[idx + 1:]
    return None


def _corrected_grade(old_grade: str, label: str):
    """\u4fdd\u5b58\u5230\u6570\u636e\u96c6\u65f6\uff0c\u6839\u636e\u624b\u52a8\u4fee\u6b63\u7684\u6807\u7b7e\u540c\u6b65\u4fee\u6b63\u5386\u53f2\u8bb0\u5f55\u7684\u7b49\u7ea7\u4e0e\u989c\u8272\uff08\u517c\u5bb9\u4e2d\u82f1\u6587\u683c\u5f0f\uff09"""
    label = (label or '').strip()
    if label not in ('\u5408\u683c', '\u4e0d\u5408\u683c'):
        return None, None
    has_en = old_grade and any(k in old_grade for k in ('Premium', 'Standard', 'Reject'))
    if label == '\u5408\u683c':
        return ('\u5408\u683c Standard' if has_en else '\u5408\u683c'), '(0, 200, 0)'
    return ('\u4e0d\u5408\u683c Reject' if has_en else '\u4e0d\u5408\u683c'), '(0, 0, 255)'


def _delete_record_files(record):
    """删除记录关联的所有图片文件（原图/结果图/四模型可视化/YOLO检测图）"""
    names = []
    if record.filename:
        names.append(record.filename)
        # 四模型可视化 + YOLO检测图与上传文件同名前缀
        for prefix in ("svm_", "rf_", "dl_", "yolo_", "detect_"):
            names.append(prefix + record.filename)
    if record.result_filename:
        names.append(record.result_filename)
    for fname in names:
        if not fname:
            continue
        # 防止路径穿越：仅允许文件名本身
        if os.path.basename(fname) != fname:
            continue
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass


def _apply_record_filter(query, filt):
    """按筛选条件筛选记录：category / grade / date_from / date_to"""
    from datetime import timedelta
    filt = filt or {}
    category = filt.get("category")
    if category:
        query = query.filter(DetectionRecord.category == category)
    grade = filt.get("grade")
    if grade:
        if grade == "premium":
            query = query.filter(DetectionRecord.grade.contains("Premium"))
        elif grade == "standard":
            query = query.filter(DetectionRecord.grade.contains("Standard"))
        elif grade == "reject":
            query = query.filter(DetectionRecord.grade.contains("Reject"))
    date_from = filt.get("date_from")
    if date_from:
        query = query.filter(DetectionRecord.created_at >= datetime.strptime(date_from, "%Y-%m-%d"))
    date_to = filt.get("date_to")
    if date_to:
        query = query.filter(
            DetectionRecord.created_at < datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1))
    return query


def _select_records(data):
    """支持 ids 或 filter 两种选择方式"""
    ids = data.get("ids")
    query = DetectionRecord.query
    if ids:
        query = query.filter(DetectionRecord.id.in_(ids))
    else:
        query = _apply_record_filter(query, data.get("filter"))
    return query.all()


@app.route('/api/records/count', methods=['POST'])
def records_count():
    """按筛选条件返回匹配记录数（用于批量管理前确认）"""
    data = request.get_json(silent=True) or {}
    query = _apply_record_filter(DetectionRecord.query, data.get("filter"))
    return jsonify({"count": query.count()})


@app.route('/api/records/batch_delete', methods=['POST'])
def batch_delete():
    data = request.get_json(silent=True) or {}
    if not data.get("ids") and not data.get("filter") and not data.get("delete_all"):
        return jsonify({"error": "请提供 ids 或筛选条件"}), 400
    records = _select_records(data)
    n = len(records)
    for r in records:
        _delete_record_files(r)
        db.session.delete(r)
    db.session.commit()
    return jsonify({"success": True, "deleted": n})


@app.route('/api/records/batch_update', methods=['POST'])
def batch_update():
    data = request.get_json(silent=True) or {}
    if not data.get("ids") and not data.get("filter"):
        return jsonify({"error": "请提供 ids 或筛选条件"}), 400
    updates = data.get("updates") or {}
    grade = updates.get("grade")
    category = updates.get("category")
    if not grade and not category:
        return jsonify({"error": "没有要更新的字段"}), 400
    GRADE_MAP = {
        "premium": "优等 Premium",
        "standard": "合格 Standard",
        "reject": "不合格 Reject",
    }
    records = _select_records(data)
    n = len(records)
    for r in records:
        if grade:
            r.grade = GRADE_MAP.get(grade, grade)
        if category:
            r.category = category
    db.session.commit()
    return jsonify({"success": True, "updated": n})


@app.route('/api/records/mark_reviewed', methods=['POST'])
def records_mark_reviewed():
    """将选中记录标记为已复核（清除待复核标记与原因）"""
    data = request.get_json(silent=True) or {}
    if not data.get("ids") and not data.get("filter"):
        return jsonify({"error": "请提供 ids 或筛选条件"}), 400
    records = _select_records(data)
    n = 0
    for r in records:
        if r.needs_review:
            r.needs_review = False
            r.review_reason = ""
            n += 1
    db.session.commit()
    return jsonify({"success": True, "cleared": n})


@app.route('/stats')
def stats_page():
    module, default_category = _module_context()
    return render_template('stats.html', active_tab='stats', module=module, default_category=default_category)


@app.route('/api/stats')
def stats():
    category = request.args.get('category', '')
    query = DetectionRecord.query
    if category:
        query = query.filter(DetectionRecord.category == category)
    records = query.all()
    total = len(records)
    counts = {"premium": 0, "standard": 0, "reject": 0}
    for r in records:
        counts[_grade_class(r.grade)] += 1
    return jsonify({
        'total': total,
        'premium': counts["premium"],
        'standard': counts["standard"],
        'reject': counts["reject"],
    })



@app.route('/api/stats/daily')
def stats_daily():
    from datetime import datetime, timedelta
    from models import DetectionRecord
    import collections
    days = request.args.get('days', 7, type=int)
    category = request.args.get('category', '')
    today = datetime.now().date()
    daily = collections.OrderedDict()
    for i in range(days - 1, -1, -1):
        d = today - timedelta(days=i)
        key = d.isoformat()
        daily[key] = {'total': 0, 'premium': 0, 'standard': 0, 'reject': 0}
    query = DetectionRecord.query.filter(
        DetectionRecord.created_at >= (today - timedelta(days=days-1)))
    if category:
        query = query.filter(DetectionRecord.category == category)
    records = query.all()
    for r in records:
        key = r.created_at.date().isoformat()
        if key in daily:
            daily[key]['total'] += 1
            cls = _grade_class(r.grade)
            if cls == "premium":
                daily[key]['premium'] += 1
            elif cls == "reject":
                daily[key]['reject'] += 1
            else:
                daily[key]['standard'] += 1
    return jsonify(list(daily.values()))

def _unlabeled_files(category: str = ""):
    """聚合未标注图片。category 非空时只聚合该品类。"""
    base = config.ROOT_DIR / "dataset_collected"
    if category:
        candidates = [base / category / "unlabeled"]
    else:
        candidates = [base / "unlabeled"]
        # 外观模块仅聚合外观品类，切片 apple_slice 独立走切片模块
        for cat in ("apple", "bread", "beverage"):
            candidates.append(base / cat / "unlabeled")
    files = []
    for d in candidates:
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    files.append(f)
    return files


@app.route('/annotate')
def annotate_page():
    module, default_category = _module_context()
    return render_template('annotate.html', active_tab='annotate', module=module, default_category=default_category)


@app.route('/api/annotate/queue')
def annotate_queue():
    _module, _default_category = _module_context()
    files = _unlabeled_files(_default_category)
    items = [{"path": str(f.relative_to(config.ROOT_DIR / "dataset_collected")).replace("\\", "/"),
              "name": f.name, "from_review": f.name.startswith("review_")} for f in files]
    return jsonify({"items": items, "count": len(items)})


@app.route('/api/annotate/save', methods=['POST'])
def annotate_save():
    import shutil
    data = request.get_json(silent=True) or {}
    src_rel = data.get("src", "")
    category = data.get("category", "")
    label = data.get("label", "")
    if not (src_rel and category and label):
        return jsonify({"error": "参数不完整"}), 400
    if not (_safe_component(category) and _safe_component(label)):
        return jsonify({"error": "非法的品类或标签"}), 400
    base = config.ROOT_DIR / "dataset_collected"
    src = _resolve_under(base, src_rel)
    if src is None or not src.is_file():
        return jsonify({"error": "文件不存在"}), 404
    dst_dir = base / category / label
    dst_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    dst = dst_dir / f"{label}_{ts}{src.suffix.lower()}"
    shutil.move(str(src), str(dst))
    logger.info("标注: %s -> %s/%s/%s", src_rel, category, label, dst.name)
    # 来自检测复核队列的图片：同步修正历史记录等级并清除待复核标记
    synced = False
    original = _review_original_name(src.name)
    if original:
        rec = DetectionRecord.query.filter_by(filename=original).first()
        if rec is not None:
            new_grade, new_color = _corrected_grade(rec.grade, label)
            if new_grade:
                rec.grade = new_grade
                rec.grade_color = new_color
            rec.needs_review = False
            rec.review_reason = ""
            db.session.commit()
            synced = True
            logger.info("复核同步: 记录 #%s 等级->%s，清除待复核", rec.id, new_grade or rec.grade)
    _module, _default_category = _module_context()
    return jsonify({"success": True, "filename": dst.name, "synced": synced, "remaining": len(_unlabeled_files(_default_category))})


@app.route('/api/annotate/skip', methods=['POST'])
def annotate_skip():
    import shutil
    data = request.get_json(silent=True) or {}
    src_rel = data.get("src", "")
    if not src_rel:
        return jsonify({"error": "参数不完整"}), 400
    base = config.ROOT_DIR / "dataset_collected"
    src = _resolve_under(base, src_rel)
    if src is not None and src.is_file():
        skip_dir = base / "_skipped"
        skip_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(skip_dir / src.name))
    _module, _default_category = _module_context()
    return jsonify({"success": True, "remaining": len(_unlabeled_files(_default_category))})


@app.route('/annotations')
def browse_annotations():
    import glob, shutil
    preview_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo_detect_dataset', 'preview_annotations')
    preview_dst = os.path.join(app.config['UPLOAD_FOLDER'], '_annotations')
    
    # Copy preview images to static folder on first access
    if os.path.exists(preview_src) and not os.path.exists(preview_dst):
        os.makedirs(preview_dst, exist_ok=True)
        for fname in os.listdir(preview_src):
            shutil.copy2(os.path.join(preview_src, fname), os.path.join(preview_dst, fname))
    
    images = []
    if os.path.exists(preview_dst):
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        for ext in exts:
            images.extend(glob.glob(os.path.join(preview_dst, ext)))
        images = sorted([os.path.basename(p) for p in images])
    
    module, default_category = _module_context()
    return render_template('annotations.html', images=images, count=len(images), module=module, default_category=default_category)



@app.route('/api/export/report')
def export_report():
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from io import BytesIO
    from flask import send_file

    module, _dc = _module_context()
    records_q = DetectionRecord.query.order_by(DetectionRecord.created_at.desc())
    if module == 'slice':
        records_q = records_q.filter(DetectionRecord.category == 'apple_slice')
    records = records_q.all()

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "检测报告"

    # Header styling
    header_font = Font(name="Microsoft YaHei", bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="1a237e", end_color="1a237e", fill_type="solid")
    header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    thin_border = Border(
        left=Side(style="thin"),
        right=Side(style="thin"),
        top=Side(style="thin"),
        bottom=Side(style="thin")
    )

    headers = ["ID", "文件名", "等级", "严重度", "缺陷占比", "缺陷数量", "待复核",
               "苹果面积(px)", "缺陷面积(px)", "检测时间"]
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = thin_border

    # Data rows
    data_font = Font(name="Microsoft YaHei", size=10)
    data_align = Alignment(horizontal="center", vertical="center")
    grade_colors = {
        "Premium": PatternFill(start_color="00ff88", end_color="00ff88", fill_type="solid"),
        "Standard": PatternFill(start_color="ffc107", end_color="ffc107", fill_type="solid"),
        "Reject": PatternFill(start_color="ff4757", end_color="ff4757", fill_type="solid"),
    }

    for i, r in enumerate(records, 2):
        values = [
            r.id, r.filename, r.grade,
            r.severity,
            f"{r.defect_ratio * 100:.2f}%",
            r.defect_count,
            "是" if r.needs_review else "",
            r.apple_area, r.defect_area,
            r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else ""
        ]
        for col, v in enumerate(values, 1):
            cell = ws.cell(row=i, column=col, value=v)
            cell.font = data_font
            cell.alignment = data_align
            cell.border = thin_border
            # Color grade column
            if col == 3:
                for keyword, fill in grade_colors.items():
                    if keyword in str(r.grade):
                        cell.fill = fill
                        break

    # Column widths
    widths = [8, 30, 15, 10, 12, 10, 10, 14, 14, 20]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + i)].width = w

    # Summary sheet
    ws2 = wb.create_sheet(title="统计摘要")
    total = len(records)
    premium = sum(1 for r in records if "Premium" in str(r.grade))
    standard = sum(1 for r in records if "Standard" in str(r.grade))
    reject = sum(1 for r in records if "Reject" in str(r.grade))
    review = sum(1 for r in records if r.needs_review)

    summary = [
        ["指标", "数值"],
        ["累计检测", total],
        ["优质", premium],
        ["标准", standard],
        ["不合格", reject],
        ["优质率", f"{premium / total * 100:.1f}%" if total else "0%"],
        ["不合格率", f"{reject / total * 100:.1f}%" if total else "0%"],
        ["待复核", review],
    ]

    for row_data in summary:
        ws2.append(row_data)

    for col in range(1, 3):
        cell = ws2.cell(row=1, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = thin_border

    ws2.column_dimensions["A"].width = 15
    ws2.column_dimensions["B"].width = 15

    output = BytesIO()
    wb.save(output)
    output.seek(0)

    _mod_name = '切片检测报告' if module == 'slice' else '苹果品控报告'
    filename = f"{_mod_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=filename
    )




# ==================== HACCP 品控视图 ====================
@app.route('/haccp')
def haccp_page():
    module, default_category = _module_context()
    return render_template('haccp.html', active_tab='haccp', module=module, default_category=default_category,
                           today=datetime.now().strftime('%Y-%m-%d'))


@app.route('/api/haccp/overview')
def haccp_overview():
    category = request.args.get('category', '')
    if category not in haccp_service.HACCP_FLOWS:
        category = 'apple'
    period = request.args.get('period', '') or datetime.now().strftime('%Y-%m-%d')
    return jsonify(haccp_service.overview(category, period))


@app.route('/api/haccp/records')
def haccp_records():
    category = request.args.get('category', '')
    query = QualityRecord.query.order_by(QualityRecord.created_at.desc())
    if category:
        query = query.filter(QualityRecord.category == category)
    records = query.limit(50).all()
    return jsonify([r.to_dict() for r in records])


@app.route('/api/haccp/generate', methods=['POST'])
def haccp_generate():
    data = request.get_json(silent=True) or {}
    category = data.get('category', 'apple')
    if category not in haccp_service.HACCP_FLOWS:
        category = 'apple'
    period = data.get('period', '') or datetime.now().strftime('%Y-%m-%d')
    operator = (_current_user().get('nickname') or _current_user().get('email')
                or '\u672c\u5730\u7528\u6237')
    rec = haccp_service.generate_record(category, period, operator)
    return jsonify({'success': True, 'record': rec.to_dict()})


@app.route('/api/haccp/export/<int:record_id>')
def haccp_export(record_id):
    from flask import send_file
    rec = QualityRecord.query.get_or_404(record_id)
    buf, filename = haccp_service.export_record_excel(rec)
    return send_file(
        buf,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=filename
    )


@app.route('/manifest.json')
def manifest_json():
    return send_from_directory(str(config.STATIC_DIR), 'manifest.json')


@app.route('/api/collect/preview', methods=['POST'])
def collect_preview():
    """\u667a\u80fd\u5206\u6790\uff1a\u8d28\u91cf\u8bc4\u5206 + \u53bb\u91cd + \u88c1\u526a\u9884\u89c8\uff08\u4e0d\u4fdd\u5b58\uff09"""
    if 'file' not in request.files:
        return jsonify({'error': '\u672a\u627e\u5230\u56fe\u7247'}), 400
    file = request.files['file']
    category = request.form.get('category', 'apple')
    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': '\u65e0\u6cd5\u89e3\u6790\u56fe\u7247'}), 400

    quality = dataset_tools.quality_check(img)
    dedup = dataset_tools.find_duplicates(img, config.ROOT_DIR / 'dataset_collected')
    crop_img, ratio = dataset_tools.crop_subject(img, category)

    preview_url = None
    if crop_img is not None:
        preview_dir = config.UPLOAD_DIR / '_previews'
        preview_dir.mkdir(parents=True, exist_ok=True)
        pname = f'preview_{uuid.uuid4().hex[:10]}.jpg'
        if dataset_tools.save_cropped(crop_img, preview_dir / pname, 300):
            preview_url = url_for('static', filename=f'uploads/_previews/{pname}')

    return jsonify({
        'quality': quality,
        'dedup': dedup,
        'crop': {'available': crop_img is not None, 'ratio': ratio,
                 'preview_url': preview_url},
    })


@app.route('/api/collect', methods=['POST'])
def collect_image():
    """\u4fdd\u5b58\u56fe\u7247\u5230\u6570\u636e\u96c6\uff1a\u652f\u6301\u667a\u80fd\u9009\u9879\uff08\u88c1\u526a/\u8d28\u91cf\u68c0\u67e5/\u53bb\u91cd/\u589e\u5f3a/\u7f3a\u9677\u7ec6\u5206\uff09"""
    if 'file' not in request.files:
        return jsonify({'error': '\u672a\u627e\u5230\u56fe\u7247'}), 400
    file = request.files['file']
    category = request.form.get('category', 'unlabeled')
    label = request.form.get('label', 'unlabeled')
    defect_type = request.form.get('defect_type', '')
    record_id = request.form.get('record_id', type=int)
    try:
        opts = json.loads(request.form.get('options', '{}') or '{}')
    except Exception:
        opts = {}
    if file.filename == '':
        return jsonify({'error': '\u6587\u4ef6\u540d\u4e3a\u7a7a'}), 400

    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': '\u65e0\u6cd5\u89e3\u6790\u56fe\u7247'}), 400

    report = {'quality': None, 'dedup': None, 'crop': False, 'augment': []}

    if opts.get('quality_check'):
        q = dataset_tools.quality_check(img)
        report['quality'] = q
        if not q['ok'] and not opts.get('force'):
            return jsonify({'error': '\u56fe\u7247\u8d28\u91cf\u8fc7\u4f4e\uff08\u6a21\u7cca/\u66dd\u5149\u5f02\u5e38\uff09\uff0c\u5df2\u963b\u6b62\u4fdd\u5b58\u3002\u53ef\u53d6\u6d88\u52fe\u9009\u8d28\u91cf\u68c0\u67e5\u6216\u52fe\u9009\u201c\u5f3a\u5236\u4fdd\u5b58\u201d\u3002',
                            'quality': q}), 400

    if opts.get('dedup'):
        dup = dataset_tools.find_duplicates(img, config.ROOT_DIR / 'dataset_collected')
        report['dedup'] = dup
        if dup['duplicates'] and not opts.get('force'):
            return jsonify({'error': '\u53d1\u73b0\u7591\u4f3c\u91cd\u590d\u56fe\u7247\uff0c\u5df2\u963b\u6b62\u4fdd\u5b58\u3002\u53ef\u52fe\u9009\u201c\u5f3a\u5236\u4fdd\u5b58\u201d\u3002',
                            'dedup': dup}), 400

    if not (_safe_component(category) and _safe_component(label)):
        return jsonify({'error': '非法的品类或标签'}), 400
    base = config.ROOT_DIR / 'dataset_collected'
    label_dir = base / category / label
    label_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    label_part = label if label != 'unlabeled' else 'unlabeled'
    if label == '\u4e0d\u5408\u683c' and defect_type:
        label_part = f'\u4e0d\u5408\u683c_{defect_type}'
    stem = f'{label_part}_{ts}'

    saved = []
    if opts.get('crop'):
        crop_img, _ratio = dataset_tools.crop_subject(img, category)
        if crop_img is not None:
            fname = f'{stem}_crop.jpg'
            if dataset_tools.save_cropped(crop_img, label_dir / fname):
                saved.append(fname)
                report['crop'] = True

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ('.jpg', '.jpeg', '.png', '.bmp'):
        ext = '.jpg'
    orig_fname = f'{stem}{ext}'
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(label_dir / orig_fname))
        saved.append(orig_fname)

    if opts.get('augment'):
        n = int(opts.get('augment_count', 4) or 4)
        aug_files = dataset_tools.save_augmented(img, stem, label_dir, n)
        report['augment'] = aug_files
        saved.extend(aug_files)

    history_updated = False
    if record_id:
        rec = db.session.get(DetectionRecord, record_id)
        if rec is not None:
            new_grade, new_color = _corrected_grade(rec.grade, label)
            if new_grade:
                rec.grade = new_grade
                rec.grade_color = new_color
                db.session.commit()
                history_updated = True

    # \u591a\u6a21\u6001\u5173\u8054\uff1a\u63d0\u4f9b sample_id \u65f6\uff0c\u539f\u56fe\u540c\u6b65\u590d\u5236\u5230 multimodal_data/visual/{sample_id}.jpg
    visual_saved = None
    sample_id = (request.form.get('sample_id') or '').strip()
    if sample_id:
        import shutil as _shutil
        vdir = config.ROOT_DIR / 'multimodal_data' / 'visual'
        try:
            vdir.mkdir(parents=True, exist_ok=True)
            vname = f"{sample_id}{ext}"
            src = label_dir / orig_fname
            dst = vdir / vname
            if src.exists() and not dst.exists():
                _shutil.copy2(str(src), str(dst))
                visual_saved = vname
        except Exception as e:
            logger.warning("\u6837\u672c\u7f16\u53f7\u5173\u8054\u4fdd\u5b58\u5931\u8d25: %s", e)

    logger.info("\u6570\u636e\u96c6\u6536\u96c6: [%s/%s] %s", category, label, saved)
    return jsonify({'success': True, 'category': category, 'label': label,
                    'files': saved, 'report': report,
                    'history_updated': history_updated,
                    'visual_saved': visual_saved})

@app.route('/api/dataset/list')
def list_dataset():
    """数据集列表: 品类 → 标签 → 图片（可按 category 过滤）"""
    category = request.args.get('category', '')
    data = dataset_service.list()
    if category:
        if category in data.get("categories", {}):
            data = {"categories": {category: data["categories"][category]},
                    "total": sum(len(v) for v in data["categories"][category].values())}
        else:
            # 请求的品类无数据时返回空，避免把其他品类（如整果）显示出来
            data = {"categories": {}, "total": 0}
    return jsonify(data)


@app.route('/api/dataset/delete', methods=['POST'])
def delete_dataset_item():
    category = request.form.get('category', '')
    label = request.form.get('label', '')
    filename = request.form.get('filename', '')
    if not (category and label and filename):
        return jsonify({'error': '参数不完整'}), 400
    ok = dataset_service.delete(category, label, filename)
    if not ok:
        return jsonify({'error': '文件不存在或参数非法'}), 404
    return jsonify({'success': True})



@app.route('/api/dataset/batch_delete', methods=['POST'])
def dataset_batch_delete():
    """批量删除数据集图片：同步删除数据集文件夹中对应文件"""
    category = request.form.get('category', '')
    if not _safe_component(category):
        return jsonify({'error': '非法品类'}), 400
    try:
        items = json.loads(request.form.get('items', '[]') or '[]')
    except Exception:
        return jsonify({'error': '参数无效'}), 400
    if not isinstance(items, list) or not items:
        return jsonify({'error': '未选择图片'}), 400
    if len(items) > 200:
        return jsonify({'error': '单批最多 200 张'}), 400
    deleted = 0
    failed = []
    for it in items:
        if not isinstance(it, dict):
            failed.append({'label': '', 'filename': str(it), 'reason': '参数无效'})
            continue
        label = str(it.get('label', ''))
        filename = str(it.get('filename', ''))
        cat = str(it.get('category', '') or category)
        if not _safe_component(cat):
            failed.append({'label': label, 'filename': filename, 'reason': '非法品类'})
            continue
        if dataset_service.delete(cat, label, filename):
            deleted += 1
        else:
            failed.append({'label': label, 'filename': filename,
                           'reason': '文件不存在或参数非法'})
    logger.info('数据集批量删除: [%s] 成功 %d, 失败 %d',
                category, deleted, len(failed))
    return jsonify({'success': True, 'deleted': deleted, 'failed': failed})


@app.route('/api/dataset/move', methods=['POST'])
def move_dataset_image():
    """修正标签：移动图片到另一个标签（合格/不合格）"""
    category = request.form.get('category', '')
    filename = request.form.get('filename', '')
    from_label = request.form.get('from_label', '')
    to_label = request.form.get('to_label', '')
    if not (category and filename and from_label and to_label):
        return jsonify({'error': '参数不完整'}), 400
    ok, info = dataset_service.move(category, filename, from_label, to_label)
    if not ok:
        return jsonify({'error': info}), 404
    logger.info("数据集修正: [%s/%s] %s -> %s (%s)", category, from_label, filename, to_label, info)
    return jsonify({'success': True, 'filename': info})

def _auto_bbox(img):
    """自动分割主物体，返回归一化 bbox (xc, yc, w, h)；未检出返回 None"""
    ih, iw = img.shape[:2]
    mask, _ = segment_apple(cv2.GaussianBlur(img, (5, 5), 0))
    if mask is None:
        return None
    co, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not co:
        return None
    largest = max(co, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.003 * ih * iw:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    return (x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih


def _preview_with_box(img, bbox, max_side=420):
    """\u751f\u6210\u5e26\u6846\u9884\u89c8\u56fe\u7684 base64 dataURL\uff08\u7528\u4e8e\u6279\u91cf\u5bfc\u5165\u9884\u89c8\uff09"""
    import base64
    draw = img.copy()
    if bbox and len(bbox) == 4:
        ih, iw = draw.shape[:2]
        x = int(max(0, (bbox[0] - bbox[2] / 2) * iw))
        y = int(max(0, (bbox[1] - bbox[3] / 2) * ih))
        w = int(min(iw - x, bbox[2] * iw))
        h = int(min(ih - y, bbox[3] * ih))
        cv2.rectangle(draw, (x, y), (x + w, y + h), (0, 212, 255), 2)
        cv2.putText(draw, 'auto', (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 212, 255), 1)
    hh, ww = draw.shape[:2]
    scale = max_side / max(hh, ww)
    if scale < 1:
        draw = cv2.resize(draw, (int(ww * scale), int(hh * scale)))
    ok, buf = cv2.imencode('.jpg', draw, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    return 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('ascii')


@app.route('/api/import/segment', methods=['POST'])
def import_segment():
    """\u6279\u91cf\u5bfc\u5165\uff1a\u81ea\u52a8\u5206\u5272\u751f\u6210 bbox\uff0c\u8fd4\u56de\u5e26\u6846\u9884\u89c8"""
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': '\u672a\u9009\u62e9\u56fe\u7247'}), 400
    if len(files) > 60:
        return jsonify({'error': '\u5355\u6279\u6700\u591a 60 \u5f20'}), 400
    items = []
    for f in files:
        data = f.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            items.append({'name': f.filename, 'w': 0, 'h': 0, 'bbox': None, 'preview': None})
            continue
        ih, iw = img.shape[:2]
        bbox = _auto_bbox(img)
        items.append({'name': f.filename, 'w': iw, 'h': ih, 'bbox': bbox,
                      'preview': _preview_with_box(img, bbox)})
    return jsonify({'success': True, 'items': items})


@app.route('/api/dataset/batch_import', methods=['POST'])
def dataset_batch_import():
    """批量导入照片（不检测）：人工选择 合格/不合格 与 不合格种类，直接入库"""
    category = request.form.get('category', '')
    if not _safe_component(category):
        return jsonify({'error': '非法品类'}), 400
    files = request.files.getlist('files')
    labels = request.form.getlist('labels')
    defects = request.form.getlist('defects')
    try:
        boxes = json.loads(request.form.get('boxes', 'null') or 'null')
    except Exception:
        boxes = None
    if boxes is not None and not isinstance(boxes, list):
        boxes = None
    if not files:
        return jsonify({'error': '未选择图片'}), 400
    if len(files) > 60:
        return jsonify({'error': '单批最多 60 张'}), 400
    qc_enabled = request.form.get('quality_check', '') in ('1', 'true', 'on')
    dup_enabled = request.form.get('dedup', '') in ('1', 'true', 'on')
    force_enabled = request.form.get('force', '') in ('1', 'true', 'on')

    base = config.ROOT_DIR / 'dataset_collected'
    results = []
    ts_base = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    for i, f in enumerate(files):
        label = (labels[i] if i < len(labels) else '合格').strip()
        defect = (defects[i] if i < len(defects) else '').strip()
        if label not in ('合格', '不合格'):
            results.append({'name': f.filename, 'status': 'error', 'reason': '标签非法'})
            continue
        label_part = label
        if label == '不合格' and defect:
            if not _safe_component(defect):
                results.append({'name': f.filename, 'status': 'error', 'reason': '缺陷类型非法'})
                continue
            label_part = f'不合格_{defect}'
        if not _safe_component(label_part):
            results.append({'name': f.filename, 'status': 'error', 'reason': '标签非法'})
            continue
        data = f.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            results.append({'name': f.filename, 'status': 'error', 'reason': '无法解析图片'})
            continue
        checks = {}
        if qc_enabled:
            q = dataset_tools.quality_check(img)
            checks['quality'] = q
            if not q.get('ok') and not force_enabled:
                results.append({'name': f.filename, 'status': 'error',
                                'reason': '质量过低（模糊/曝光异常，得分%d）' % q.get('score', 0)})
                continue
        if dup_enabled:
            d = dataset_tools.find_duplicates(img, base)
            checks['dedup'] = d
            if d.get('duplicates') and not force_enabled:
                results.append({'name': f.filename, 'status': 'error',
                                'reason': '疑似重复（最高相似度%d%%）' % int(d.get('max_sim', 0) * 100)})
                continue
        label_dir = base / category / label_part
        label_dir.mkdir(parents=True, exist_ok=True)
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.png', '.bmp'):
            ext = '.jpg'
        fname = '{}_{}_{:03d}_{}{}'.format(label_part, ts_base, i, uuid.uuid4().hex[:4], ext)
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            results.append({'name': f.filename, 'status': 'error', 'reason': '图片编码失败'})
            continue
        buf.tofile(str(label_dir / fname))
        yolo_name = None
        if boxes is not None and i < len(boxes) and boxes[i] and category != 'apple_slice':
            bb = boxes[i]
            if len(bb) == 4:
                try:
                    xc, yc, bw, bh = [float(v) for v in bb]
                except Exception:
                    xc = yc = bw = bh = None
                if xc is not None and 0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1:
                    yimg_dir = YOLO_DETECT_DIR / 'images' / 'train'
                    ylbl_dir = YOLO_DETECT_DIR / 'labels' / 'train'
                    yimg_dir.mkdir(parents=True, exist_ok=True)
                    ylbl_dir.mkdir(parents=True, exist_ok=True)
                    ystem = 'import_{}_{}_{}'.format(category, ts_base, uuid.uuid4().hex[:6])
                    yimg = ystem + '.jpg'
                    ylbl = ystem + '.txt'
                    ok2, buf2 = cv2.imencode('.jpg', img)
                    if ok2:
                        buf2.tofile(str(yimg_dir / yimg))
                        (ylbl_dir / ylbl).write_text(
                            '0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(xc, yc, bw, bh), encoding='utf-8')
                        yolo_name = yimg
        results.append({'name': f.filename, 'status': 'ok', 'saved': fname,
                        'label': label_part, 'checks': checks, 'yolo': yolo_name})
    ok_n = sum(1 for r in results if r['status'] == 'ok')
    yolo_n = sum(1 for r in results if r.get('yolo'))
    logger.info('\u6279\u91cf\u5bfc\u5165: [%s] %d/%d \u6210\u529f, YOLO %d', category, ok_n, len(results), yolo_n)
    return jsonify({'success': True, 'total': len(results), 'ok': ok_n,
                    'yolo_exported': yolo_n, 'results': results})


@app.route('/api/models/history')
def models_history():
    return jsonify(model_registry.history())


@app.route('/api/models/rollback', methods=['POST'])
def models_rollback():
    category = request.form.get('category', '')
    model_type = request.form.get('model_type', '')
    version = request.form.get('version', '')
    ok = model_registry.rollback(category, model_type, version)
    return jsonify({'success': ok})


@app.route('/dataset_collected/<path:filename>')
def dataset_file(filename):
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_collected")
    return send_from_directory(dataset_dir, filename)


@app.route('/dataset')
def dataset_page():
    module, default_category = _module_context()
    return render_template("dataset.html", active_tab="dataset", module=module, default_category=default_category)



@app.route('/significance')
def significance():
    module, default_category = _module_context()
    return render_template('significance.html', active_tab='significance', module=module, default_category=default_category)


def _slice_stats_view(d, module):
    """切片模块只显示 apple_slice 实验数据；无数据返回空，避免混入外观结果。"""
    if module != 'slice':
        return d
    cats = d.get('categories') or []
    if 'apple_slice' not in cats:
        return {"source": "empty", "categories": [], "models": d.get('models', []),
                "accuracy": {}, "n": {}, "mcnemar": {}, "grade_accuracy": {},
                "note": "切片模块暂无实验数据：补充切片样本并运行实验（categories 含 apple_slice）后自动显示。"}
    return {"source": d.get('source', 'experiment'), "categories": ["apple_slice"],
            "models": d.get('models', []),
            "accuracy": {"apple_slice": (d.get('accuracy') or {}).get('apple_slice', {})},
            "n": {"apple_slice": (d.get('n') or {}).get('apple_slice')},
            "mcnemar": {"apple_slice": (d.get('mcnemar') or {}).get('apple_slice')},
            "grade_accuracy": {"apple_slice": (d.get('grade_accuracy') or {}).get('apple_slice')},
            "note": "切片实验数据 · " + d.get('generated_at', '')}

@app.route('/api/stats/models')
def stats_models():
    """四模型 x 三品类 准确率矩阵（优先读实验数据）"""
    module = request.args.get('module', 'surface')
    exp_path = config.OUTPUT_DIR / "experiment_results.json"
    if exp_path.exists():
        try:
            d = json.loads(exp_path.read_text(encoding="utf-8"))
            d["note"] = "实验数据 · " + d.get("generated_at", "")
            return jsonify(_slice_stats_view(d, module))
        except Exception as e:
            logger.warning(f"实验数据读取失败: {e}")
    d = {
        "source": "sample",
        "categories": ["apple", "bread", "beverage"],
        "models": ["svm", "rf", "dl", "yolo"],
        "accuracy": {
            "apple": {"svm": 99.3, "rf": 98.6, "dl": 97.1, "yolo": 99.8},
            "bread": {"svm": 91.5, "rf": 92.0, "dl": 93.2, "yolo": 94.1},
            "beverage": {"svm": 88.3, "rf": 89.0, "dl": 90.4, "yolo": 91.2},
        },
        "n": {},
        "mcnemar": {},
        "grade_accuracy": {},
        "note": "示例数据，请运行 python scripts/run_experiment.py 生成真实实验结果"
    }
    return jsonify(_slice_stats_view(d, module))


_retrain_lock = threading.Lock()
_retrain_thread = None


@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    global _retrain_thread
    data = request.get_json(silent=True) or {}
    category = data.get('category', 'apple')
    if category == 'apple_slice':
        return jsonify({'error': '切片模块使用独立规则算法检测，暂不支持重训（已禁止以防覆盖外观模型）。请在外观模块训练模型。'}), 400
    models = data.get('models', ['svm', 'rf'])
    try:
        epochs = int(data.get('epochs', 15) or 15)
    except (TypeError, ValueError):
        epochs = 15
    if _retrain_thread and _retrain_thread.is_alive():
        return jsonify({'error': '已有任务在运行'}), 409
    allowed = {'svm', 'rf', 'dl', 'yolo'}
    bad = [m for m in models if m not in allowed]
    if bad:
        return jsonify({'error': '不支持的模型: ' + ','.join(bad)}), 400
    status_file = config.INSTANCE_DIR / 'retrain_status.json'
    status = {'running': True, 'log': [], 'results': None, 'error': None}
    status_file.write_text(json.dumps(status, ensure_ascii=False), encoding='utf-8')

    def worker():
        from scripts.retrain import run_retrain
        def log(msg):
            status['log'].append(msg)
            status_file.write_text(json.dumps(status, ensure_ascii=False), encoding='utf-8')
        try:
            res = run_retrain(category, models, epochs=epochs, log=log)
            import apple_defect_detector as _det
            for _attr in ('_scaler', '_svm_model', '_rf_model', '_dl_model', '_yolo_model', '_yolo_detect_model'):
                setattr(_det, _attr, None)
            status['results'] = res
            status['log'].append('重训完成！新模型已生效（无需重启）。')
        except Exception as e:
            status['error'] = str(e)
        status['running'] = False
        status_file.write_text(json.dumps(status, ensure_ascii=False), encoding='utf-8')

    _retrain_thread = threading.Thread(target=worker, daemon=True)
    _retrain_thread.start()
    return jsonify({'success': True, 'message': '重训已开始，请在数据集页面查看进度'})


@app.route('/api/export/train', methods=['POST'])
def api_export_train():
    """仅导出/合并训练集（不训练）：model = dl | yolo"""
    global _retrain_thread
    data = request.get_json(silent=True) or {}
    category = data.get('category', 'apple')
    if category == 'apple_slice':
        return jsonify({'error': '切片模块暂不支持导出训练集（防止污染外观 YOLO 数据集）。'}), 400
    model = data.get('model', 'dl')
    if model not in ('dl', 'yolo'):
        return jsonify({'error': 'model 仅支持 dl / yolo'}), 400
    if _retrain_thread and _retrain_thread.is_alive():
        return jsonify({'error': '已有任务在运行'}), 409
    status_file = config.INSTANCE_DIR / 'retrain_status.json'
    status = {'running': True, 'log': [], 'results': None, 'error': None}
    status_file.write_text(json.dumps(status, ensure_ascii=False), encoding='utf-8')

    def worker():
        def log(msg):
            status['log'].append(msg)
            status_file.write_text(json.dumps(status, ensure_ascii=False), encoding='utf-8')
        try:
            if model == 'yolo':
                from scripts.yolo_export import merge
                r = merge(category, log=log)
                status['results'] = {'export': 'yolo', 'count': r['exported'], 'skipped': r['skipped']}
            else:
                import shutil
                from scripts.export_dataset import export
                cat_export = config.OUTPUT_DIR / 'dataset_export' / category
                if cat_export.exists():
                    shutil.rmtree(str(cat_export))
                n = export(category, 0.8, 42)
                status['results'] = {'export': 'dl', 'count': n}
            status['log'].append('导出完成，可以开始重训。')
        except Exception as e:
            status['error'] = str(e)
        status['running'] = False
        status_file.write_text(json.dumps(status, ensure_ascii=False), encoding='utf-8')

    _retrain_thread = threading.Thread(target=worker, daemon=True)
    _retrain_thread.start()
    return jsonify({'success': True, 'message': '导出任务已开始'})


# ==================== YOLO 手动标注导出 ====================
YOLO_DETECT_DIR = config.ROOT_DIR / 'yolo_detect_dataset'


@app.route('/api/yolo/export_box', methods=['POST'])
def yolo_export_box():
    """把检测页面手动框选的 bbox 导出为 YOLO 检测标注（追加到训练集）

    参数: file(原图) category bbox=[xc, yc, w, h] 归一化坐标
    """
    if 'file' not in request.files:
        return jsonify({'error': '未找到图片'}), 400
    file = request.files['file']
    category = request.form.get('category', 'apple')
    if not _safe_component(category):
        return jsonify({'error': '非法的品类'}), 400
    if category == 'apple_slice':
        return jsonify({'error': '切片模块暂不支持导出 YOLO 检测标注（防止污染外观 YOLO 数据集）'}), 400
    try:
        bbox = json.loads(request.form.get('bbox', 'null') or 'null')
    except Exception:
        bbox = None
    if not bbox or len(bbox) != 4:
        return jsonify({'error': '框参数无效'}), 400
    try:
        xc, yc, w, h = [float(v) for v in bbox]
    except Exception:
        return jsonify({'error': '框参数无效'}), 400
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return jsonify({'error': '框参数超出范围'}), 400

    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': '无法解析图片'}), 400
    ih, iw = img.shape[:2]
    half_w = w * iw / 2
    half_h = h * ih / 2
    cx, cy = xc * iw, yc * ih
    if cx - half_w < -1 or cx + half_w > iw + 1 or cy - half_h < -1 or cy + half_h > ih + 1:
        return jsonify({'error': '框超出图片边界'}), 400

    img_dir = YOLO_DETECT_DIR / 'images' / 'train'
    lbl_dir = YOLO_DETECT_DIR / 'labels' / 'train'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    stem = 'manual_{}_{}_{}'.format(category, ts, uuid.uuid4().hex[:6])
    img_name = stem + '.jpg'
    lbl_name = stem + '.txt'
    ok, buf = cv2.imencode('.jpg', img)
    if not ok:
        return jsonify({'error': '图片编码失败'}), 500
    buf.tofile(str(img_dir / img_name))
    (lbl_dir / lbl_name).write_text(
        '0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(xc, yc, w, h), encoding='utf-8')
    logger.info('YOLO 手动标注导出: %s [%s]', img_name, category)
    return jsonify({'success': True, 'image': img_name, 'label': lbl_name,
                    'files': [img_name, lbl_name]})


@app.route('/api/yolo/export_batch', methods=['POST'])
def yolo_export_batch():
    """批量导出 YOLO 检测标注：自动分割生成 bbox，可选同时存为合格样本

    参数: files(多图) category save_good(1/0)
    """
    category = request.form.get('category', 'apple')
    if not _safe_component(category):
        return jsonify({'error': '\u975e\u6cd5\u54c1\u7c7b'}), 400
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': '\u672a\u9009\u62e9\u56fe\u7247'}), 400
    if len(files) > 60:
        return jsonify({'error': '\u5355\u6279\u6700\u591a 60 \u5f20'}), 400
    save_good = request.form.get('save_good', '') in ('1', 'true', 'on')

    img_dir = YOLO_DETECT_DIR / 'images' / 'train'
    lbl_dir = YOLO_DETECT_DIR / 'labels' / 'train'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    good_dir = config.ROOT_DIR / 'dataset_collected' / category / '\u5408\u683c'
    if save_good:
        good_dir.mkdir(parents=True, exist_ok=True)

    exported = skipped = saved_good = 0
    details = []
    for f in files:
        data = f.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            details.append({'name': f.filename, 'status': 'skip', 'reason': '\u65e0\u6cd5\u89e3\u6790'})
            continue
        bbox = _auto_bbox(img)
        if bbox is None:
            skipped += 1
            details.append({'name': f.filename, 'status': 'skip', 'reason': '\u672a\u68c0\u6d4b\u51fa\u4e3b\u4f53'})
            continue
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        stem = 'auto_{}_{}_{}'.format(category, ts, uuid.uuid4().hex[:6])
        img_name = stem + '.jpg'
        lbl_name = stem + '.txt'
        ok, buf = cv2.imencode('.jpg', img)
        if not ok:
            skipped += 1
            continue
        buf.tofile(str(img_dir / img_name))
        (lbl_dir / lbl_name).write_text(
            '0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(*bbox), encoding='utf-8')
        exported += 1
        if save_good:
            gname = '\u5408\u683c_{}_{}.jpg'.format(ts, uuid.uuid4().hex[:4])
            buf.tofile(str(good_dir / gname))
            saved_good += 1
        details.append({'name': f.filename, 'status': 'ok', 'image': img_name})
    logger.info('YOLO \u6279\u91cf\u5bfc\u51fa: \u5bfc\u51fa %d, \u8df3\u8fc7 %d, \u5408\u683c %d', exported, skipped, saved_good)
    return jsonify({'success': True, 'exported': exported, 'skipped': skipped,
                    'saved_good': saved_good, 'details': details})


@app.route('/api/retrain/status')
def api_retrain_status():
    status_file = config.INSTANCE_DIR / 'retrain_status.json'
    if status_file.exists():
        try:
            return jsonify(json.loads(status_file.read_text(encoding='utf-8')))
        except Exception:
            pass
    return jsonify({'running': False, 'log': [], 'results': None, 'error': None})


# ==================== 感官数据（四觉一体） ====================
@app.route('/sensory')
def sensory_page():
    module, default_category = _module_context()
    return render_template('sensory.html', active_tab='sensory', module=module, default_category=default_category)


@app.route('/api/sensory/list')
def sensory_list():
    from models import SensoryRecord, QDAEntry
    import statistics as _st
    category = request.args.get('category', '')
    q = SensoryRecord.query
    if category:
        q = q.filter(SensoryRecord.category == category)
    recs = q.order_by(SensoryRecord.created_at.desc()).limit(500).all()
    # QDA 聚合：每 sample_id 的评分份数 + 均值 + 标准差（仅统计当前品类）
    sid_set = set(r.sample_id for r in recs) if category else None
    qda_rows = QDAEntry.query.all()
    if sid_set is not None:
        qda_rows = [r for r in qda_rows if r.sample_id in sid_set]
    qda_summary = {}
    qda_keys = ('qda_color', 'qda_odor', 'qda_taste', 'qda_texture', 'qda_overall',
                'qda_sweet', 'qda_sour', 'qda_astringent', 'qda_aroma',
                'qda_crispness', 'qda_juiciness')
    by_sid = {}
    for q in qda_rows:
        by_sid.setdefault(q.sample_id, []).append(q)
    for sid, entries in by_sid.items():
        summary = {'n': len(entries), 'panelists': []}
        for k in qda_keys:
            vals = [getattr(e, k) for e in entries if getattr(e, k) is not None]
            if vals:
                summary[k] = {'mean': round(float(_st.mean(vals)), 2),
                              'std': round(float(_st.stdev(vals)), 2) if len(vals) > 1 else None}
        for e in sorted(entries, key=lambda x: x.panelist):
            summary['panelists'].append({'panelist': e.panelist, 'qda_overall': e.qda_overall})
        qda_summary[sid] = summary
    return jsonify({'ok': True, 'records': [r.to_dict() for r in recs],
                    'qda_summary': qda_summary})


@app.route('/api/sensory/save', methods=['POST'])
def sensory_save():
    """礆化部分存 SensoryRecord（sample_id 唯一）；QDA 多人评分存 QDAEntry（sample_id+panelist 联合唯一）。"""
    from models import SensoryRecord, QDAEntry
    data = request.get_json(silent=True) or {}
    sample_id = (data.get('sample_id') or '').strip()
    if not sample_id:
        return jsonify({'ok': False, 'error': '?? sample_id'}), 400

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # ---- 礆化（SensoryRecord） ----
    updates = {
        'category': data.get('category') or 'apple',
        'visual_filename': (data.get('visual_filename') or '').strip() or None,
        'odor_csv': (data.get('odor_csv') or '').strip() or None,
        'hardness': _f(data.get('hardness')), 'weight_g': _f(data.get('weight_g')),
        'brix': _f(data.get('brix')), 'ph': _f(data.get('ph')),
        'lab_l': _f(data.get('lab_l')), 'lab_a': _f(data.get('lab_a')), 'lab_b': _f(data.get('lab_b')),
        'color_coverage': _f(data.get('color_coverage')), 'fruit_diameter_mm': _f(data.get('fruit_diameter_mm')),
        'shape_index': _f(data.get('shape_index')),
        'ta': _f(data.get('ta')), 'sugar_acid_ratio': _f(data.get('sugar_acid_ratio')),
        'vitamin_c': _f(data.get('vitamin_c')),
        'tpa_springiness': _f(data.get('tpa_springiness')), 'tpa_cohesiveness': _f(data.get('tpa_cohesiveness')),
        'tpa_chewiness': _f(data.get('tpa_chewiness')), 'juice_pct': _f(data.get('juice_pct')),
        'label': data.get('label') or None,
        'remark': data.get('remark') or '',
    }
    rec = SensoryRecord.query.filter_by(sample_id=sample_id).first()
    if rec is None:
        rec = SensoryRecord(sample_id=sample_id, **updates)
        db.session.add(rec)
    else:
        for k, v in updates.items():
            setattr(rec, k, v)

    # ---- QDA（QDAEntry） ----
    qda_keys = ('qda_color', 'qda_odor', 'qda_taste', 'qda_texture', 'qda_overall',
                'qda_sweet', 'qda_sour', 'qda_astringent', 'qda_aroma',
                'qda_crispness', 'qda_juiciness')
    qda_vals = {k: _f(data.get(k)) for k in qda_keys}
    has_qda = any(v is not None for v in qda_vals.values())
    if has_qda:
        panelist = (data.get('panelist') or '').strip() or 'default'
        qrec = QDAEntry.query.filter_by(sample_id=sample_id, panelist=panelist).first()
        if qrec is None:
            qrec = QDAEntry(sample_id=sample_id, panelist=panelist, **qda_vals)
            db.session.add(qrec)
        else:
            for k, v in qda_vals.items():
                if v is not None:
                    setattr(qrec, k, v)
    db.session.commit()
    return jsonify({'ok': True, 'id': rec.id, 'sample_id': rec.sample_id,
                    'panelist': (data.get('panelist') or '').strip() or 'default' if has_qda else None})


@app.route('/api/sensory/qda-detail')
def sensory_qda_detail():
    """某样本的 QDA 多人评分明细。"""
    from models import QDAEntry
    sample_id = (request.args.get('sample_id') or '').strip()
    if not sample_id:
        return jsonify({'ok': False, 'error': '?? sample_id'}), 400
    rows = QDAEntry.query.filter_by(sample_id=sample_id).order_by(QDAEntry.created_at.asc()).all()
    return jsonify({'ok': True, 'sample_id': sample_id,
                    'entries': [r.to_dict() for r in rows]})


@app.route('/api/sensory/delete-qda', methods=['POST'])
def sensory_delete_qda():
    """删除某条 QDA 评分。"""
    from models import QDAEntry
    data = request.get_json(silent=True) or {}
    qrec = db.session.get(QDAEntry, int(data.get('id') or 0))
    if qrec is None:
        return jsonify({'ok': False, 'error': '?????'}), 404
    db.session.delete(qrec)
    db.session.commit()
    return jsonify({'ok': True})


@app.route('/api/sensory/delete', methods=['POST'])
def sensory_delete():
    from models import SensoryRecord
    data = request.get_json(silent=True) or {}
    rec = SensoryRecord.query.get(int(data.get('id') or 0))
    if rec is None:
        return jsonify({'ok': False, 'error': '记录不存在'}), 404
    db.session.delete(rec)
    db.session.commit()
    return jsonify({'ok': True})


@app.route('/api/sensory/export', methods=['POST'])
def sensory_export():
    import io as _io
    buf = _io.StringIO()
    # 直接用内存 CSV 返回（按模块/品类过滤）
    import csv as _csv
    from models import SensoryRecord
    _cat = (request.get_json(silent=True) or {}).get('category') or request.args.get('category', '')
    _q = SensoryRecord.query
    if _cat:
        _q = _q.filter(SensoryRecord.category == _cat)
    recs = _q.order_by(SensoryRecord.created_at.desc()).limit(10000).all()
    cols = ["sample_id", "category", "visual_filename", "odor_csv", "hardness",
            "weight_g", "brix", "ph", "lab_l", "lab_a", "lab_b", "color_coverage",
            "fruit_diameter_mm", "shape_index", "ta", "sugar_acid_ratio", "vitamin_c",
            "tpa_springiness", "tpa_cohesiveness", "tpa_chewiness", "juice_pct",
            "qda_color", "qda_odor", "qda_taste", "qda_texture", "qda_overall",
            "qda_sweet", "qda_sour", "qda_astringent", "qda_aroma", "qda_crispness",
            "qda_juiciness", "label", "remark", "created_at"]
    w = _csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in recs:
        d = r.to_dict()
        w.writerow({c: d.get(c, '') for c in cols})
    resp = app.response_class(buf.getvalue(), mimetype='text/csv')
    resp.headers['Content-Disposition'] = 'attachment; filename=sensory_records.csv'
    return resp


# ==================== 电子鼻（嗅觉） ====================
@app.route('/enose')
def enose_page():
    module, default_category = _module_context()
    return render_template('enose.html', active_tab='enose', module=module, default_category=default_category)


@app.route('/api/enose/predict', methods=['POST'])
def enose_predict():
    """上传单条气味 CSV → 预处理特征 + 曲线 → 预测（若模型存在）。"""
    import sys as _sys
    enose_dir = str(config.ROOT_DIR / 'enose')
    if enose_dir not in _sys.path:
        _sys.path.insert(0, enose_dir)
    try:
        import preprocess as enose_pp
        import predict_freshness as enose_pf
    except Exception as e:
        return jsonify({'error': f'电子鼻模块导入失败: {e}'}), 500

    f = request.files.get('file')
    if f is None or not f.filename:
        return jsonify({'error': '未找到 CSV 文件'}), 400
    tmp_path = str(config.INSTANCE_DIR / ('enose_' + uuid.uuid4().hex + '.csv'))
    f.save(tmp_path)
    import pandas as _pd
    try:
        df = _pd.read_csv(tmp_path, encoding='utf-8')
        enose_pp.savgol_filter_df(df)
    except Exception as e:
        return jsonify({'error': f'CSV 解析失败: {e}'}), 400

    channels = [c for c in enose_pp.CHANNELS if c in df.columns]
    feats = enose_pf.extract_single(tmp_path)
    # 曲线降采样（最多 200 点）
    curves, labels = [], []
    import numpy as _np
    for ch in channels:
        v = _pd.to_numeric(df[ch], errors='coerce').values.astype(float)
        v = v[_np.isfinite(v)]
        n = len(v)
        if n > 200:
            idx = _np.linspace(0, n - 1, 200).astype(int)
            v = v[idx]
        curves.append([round(float(x), 3) for x in v])
        labels.append(ch)
    features = []
    for i in range(0, len(feats), 6):
        features.append([round(float(x), 4) for x in feats[i:i + 6]])

    # 预测（模块独立：切片模块用 *_slice.pkl）
    prediction, proba, model_name = None, None, None
    _emod = request.args.get('module', 'surface')
    _suffix = '_slice' if _emod == 'slice' else ''
    model_path = config.ROOT_DIR / 'enose' / 'models' / f'freshness_rf{_suffix}.pkl'
    if model_path.exists():
        try:
            import joblib
            bundle = joblib.load(str(model_path))
            clf = bundle['model'] if isinstance(bundle, dict) else bundle
            X = feats.reshape(1, -1)
            if isinstance(bundle, dict) and 'feat_cols' in bundle:
                X = _pd.DataFrame([feats], columns=bundle['feat_cols']).fillna(0)
            pred = clf.predict(X)[0]
            proba = [{'k': str(k), 'v': float(v)} for k, v in zip(clf.classes_, clf.predict_proba(X)[0])] if hasattr(clf, 'predict_proba') else None
            prediction = str(pred)
            model_name = model_path.name
        except Exception as e:
            prediction = f'预测失败: {e}'
    # 回归预测：剩余货架期/放置时间（shelf_plsr.pkl 或 shelf_rf.pkl）
    regression = None
    for _rname in (f'shelf_plsr{_suffix}.pkl', f'shelf_rf{_suffix}.pkl'):
        _rpath = config.ROOT_DIR / 'enose' / 'models' / _rname
        if _rpath.exists():
            try:
                import joblib as _joblib
                _bundle = _joblib.load(str(_rpath))
                _reg = _bundle.get('model') if isinstance(_bundle, dict) else _bundle
                _X = feats.reshape(1, -1)
                if isinstance(_bundle, dict) and _bundle.get('feat_cols'):
                    _X = _pd.DataFrame([feats], columns=_bundle['feat_cols']).fillna(0)
                _val = float(_reg.predict(_X)[0])
                _target = _bundle.get('target', 'hours') if isinstance(_bundle, dict) else 'hours'
                _unit = {'hours': '小时', 'minutes': '分钟', 'days': '天'}.get(_target, '小时')
                regression = {'value': round(_val, 2), 'unit': _unit, 'target': _target, 'model': _rpath.name}
                break
            except Exception as e:
                regression = {'value': None, 'error': str(e), 'model': _rpath.name}
                break
    return jsonify({'ok': True, 'curves': curves, 'channels': labels,
                    'features': features, 'prediction': prediction,
                    'proba': proba, 'model': model_name,
                    'regression': regression,
                    'feature_dim': int(len(feats))})


@app.route('/models')
def models_page():
    module, default_category = _module_context()
    return render_template('models.html', active_tab='models', module=module, default_category=default_category)


@app.route('/api/models/status')
def models_status():
    """返回两大模块的模型清单，证明模型完全独立。"""
    def fstat(p):
        fp = config.ROOT_DIR / p
        if fp.exists():
            return {'exists': True, 'size_kb': round(fp.stat().st_size / 1024.0, 1),
                    'updated': datetime.fromtimestamp(fp.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}
        return {'exists': False, 'size_kb': 0, 'updated': ''}
    surface = [
        {'key': 'scaler', 'name': '特征标准化', 'path': 'apple_scaler.pkl', **fstat('apple_scaler.pkl')},
        {'key': 'svm', 'name': 'SVM 支持向量机', 'path': 'apple_svm_model.dat', **fstat('apple_svm_model.dat')},
        {'key': 'rf', 'name': '随机森林', 'path': 'apple_rf_model.pkl', **fstat('apple_rf_model.pkl')},
        {'key': 'dl', 'name': '深度学习 MobileNetV3', 'path': 'apple_dl_model.pth', **fstat('apple_dl_model.pth')},
        {'key': 'yolo_cls', 'name': 'YOLO 分类', 'path': 'yolo_output/yolo_classify-2/weights/best.pt', **fstat('yolo_output/yolo_classify-2/weights/best.pt')},
        {'key': 'yolo_det', 'name': 'YOLO 检测', 'path': 'yolo_detect_best.pt', **fstat('yolo_detect_best.pt')},
    ]
    slice_models = [
        {'key': 'rule', 'name': '切片褐变规则算法', 'path': 'slice_detector.py （内置，无需模型文件）',
         'exists': True, 'size_kb': 0, 'updated': '内置'},
        {'key': 'slice_unet', 'name': '切片 U-Net 分割模型', 'path': 'models/slice_unet.pt', **fstat('models/slice_unet.pt')},
        {'key': 'enose_slice', 'name': '切片电子鼻模型', 'path': 'enose/models/freshness_rf_slice.pkl', **fstat('enose/models/freshness_rf_slice.pkl')},
    ]
    return jsonify({'surface': surface, 'slice': slice_models})


# ==================== 切片连续监测 CSV 实验日志 ====================
SLICE_MONITOR_LOG_CSV = config.ROOT_DIR / 'multimodal_data' / 'slice_monitor_log.csv'
SLICE_MONITOR_LOG_FIELDS = ['timestamp', 'source', 'grade', 'status', 'browning_ratio',
                            'bi_pct', 'bi_lab', 'browning_px', 'flesh_px', 'core_px',
                            'seed_px', 'slice_px', 'ring_inner', 'ring_mid', 'ring_outer']


def _append_slice_monitor_log(res, source='camera'):
    """连续监测实验日志：把一帧检测结果追加为一行 CSV，文件不存在时先写表头。返回累计数据行数。"""
    import csv as _csv
    import datetime as _dt
    csv_path = SLICE_MONITOR_LOG_CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ring = res.get('ring') or {}
    row = {
        'timestamp': _dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': source,
        'grade': res.get('grade'),
        'status': res.get('status'),
        'browning_ratio': res.get('browning_ratio'),
        'bi_pct': res.get('bi_pct'),
        'bi_lab': res.get('bi_lab'),
        'browning_px': res.get('browning_px'),
        'flesh_px': res.get('flesh_px'),
        'core_px': res.get('core_px'),
        'seed_px': res.get('seed_px'),
        'slice_px': res.get('slice_px'),
        'ring_inner': (ring.get('inner') or {}).get('ratio'),
        'ring_mid': (ring.get('mid') or {}).get('ratio'),
        'ring_outer': (ring.get('outer') or {}).get('ratio'),
    }
    need_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open('a', encoding='utf-8-sig', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=SLICE_MONITOR_LOG_FIELDS)
        if need_header:
            w.writeheader()
        w.writerow(row)
    try:
        n = sum(1 for _ in csv_path.open('r', encoding='utf-8-sig')) - 1
    except Exception:
        n = 0
    return max(n, 0)


def _slice_log_info():
    import datetime as _dt
    p = SLICE_MONITOR_LOG_CSV
    if not p.exists() or p.stat().st_size == 0:
        return {'exists': False, 'rows': 0, 'path': str(p), 'updated_at': None}
    try:
        rows = sum(1 for _ in p.open('r', encoding='utf-8-sig')) - 1
    except Exception:
        rows = 0
    return {'exists': True, 'rows': max(rows, 0), 'path': str(p),
            'updated_at': _dt.datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}


@app.route('/api/slice/monitor_log')
def slice_monitor_log():
    """连续监测 CSV 日志状态：是否存在 / 行数 / 更新时间"""
    return jsonify(_slice_log_info())


@app.route('/api/slice/monitor_log/download')
def slice_monitor_log_download():
    """下载连续监测 CSV 日志"""
    p = SLICE_MONITOR_LOG_CSV
    if not p.exists() or p.stat().st_size == 0:
        return jsonify({'error': '还没有日志，请先勾选"连续监测时自动记录到 CSV"再连续监测'}), 404
    from flask import send_file
    return send_file(str(p), mimetype='text/csv', as_attachment=True,
                     download_name='slice_monitor_log.csv')


@app.route('/api/slice/monitor_log/clear', methods=['POST'])
def slice_monitor_log_clear():
    """清空连续监测 CSV 日志（删除文件，下次记录时重建表头）"""
    import time as _time
    p = SLICE_MONITOR_LOG_CSV
    if p.exists():
        _last_err = None
        for _attempt in range(5):
            try:
                p.unlink()
                _last_err = None
                break
            except Exception as _e:
                _last_err = _e
                _time.sleep(0.3)
        if _last_err is not None:
            logger.warning(f"切片监测日志清空失败: {_last_err}")
            return jsonify({'ok': False, 'error': str(_last_err)}), 500
    return jsonify({'ok': True, 'rows': 0})


@app.route('/api/slice/recent')
def slice_recent():
    """返回最近的切片检测记录（监测页展示）"""
    recs = DetectionRecord.query.filter_by(category='apple_slice') \
        .order_by(DetectionRecord.created_at.desc()).limit(6).all()
    return jsonify([{
        'id': r.id, 'grade': r.grade, 'severity': r.severity,
        'defect_ratio': r.defect_ratio,
        'created_at': r.created_at.strftime('%m-%d %H:%M') if r.created_at else '',
        'result_filename': r.result_filename or '',
    } for r in recs])


# ==================== 切片掩膜标注工作台 ====================
@app.route('/slice_annotate')
def slice_annotate_page():
    return render_template('slice_annotate.html', active_tab='slice_annotate',
                           module='slice', default_category='apple_slice')


def _slice_seg_root():
    return config.ROOT_DIR / 'slice_seg_dataset'


def _slice_seg_manifest():
    p = _slice_seg_root() / 'manifest.json'
    if p.exists():
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


def _safe_slice_name(name):
    """校验切片数据集文件名，防路径穿越"""
    import re as _re
    if not name or not _re.fullmatch(r'slice_[0-9a-f]{10}_\d{4}\.jpg', name):
        return None
    return name


@app.route('/slice_seg/<path:filename>')
def slice_seg_static(filename):
    return send_from_directory(str(_slice_seg_root()), filename)


@app.route('/api/slice_annotate/queue')
def slice_annotate_queue():
    root = _slice_seg_root()
    images_dir = root / 'images'
    masks_dir = root / 'masks'
    manifest = _slice_seg_manifest()
    items = []
    for name in sorted(manifest.keys()):
        img_path = images_dir / name
        mask_path = masks_dir / name.replace('.jpg', '.png')
        if not img_path.is_file() or not mask_path.is_file():
            continue
        corrected = mask_path.stat().st_mtime > img_path.stat().st_mtime + 60
        items.append({'name': name, 'src': manifest[name].get('src', ''),
                      'img_url': '/slice_seg/images/' + name,
                      'mask_url': '/api/slice_annotate/mask?name=' + name,
                      'corrected': corrected})
    return jsonify({'items': items, 'count': len(items),
                    'unmasked': sum(1 for n in manifest if not (masks_dir / n.replace('.jpg', '.png')).is_file())})


@app.route('/api/slice_annotate/ingest', methods=['POST'])
def slice_annotate_ingest():
    """把 dataset_collected/apple_slice 里新增的切片照片生成伪标签并入训练集"""
    import sys as _sys
    root = str(config.ROOT_DIR)
    if root not in _sys.path:
        _sys.path.insert(0, root)
    from scripts.slice_seg_prepare import build_dataset
    try:
        info = build_dataset(config.ROOT_DIR / 'dataset_collected' / 'apple_slice',
                             _slice_seg_root())
        return jsonify({'ok': True, **{k: v for k, v in info.items() if k != 'errors'}})
    except Exception as e:
        logger.error(f"切片伪标签生成失败: {e}")
        return jsonify({'ok': False, 'error': str(e)}), 500


def _slice_rule_seg(img):
    import sys as _sys
    root = str(config.ROOT_DIR)
    if root not in _sys.path:
        _sys.path.insert(0, root)
    import slice_detector as _sd
    return _sd.rule_segmentation(img)


@app.route('/api/slice_annotate/mask')
def slice_annotate_mask():
    import base64 as _b64
    name = _safe_slice_name(request.args.get('name', ''))
    if not name:
        return jsonify({'error': '非法的文件名'}), 400
    mask_path = _slice_seg_root() / 'masks' / name.replace('.jpg', '.png')
    if not mask_path.is_file():
        # 没有掩膜时用规则算法即时生成，便于直接进入修正
        img_path = _slice_seg_root() / 'images' / name
        if not img_path.is_file():
            return jsonify({'error': '图片不存在'}), 404
        img = cv2.imread(str(img_path))
        if img is None:
            return jsonify({'error': '图片读取失败'}), 500
        seg = _slice_rule_seg(img)
        ok, buf = cv2.imencode('.png', seg)
        if not ok:
            return jsonify({'error': '掩膜编码失败'}), 500
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.write_bytes(buf.tobytes())
    data = mask_path.read_bytes()
    return jsonify({'name': name, 'mask_data_url': 'data:image/png;base64,' + _b64.b64encode(data).decode('ascii')})


@app.route('/api/slice_annotate/save', methods=['POST'])
def slice_annotate_save():
    import base64 as _b64
    data = request.get_json(silent=True) or {}
    name = _safe_slice_name(data.get('name', ''))
    data_url = str(data.get('mask_data_url', ''))
    if not name or not data_url.startswith('data:image/png;base64,'):
        return jsonify({'error': '参数不完整'}), 400
    try:
        raw = _b64.b64decode(data_url.split(',', 1)[1])
        mask = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return jsonify({'error': '掩膜解析失败'}), 400
        mask = np.clip(mask, 0, 4).astype(np.uint8)
        mask_path = _slice_seg_root() / 'masks' / name.replace('.jpg', '.png')
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), mask)
        logger.info("切片掩膜已保存: %s", name)
        # 同步重算训练/验证划分与类别权重（同一随机种子，划分稳定）
        try:
            import sys as _sys
            root = str(config.ROOT_DIR)
            if root not in _sys.path:
                _sys.path.insert(0, root)
            from scripts.slice_seg_prepare import build_dataset
            build_dataset(config.ROOT_DIR / 'dataset_collected' / 'apple_slice', _slice_seg_root())
        except Exception:
            pass
        return jsonify({'ok': True, 'name': name})
    except Exception as e:
        logger.error(f"切片掩膜保存失败: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/slice_annotate/reset', methods=['POST'])
def slice_annotate_reset():
    """用规则算法重新生成某张图的掩膜（覆盖人工修改）"""
    data = request.get_json(silent=True) or {}
    name = _safe_slice_name(data.get('name', ''))
    if not name:
        return jsonify({'error': '非法的文件名'}), 400
    img_path = _slice_seg_root() / 'images' / name
    if not img_path.is_file():
        return jsonify({'error': '图片不存在'}), 404
    img = cv2.imread(str(img_path))
    if img is None:
        return jsonify({'error': '图片读取失败'}), 500
    seg = _slice_rule_seg(img)
    mask_path = _slice_seg_root() / 'masks' / name.replace('.jpg', '.png')
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), seg)
    return jsonify({'ok': True, 'name': name})


@app.route('/slice')
def slice_page():
    return render_template('slice.html', active_tab='slice',
                           module='slice', default_category='apple_slice')


@app.route('/api/slice/detect', methods=['POST'])
def slice_detect():
    """上传切片照片 → 褐变面积占比 / BI / 等级 + 标注图。"""
    if 'file' not in request.files:
        return jsonify({'error': '未找到图片文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式，请上传 JPG/PNG/BMP 图片'}), 400
    try:
        import sys as _sys
        root = str(config.ROOT_DIR)
        if root not in _sys.path:
            _sys.path.insert(0, root)
        import slice_detector as sd
        data = file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': '图片解析失败，请换一张试试'}), 400
        _engine = request.form.get('engine', 'auto')
        if _engine not in ('rule', 'unet', 'auto'):
            _engine = 'auto'
        res = sd.analyze_slice(img, engine=_engine)
        annotated = res.pop('annotated')
        fname = 'slice_' + uuid.uuid4().hex + '.jpg'
        config.UPLOAD_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(config.UPLOAD_DIR / fname), annotated)
        res['annotated_url'] = '/static/uploads/' + fname
        # 自动入库（category=apple_slice），供历史/统计模块使用；
        # save=0 表示不保存（用于摄像头“连续监测”自动抓帧，避免刷屏历史）
        if request.form.get('save', '1') != '0':
            try:
                from models import DetectionRecord as _DR
                src_name = 'slice_' + uuid.uuid4().hex + '_src.jpg'
                cv2.imwrite(str(config.UPLOAD_DIR / src_name), img)
                _rec = _DR(category='apple_slice', filename=src_name, result_filename=fname,
                           grade=res['status'], grade_color='(0,200,0)',
                           defect_ratio=float(res['browning_ratio']) / 100.0,
                           defect_count=0, apple_area=int(res['slice_px']),
                           defect_area=int(res['browning_px']),
                           severity='等级' + str(res['grade']), needs_review=False)
                db.session.add(_rec)
                db.session.commit()
                res['record_id'] = _rec.id
            except Exception as _e:
                logger.warning(f"切片检测记录保存失败: {_e}")
        # 实验数据：连续监测勾选"记录 CSV"时，把每一帧结果追加到日志
        if request.form.get('log_csv') == '1':
            try:
                res['log_rows'] = _append_slice_monitor_log(res, request.form.get('src', 'camera'))
            except Exception as _e:
                logger.warning(f"切片监测日志写入失败: {_e}")
        return jsonify({'ok': True, **res})
    except Exception as e:
        logger.error(f"切片检测失败: {e}")
        return jsonify({'error': f'切片检测失败: {str(e)}'}), 500


# ==================== 缺陷块类型标注（8.1 数据积累） ====================
@app.route('/api/defect_block/save', methods=['POST'])
def defect_block_save():
    """保存缺陷块类型标注到 multimodal_data/defect_blocks.csv（供 train_defect_type.py 使用）。"""
    import csv as _csv
    data = request.get_json(silent=True) or {}
    filename = str(data.get('filename') or '').strip()
    block_index = int(data.get('block_index') or 0)
    dtype = str(data.get('type') or '').strip()
    if not filename or not dtype:
        return jsonify({'ok': False, 'error': '缺少 filename 或 type'}), 400
    csv_path = config.ROOT_DIR / 'multimodal_data' / 'defect_blocks.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = {
        'filename': filename, 'block_index': block_index, 'type': dtype,
        'area': data.get('area', ''), 'circularity': data.get('circularity', ''),
        'aspect': data.get('aspect', ''), 'hsv': json.dumps(data.get('hsv', []), ensure_ascii=False),
        'glcm': json.dumps(data.get('glcm', []), ensure_ascii=False),
        'grade': data.get('grade', ''), 'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    fields = list(new_row.keys())
    rows = []
    if csv_path.exists():
        with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
            rd = _csv.DictReader(f)
            rows = [r for r in rd
                    if not (r.get('filename') == filename
                            and str(r.get('block_index')) == str(block_index))]
    rows.append(new_row)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return jsonify({'ok': True, 'count': len(rows)})



# ==================== 多模态融合（四觉一体） ====================
@app.route('/api/sensory/train', methods=['POST'])
def sensory_train():
    """读取 multimodal_data 下四觉数据，训练单模态 + 融合模型。"""
    from services.fusion_service import FusionService
    try:
        report = FusionService().train()
        return jsonify({'ok': True, 'report': report})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/sensory/fusion-predict', methods=['POST'])
def sensory_fusion_predict():
    """单个样本融合预测：各单模态 + 融合结果。"""
    from services.fusion_service import FusionService
    data = request.get_json(silent=True) or {}
    sid = (data.get('sample_id') or '').strip()
    if not sid:
        return jsonify({'ok': False, 'error': '缺少 sample_id'}), 400
    try:
        out = FusionService().predict(sid, phys_values=data.get('phys_values'))
        out['ok'] = True
        return jsonify(out)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT
    )
    # 日志文件轮转（单文件最大5MB，保留近3份历史）
    _log_dir = config.ROOT_DIR / "logs"
    _log_dir.mkdir(exist_ok=True)
    _fh = logging.handlers.RotatingFileHandler(
        _log_dir / "app.log", maxBytes=5 * 1024 * 1024,
        backupCount=3, encoding="utf-8")
    _fh.setFormatter(logging.Formatter(config.LOG_FORMAT, config.LOG_DATE_FORMAT))
    logging.getLogger().addHandler(_fh)
    with app.app_context():
        db.create_all()
        _migrate_db()
    warmup_models()
    # 显示本机网络地址，方便手机访问
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        logger.info(f"✔ 手机访问地址: http://{local_ip}:{config.PORT}/")
        logger.info(f"✔ 本机访问地址: http://127.0.0.1:{config.PORT}/")
    except:
        logger.info(f"✔ 访问地址: http://[your-ip]:{config.PORT}/")
    # 生产模式优先使用 waitress（并发稳定）；未安装则回退 Flask 开发服务器
    try:
        from waitress import serve as _waitress_serve
        _waitress_serve(app, host=config.HOST, port=config.PORT, threads=8)
    except ImportError:
        app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)
