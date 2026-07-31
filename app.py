import json
import os
import sys
import uuid
import logging
import threading
import logging.handlers
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
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

from models import db, DetectionRecord
from services import dataset_tools

logger = logging.getLogger("app")

sys.path.insert(0, str(config.ROOT_DIR))
os.environ['YOLO_CONFIG_DIR'] = str(config.ROOT_DIR / '.ultralytics')

app = Flask(__name__)

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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

db.init_app(app)

from services.detection_service import DetectionService
from services.dataset_service import DatasetService
from services.model_registry import ModelRegistry
detection_service = DetectionService()
dataset_service = DatasetService()
model_registry = ModelRegistry()

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html', active_tab='monitor')


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
    tuning = {
        'sensitivity': request.form.get('sensitivity', 5),
        'min_area': request.form.get('min_area', 200),
        'premium_th': request.form.get('premium_th', 0.05),
        'standard_th': request.form.get('standard_th', 0.15),
    }

    try:
        result = detection_service.process_upload(file, category, tuning)
        if 'error' in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        logger.error(f"检测失败: {e}")
        return jsonify({'error': f'检测失败: {str(e)}'}), 500


@app.route('/history')
def history():
    page = request.args.get('page', 1, type=int)
    per_page = 12
    grade_filter = request.args.get('grade', '')
    category = request.args.get('category', '')

    query = DetectionRecord.query.order_by(DetectionRecord.created_at.desc())
    if category:
        query = query.filter(DetectionRecord.category == category)
    if grade_filter:
        query = query.filter(DetectionRecord.grade.contains(grade_filter))

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    records = pagination.items

    return render_template('history.html', active_tab='history', records=records, pagination=pagination, grade_filter=grade_filter, category=category)
    page = request.args.get('page', 1, type=int)
    per_page = 12
    grade_filter = request.args.get('grade', '')

    query = DetectionRecord.query.order_by(DetectionRecord.created_at.desc())
    if grade_filter:
        query = query.filter(DetectionRecord.grade.contains(grade_filter))

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    records = pagination.items

    return render_template('history.html',
                           records=records,
                           pagination=pagination,
                           grade_filter=grade_filter)


@app.route('/detail/<int:record_id>')
def detail(record_id):
    record = DetectionRecord.query.get_or_404(record_id)
    return render_template('detail.html', record=record)


@app.route('/delete/<int:record_id>', methods=['POST'])
def delete(record_id):
    record = DetectionRecord.query.get_or_404(record_id)
    try:
        # 删除关联图片文件
        for fname in [record.filename, record.result_filename]:
            if fname:
                fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
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


def _delete_record_files(record):
    for fname in [record.filename, record.result_filename]:
        if fname:
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


@app.route('/stats')
def stats_page():
    return render_template('stats.html', active_tab='stats')


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
    today = datetime.utcnow().date()
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

def _unlabeled_files():
    """聚合所有未标注图片（平铺 unlabeled/ 与 各品类下的 unlabeled/）"""
    base = config.ROOT_DIR / "dataset_collected"
    candidates = [base / "unlabeled"]
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
    return render_template('annotate.html', active_tab='annotate')


@app.route('/api/annotate/queue')
def annotate_queue():
    files = _unlabeled_files()
    items = [{"path": str(f.relative_to(config.ROOT_DIR / "dataset_collected")).replace("\\", "/"),
              "name": f.name} for f in files]
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
    base = config.ROOT_DIR / "dataset_collected"
    src = base / src_rel
    if not src.is_file():
        return jsonify({"error": "文件不存在"}), 404
    dst_dir = base / category / label
    dst_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    dst = dst_dir / f"{label}_{ts}{src.suffix.lower()}"
    shutil.move(str(src), str(dst))
    logger.info("标注: %s -> %s/%s/%s", src_rel, category, label, dst.name)
    return jsonify({"success": True, "filename": dst.name, "remaining": len(_unlabeled_files())})


@app.route('/api/annotate/skip', methods=['POST'])
def annotate_skip():
    import shutil
    data = request.get_json(silent=True) or {}
    src_rel = data.get("src", "")
    if not src_rel:
        return jsonify({"error": "参数不完整"}), 400
    base = config.ROOT_DIR / "dataset_collected"
    src = base / src_rel
    if src.is_file():
        skip_dir = base / "_skipped"
        skip_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(skip_dir / src.name))
    return jsonify({"success": True, "remaining": len(_unlabeled_files())})


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
    
    return render_template('annotations.html', images=images, count=len(images))



@app.route('/api/export/report')
def export_report():
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from io import BytesIO
    from flask import send_file

    records = DetectionRecord.query.order_by(DetectionRecord.created_at.desc()).all()

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

    headers = ["ID", "文件名", "等级", "缺陷占比", "缺陷数量",
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
            f"{r.defect_ratio * 100:.2f}%",
            r.defect_count, r.apple_area, r.defect_area,
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
    widths = [8, 30, 15, 12, 10, 14, 14, 20]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + i)].width = w

    # Summary sheet
    ws2 = wb.create_sheet(title="统计摘要")
    total = len(records)
    premium = sum(1 for r in records if "Premium" in str(r.grade))
    standard = sum(1 for r in records if "Standard" in str(r.grade))
    reject = sum(1 for r in records if "Reject" in str(r.grade))

    summary = [
        ["指标", "数值"],
        ["累计检测", total],
        ["优质", premium],
        ["标准", standard],
        ["不合格", reject],
        ["优质率", f"{premium / total * 100:.1f}%" if total else "0%"],
        ["不合格率", f"{reject / total * 100:.1f}%" if total else "0%"],
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

    filename = f"苹果品控报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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

    logger.info("\u6570\u636e\u96c6\u6536\u96c6: [%s/%s] %s", category, label, saved)
    return jsonify({'success': True, 'category': category, 'label': label,
                    'files': saved, 'report': report})

@app.route('/api/dataset/list')
def list_dataset():
    """数据集列表: 品类 → 标签 → 图片（可按 category 过滤）"""
    category = request.args.get('category', '')
    data = dataset_service.list()
    if category and category in data.get("categories", {}):
        data = {"categories": {category: data["categories"][category]},
                "total": sum(len(v) for v in data["categories"][category].values())}
    return jsonify(data)


@app.route('/api/dataset/delete', methods=['POST'])
def delete_dataset_item():
    category = request.form.get('category', '')
    label = request.form.get('label', '')
    filename = request.form.get('filename', '')
    if not (category and label and filename):
        return jsonify({'error': '参数不完整'}), 400
    ok = dataset_service.delete(category, label, filename)
    return jsonify({'success': ok})



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
    return render_template("dataset.html", active_tab="dataset")



@app.route('/significance')
def significance():
    return render_template('significance.html', active_tab='significance')


@app.route('/api/stats/models')
def stats_models():
    """四模型 x 三品类 准确率矩阵（优先读实验数据）"""
    exp_path = config.OUTPUT_DIR / "experiment_results.json"
    if exp_path.exists():
        try:
            d = json.loads(exp_path.read_text(encoding="utf-8"))
            d["note"] = "实验数据 · " + d.get("generated_at", "")
            return jsonify(d)
        except Exception as e:
            logger.warning(f"实验数据读取失败: {e}")
    return jsonify({
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
    })


_retrain_lock = threading.Lock()
_retrain_thread = None


@app.route('/api/retrain', methods=['POST'])
def api_retrain():
    global _retrain_thread
    data = request.get_json(silent=True) or {}
    category = data.get('category', 'apple')
    models = data.get('models', ['svm', 'rf'])
    if _retrain_thread and _retrain_thread.is_alive():
        return jsonify({'error': '已有训练任务在运行'}), 409
    status_file = config.INSTANCE_DIR / 'retrain_status.json'
    status = {'running': True, 'log': [], 'results': None, 'error': None}
    status_file.write_text(json.dumps(status, ensure_ascii=False), encoding='utf-8')

    def worker():
        from scripts.retrain import run_retrain
        def log(msg):
            status['log'].append(msg)
            status_file.write_text(json.dumps(status, ensure_ascii=False), encoding='utf-8')
        try:
            res = run_retrain(category, models, log=log)
            status['results'] = res
            status['log'].append('重训完成！请重启服务使新模型生效。')
        except Exception as e:
            status['error'] = str(e)
        status['running'] = False
        status_file.write_text(json.dumps(status, ensure_ascii=False), encoding='utf-8')

    _retrain_thread = threading.Thread(target=worker, daemon=True)
    _retrain_thread.start()
    return jsonify({'success': True, 'message': '重训已开始，请在数据集页面查看进度'})


@app.route('/api/retrain/status')
def api_retrain_status():
    status_file = config.INSTANCE_DIR / 'retrain_status.json'
    if status_file.exists():
        try:
            return jsonify(json.loads(status_file.read_text(encoding='utf-8')))
        except Exception:
            pass
    return jsonify({'running': False, 'log': [], 'results': None, 'error': None})


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
    app.run(debug=config.DEBUG, host=config.HOST, port=config.PORT)