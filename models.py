from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class DetectionRecord(db.Model):
    __tablename__ = 'detection_records'

    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(20), nullable=False, default='apple', index=True)
    filename = db.Column(db.String(255), nullable=False)
    result_filename = db.Column(db.String(255), nullable=True)
    grade = db.Column(db.String(50), nullable=False)
    grade_color = db.Column(db.String(20), nullable=False, default='(0,200,0)')
    defect_ratio = db.Column(db.Float, nullable=False, default=0.0)
    defect_count = db.Column(db.Integer, nullable=False, default=0)
    apple_area = db.Column(db.Integer, nullable=False, default=0)
    defect_area = db.Column(db.Integer, nullable=False, default=0)
    severity = db.Column(db.String(10), nullable=False, default='轻度')
    needs_review = db.Column(db.Boolean, nullable=False, default=False)
    review_reason = db.Column(db.String(255), nullable=False, default='')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)

    def to_dict(self):
        return {
            'id': self.id,
            'category': self.category,
            'filename': self.filename,
            'result_filename': self.result_filename,
            'grade': self.grade,
            'defect_ratio': self.defect_ratio,
            'defect_count': self.defect_count,
            'apple_area': self.apple_area,
            'defect_area': self.defect_area,
            'severity': self.severity,
            'needs_review': self.needs_review,
            'review_reason': self.review_reason,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        }


class SensoryRecord(db.Model):
    """四觉一体感官数据记录（远期架构，见 docs/sensory-system-plan.md）。
    同一 sample_id 关联视觉图/气味曲线/触觉理化/人工感官评分。"""
    __tablename__ = 'sensory_records'

    id = db.Column(db.Integer, primary_key=True)
    sample_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    category = db.Column(db.String(20), nullable=False, default='apple', index=True)
    # 视觉
    visual_filename = db.Column(db.String(255), nullable=True)
    visual_features = db.Column(db.Text, nullable=True)   # JSON，视觉特征向量
    # 嗅觉
    odor_csv = db.Column(db.String(255), nullable=True)   # enose/data/raw 下的气味 CSV
    odor_features = db.Column(db.Text, nullable=True)     # JSON，气味特征向量
    # 触觉/理化（基础）
    hardness = db.Column(db.Float, nullable=True)         # 硬度计 kg/cm2
    weight_g = db.Column(db.Float, nullable=True)
    brix = db.Column(db.Float, nullable=True)             # 糖度
    ph = db.Column(db.Float, nullable=True)
    # 视觉量化指标（色差仪/图像提取）
    lab_l = db.Column(db.Float, nullable=True)            # CIELAB 亮度 L*
    lab_a = db.Column(db.Float, nullable=True)            # CIELAB 红绿 a*
    lab_b = db.Column(db.Float, nullable=True)            # CIELAB 黄蓝 b*
    color_coverage = db.Column(db.Float, nullable=True)   # 红色着色率 %
    fruit_diameter_mm = db.Column(db.Float, nullable=True)  # 果径/横径 mm
    shape_index = db.Column(db.Float, nullable=True)      # 果形指数（纵横径比）
    # 味觉/理化
    ta = db.Column(db.Float, nullable=True)               # 可滴定酸度 %（NaOH 滴定）
    sugar_acid_ratio = db.Column(db.Float, nullable=True) # 糖酸比 = Brix / TA
    vitamin_c = db.Column(db.Float, nullable=True)        # 维生素C mg/100g（可选）
    # 触觉/质地（TPA 质构仪或手持）
    tpa_springiness = db.Column(db.Float, nullable=True)  # 弹性
    tpa_cohesiveness = db.Column(db.Float, nullable=True) # 内聚性
    tpa_chewiness = db.Column(db.Float, nullable=True)    # 咀嚼性
    juice_pct = db.Column(db.Float, nullable=True)        # 出汁率 %（可选）
    # 人工感官 QDA（1~9 分）基础五项 + 扩展六项
    qda_color = db.Column(db.Float, nullable=True)        # 外观颜色
    qda_odor = db.Column(db.Float, nullable=True)         # 气味总体
    qda_taste = db.Column(db.Float, nullable=True)        # 滋味总体
    qda_texture = db.Column(db.Float, nullable=True)      # 质地总体
    qda_overall = db.Column(db.Float, nullable=True)      # 总体接受度
    qda_sweet = db.Column(db.Float, nullable=True)        # 甜味强度
    qda_sour = db.Column(db.Float, nullable=True)         # 酸味强度
    qda_astringent = db.Column(db.Float, nullable=True)   # 涩味强度
    qda_aroma = db.Column(db.Float, nullable=True)        # 果香强度
    qda_crispness = db.Column(db.Float, nullable=True)    # 脆度
    qda_juiciness = db.Column(db.Float, nullable=True)    # 多汁性
    # 标注
    label = db.Column(db.String(20), nullable=True)       # 合格/不合格/等级
    remark = db.Column(db.String(255), nullable=True, default='')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)

    def to_dict(self):
        import json as _json
        def _load(x):
            try:
                return _json.loads(x) if x else None
            except Exception:
                return x
        return {
            'id': self.id, 'sample_id': self.sample_id, 'category': self.category,
            'visual_filename': self.visual_filename, 'visual_features': _load(self.visual_features),
            'odor_csv': self.odor_csv, 'odor_features': _load(self.odor_features),
            'hardness': self.hardness, 'weight_g': self.weight_g,
            'brix': self.brix, 'ph': self.ph,
            'lab_l': self.lab_l, 'lab_a': self.lab_a, 'lab_b': self.lab_b,
            'color_coverage': self.color_coverage, 'fruit_diameter_mm': self.fruit_diameter_mm,
            'shape_index': self.shape_index,
            'ta': self.ta, 'sugar_acid_ratio': self.sugar_acid_ratio, 'vitamin_c': self.vitamin_c,
            'tpa_springiness': self.tpa_springiness, 'tpa_cohesiveness': self.tpa_cohesiveness,
            'tpa_chewiness': self.tpa_chewiness, 'juice_pct': self.juice_pct,
            'qda_color': self.qda_color, 'qda_odor': self.qda_odor,
            'qda_taste': self.qda_taste, 'qda_texture': self.qda_texture,
            'qda_overall': self.qda_overall,
            'qda_sweet': self.qda_sweet, 'qda_sour': self.qda_sour,
            'qda_astringent': self.qda_astringent, 'qda_aroma': self.qda_aroma,
            'qda_crispness': self.qda_crispness, 'qda_juiciness': self.qda_juiciness,
            'label': self.label, 'remark': self.remark,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        }


class QDAEntry(db.Model):
    """人工感官评分单条记录（QDA 多人评鉴，同一 sample_id 多条，取均值）
    sample_id + panelist 联合唯一。"""
    __tablename__ = 'qda_entries'

    id = db.Column(db.Integer, primary_key=True)
    sample_id = db.Column(db.String(50), nullable=False, index=True)
    panelist = db.Column(db.String(50), nullable=False, default='default')
    qda_color = db.Column(db.Float, nullable=True)
    qda_odor = db.Column(db.Float, nullable=True)
    qda_taste = db.Column(db.Float, nullable=True)
    qda_texture = db.Column(db.Float, nullable=True)
    qda_overall = db.Column(db.Float, nullable=True)
    qda_sweet = db.Column(db.Float, nullable=True)
    qda_sour = db.Column(db.Float, nullable=True)
    qda_astringent = db.Column(db.Float, nullable=True)
    qda_aroma = db.Column(db.Float, nullable=True)
    qda_crispness = db.Column(db.Float, nullable=True)
    qda_juiciness = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)

    __table_args__ = (db.UniqueConstraint('sample_id', 'panelist', name='uq_qda_sample_panelist'),)

    def to_dict(self):
        return {
            'id': self.id, 'sample_id': self.sample_id, 'panelist': self.panelist,
            'qda_color': self.qda_color, 'qda_odor': self.qda_odor,
            'qda_taste': self.qda_taste, 'qda_texture': self.qda_texture,
            'qda_overall': self.qda_overall,
            'qda_sweet': self.qda_sweet, 'qda_sour': self.qda_sour,
            'qda_astringent': self.qda_astringent, 'qda_aroma': self.qda_aroma,
            'qda_crispness': self.qda_crispness, 'qda_juiciness': self.qda_juiciness,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        }


class QualityRecord(db.Model):
    __tablename__ = 'quality_records'

    id = db.Column(db.Integer, primary_key=True)
    record_no = db.Column(db.String(50), unique=True, nullable=False, index=True)
    category = db.Column(db.String(20), nullable=False, default='apple', index=True)
    period = db.Column(db.String(20), nullable=False, default='', index=True)
    total = db.Column(db.Integer, nullable=False, default=0)
    premium = db.Column(db.Integer, nullable=False, default=0)
    standard = db.Column(db.Integer, nullable=False, default=0)
    reject = db.Column(db.Integer, nullable=False, default=0)
    review_count = db.Column(db.Integer, nullable=False, default=0)
    pass_rate = db.Column(db.Float, nullable=False, default=0.0)
    ccp_status = db.Column(db.Text, nullable=False, default='[]')
    conclusion = db.Column(db.String(255), nullable=False, default='')
    operator = db.Column(db.String(100), nullable=False, default='')
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)

    def to_dict(self):
        import json
        try:
            ccp = json.loads(self.ccp_status or '[]')
        except Exception:
            ccp = []
        return {
            'id': self.id,
            'record_no': self.record_no,
            'category': self.category,
            'period': self.period,
            'total': self.total,
            'premium': self.premium,
            'standard': self.standard,
            'reject': self.reject,
            'review_count': self.review_count,
            'pass_rate': self.pass_rate,
            'ccp_status': ccp,
            'conclusion': self.conclusion,
            'operator': self.operator,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        }
