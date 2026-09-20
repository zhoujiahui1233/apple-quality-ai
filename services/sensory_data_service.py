# -*- coding: utf-8 -*-
"""
sensory_data_service.py - 四觉感官数据记录服务（远期架构骨架）

提供 SensoryRecord 的增查改接口，供后续 app.py 路由与前端感官数据面板调用。
骨架阶段仅实现基础 CRUD 与 CSV 导入；TODO 见下方标记。

用法（在 app 中接线）：
  from services.sensory_data_service import SensoryDataService
  svc = SensoryDataService()
  svc.create(sample_id="20260806_apple_001", category="apple", ...)
"""
import json
from datetime import datetime

from models import db, SensoryRecord


class SensoryDataService:
    def create(self, sample_id, category="apple", visual_filename=None,
               visual_features=None, odor_csv=None, odor_features=None,
               hardness=None, weight_g=None, brix=None, ph=None,
               lab_l=None, lab_a=None, lab_b=None, color_coverage=None,
               fruit_diameter_mm=None, shape_index=None,
               ta=None, sugar_acid_ratio=None, vitamin_c=None,
               tpa_springiness=None, tpa_cohesiveness=None, tpa_chewiness=None, juice_pct=None,
               qda_color=None, qda_odor=None, qda_taste=None, qda_texture=None,
               qda_overall=None, qda_sweet=None, qda_sour=None, qda_astringent=None,
               qda_aroma=None, qda_crispness=None, qda_juiciness=None,
               label=None, remark=""):
        """新增一条感官记录（同一样本四觉数据关联）。"""
        rec = SensoryRecord(
            sample_id=sample_id, category=category,
            visual_filename=visual_filename,
            visual_features=json.dumps(visual_features, ensure_ascii=False) if visual_features else None,
            odor_csv=odor_csv,
            odor_features=json.dumps(odor_features, ensure_ascii=False) if odor_features else None,
            hardness=hardness, weight_g=weight_g, brix=brix, ph=ph,
            lab_l=lab_l, lab_a=lab_a, lab_b=lab_b, color_coverage=color_coverage,
            fruit_diameter_mm=fruit_diameter_mm, shape_index=shape_index,
            ta=ta, sugar_acid_ratio=sugar_acid_ratio, vitamin_c=vitamin_c,
            tpa_springiness=tpa_springiness, tpa_cohesiveness=tpa_cohesiveness,
            tpa_chewiness=tpa_chewiness, juice_pct=juice_pct,
            qda_color=qda_color, qda_odor=qda_odor,
            qda_taste=qda_taste, qda_texture=qda_texture, qda_overall=qda_overall,
            qda_sweet=qda_sweet, qda_sour=qda_sour, qda_astringent=qda_astringent,
            qda_aroma=qda_aroma, qda_crispness=qda_crispness, qda_juiciness=qda_juiciness,
            label=label, remark=remark,
        )
        db.session.add(rec)
        db.session.commit()
        return rec
    def get(self, sample_id):
        return SensoryRecord.query.filter_by(sample_id=sample_id).first()

    def list(self, category=None, limit=100):
        q = SensoryRecord.query
        if category:
            q = q.filter_by(category=category)
        return q.order_by(SensoryRecord.created_at.desc()).limit(limit).all()

    def update_phys(self, sample_id, hardness=None, weight_g=None, brix=None, ph=None):
        """补录触觉/理化数据（同一批样本后续补测）。"""
        rec = self.get(sample_id)
        if rec is None:
            return None
        if hardness is not None: rec.hardness = hardness
        if weight_g is not None: rec.weight_g = weight_g
        if brix is not None: rec.brix = brix
        if ph is not None: rec.ph = ph
        db.session.commit()
        return rec

    def update_qda(self, sample_id, qda_color=None, qda_odor=None, qda_taste=None,
                   qda_texture=None, qda_overall=None):
        """补录人工 QDA 感官评分。"""
        rec = self.get(sample_id)
        if rec is None:
            return None
        for k, v in [("qda_color", qda_color), ("qda_odor", qda_odor),
                     ("qda_taste", qda_taste), ("qda_texture", qda_texture),
                     ("qda_overall", qda_overall)]:
            if v is not None:
                setattr(rec, k, v)
        db.session.commit()
        return rec

    def export_csv(self, path="output/sensory_records.csv"):
        """导出四觉记录为 CSV（供 multimodal_fusion / sensory_correlation 使用）。"""
        import csv as _csv
        import os
        recs = self.list(limit=10000)
        if not recs:
            return 0
        cols = ["sample_id", "category", "visual_filename", "odor_csv",
                "hardness", "weight_g", "brix", "ph",
                "lab_l", "lab_a", "lab_b", "color_coverage", "fruit_diameter_mm", "shape_index",
                "ta", "sugar_acid_ratio", "vitamin_c",
                "tpa_springiness", "tpa_cohesiveness", "tpa_chewiness", "juice_pct",
                "qda_color", "qda_odor", "qda_taste", "qda_texture", "qda_overall",
                "qda_sweet", "qda_sour", "qda_astringent", "qda_aroma", "qda_crispness", "qda_juiciness",
                "label", "created_at"]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in recs:
                d = r.to_dict()
                w.writerow({c: d.get(c) for c in cols})
        return len(recs)

    # TODO(远期): 与检测记录联动（上传图片检测后自动建 SensoryRecord）
    # TODO(远期): 电子鼻采集完成后自动回填 odor_csv / odor_features
    # TODO(远期): 前端"感官数据"面板（四维数据展示 + QDA 评分录入）
