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
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        }
