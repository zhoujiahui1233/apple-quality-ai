# 智果 · AI 品控系统

基于机器视觉与深度学习的**多品类食品品质无损检测与分级系统**（本科毕业设计项目，食品科学与工程专业）。

支持苹果、面包、饮料三类食品的表面缺陷检测与三级分级（优等 / 合格 / 不合格），集成 **SVM、随机森林、深度学习、YOLO** 四种模型，并提供显著性分析、数据闭环与移动端访问。

## 功能特性

- **多品类检测**：苹果（疤痕/腐烂/磕伤/污点）、面包（霉斑/焦糊）、饮料（液位/污渍/标签）
- **四模型对比**：SVM、随机森林、深度学习（CNN）、YOLO 统一检测，附 Grad-CAM 热力图
- **三级分级**：优等 / 合格 / 不合格，附带置信度与判定依据
- **数据闭环**：拍照 → 检测 → 标注入库 → 数据集管理 → 重训 → 模型版本管理与回滚
- **显著性分析**：四模型×三品类准确率矩阵、McNemar 检验、Kappa 一致性、混淆矩阵
- **报告导出**：单次检测报告与批次汇总报告（Word/PDF）
- **移动端**：手机拍照上传、局域网访问、PWA 离线壳

## 快速开始

```bash
# 1. 安装依赖（建议 conda/pip 环境）
pip install -r requirements.txt

# 2. 启动服务（两种方式任选）
双击 start.bat
# 或
python app.py

# 3. 浏览器访问
# 本机: http://127.0.0.1:5020/
# 手机（同一局域网）: http://<电脑局域网IP>:5020/
```

首次启动会预加载模型，约需 10~15 秒。

## 目录结构

```
E:\apple
├── app.py                      # Flask 主服务（路由）
├── config.py                   # 集中配置（路径/阈值/端口）
├── models.py                   # 数据库模型（检测记录）
├── apple_defect_detector.py    # 苹果检测算法库（分割/特征/四模型）
├── detector/                   # 品类适配器框架
│   ├── base.py                 # 检测接口与数据结构
│   ├── registry.py             # 品类注册表
│   ├── apple.py                # 苹果适配器
│   ├── bread.py                # 面包适配器
│   ├── beverage.py             # 饮料适配器
│   └── pipeline.py             # 统一检测管线
├── services/                   # 业务逻辑层
│   ├── detection_service.py    # 检测编排
│   ├── dataset_service.py      # 数据集管理
│   ├── model_registry.py       # 模型版本管理
│   └── stats_service.py        # 显著性统计
├── templates/                  # 前端页面（实时监测/历史/统计/数据集/显著性）
├── static/                     # 静态资源与上传图片
├── tests/                      # pytest 单元测试
├── scripts/backup.py           # 备份工具
├── dataset_collected/          # 收集的数据集（按 品类/标签 两级）
├── logs/                       # 运行日志（自动轮转）
└── backups/                    # 备份压缩包
```

## 测试

```bash
python -m pytest tests/ -v
```

## 备份与恢复

```bash
python scripts/backup.py            # 备份代码+数据（不含大模型权重）
python scripts/backup.py --full     # 全量备份（含模型权重）
python scripts/backup.py --list     # 查看已有备份
python scripts/backup.py --clean 5  # 只保留最新 5 个备份
```

备份文件位于 `backups/backup_YYYYMMDD_HHMMSS.zip`。

## 关键技术

- 前端：深色科技蓝 + 玻璃态风格，Chart.js 可视化，PWA
- 后端：Flask + SQLite（Flask-SQLAlchemy）
- 视觉：OpenCV（HSV 分割）、PyTorch、Ultralytics YOLO
- 统计：scipy（McNemar/Kappa/相关分析）

## 论文相关

- 设计文档：`docs/superpowers/specs/2026-07-31-food-quality-system-design.md`
- 论文大纲：`docs/thesis-outline.md`