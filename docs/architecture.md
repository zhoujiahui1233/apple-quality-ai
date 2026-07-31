# 系统架构说明

- 版本：v2.0（多品类扩展版）
- 架构风格：单体 Flask 应用 + 品类插件化检测引擎 + 服务层

## 1. 总体分层

```
┌─────────────────────────────────────────────┐
│  前端 templates/ + static/（实时监测/历史/统计/ │
│  数据集/显著性分析，移动端 PWA）               │
├─────────────────────────────────────────────┤
│  路由层 app.py（Flask，REST + 页面渲染）       │
├─────────────────────────────────────────────┤
│  服务层 services/                            │
│   detection_service  检测编排                │
│   dataset_service    数据集管理               │
│   model_registry     模型版本/回滚            │
│   stats_service      显著性统计               │
├─────────────────────────────────────────────┤
│  检测引擎 detector/                          │
│   base.py 接口  → registry.py 注册表          │
│   apple / bread / beverage 品类适配器         │
│   pipeline.py 统一管线                       │
├─────────────────────────────────────────────┤
│  数据层  SQLite(instance/database.db)        │
│         dataset_collected/ 图片数据集         │
│         models_registry.json 模型版本         │
└─────────────────────────────────────────────┘
```

## 2. 品类适配器（detector/base.py）

新增品类只需实现三个方法并注册：

| 方法 | 职责 |
|------|------|
| `detect_surface(img, mask, contour)` | 返回缺陷区域列表（种类/面积/严重度） |
| `classify_grade(result)` | 按缺陷规则分级（优等/合格/不合格） |
| `visualize(img, result, grade)` | 绘制检测框与文字 |

注册方式（`detector/registry.py`）：`CategoryRegistry.register(BreadAdapter)`。
前端品类切换器会自动读取注册表，新增品类无需改前端。

## 3. 统一检测流程

```
上传图片 + 品类参数（apple / bread / beverage / auto）
  → 预处理（缩放、归一化）
  → 品类自动识别（category=auto 时，基于颜色启发式，detector/autodetect.py）
  → 分割食品区域（HSV / YOLO 定位）
  → 分割食品区域（HSV / YOLO 定位）
  → 品类适配器 detect_surface()
  → 品类适配器 classify_grade()
  → 四模型对比推理（SVM/RF/DL/YOLO，苹果品类）
  → 可视化（检测框 + 热力图）
  → 结果入库（DetectionRecord）
  → 前端展示 + 可选保存到数据集
```

苹果品类保留原有四模型完整逻辑；面包/饮料走通用管线，为后续接入
四模型对比预留接口（`stats_service.model_comparison_table`）。

## 4. 数据闭环

```
手机/电脑拍照 → /api/collect 标注入库
  → dataset_collected/{品类}/{标签}/ 图片
  → 数据集管理页（浏览/删除）
  → 离线训练（SVM/RF/DL/YOLO 脚本）
  → models_registry.json 登记版本+准确率
  → /api/models/rollback 一键回滚
```

## 5. 显著性分析（services/stats_service.py）

- `kappa`：机器分级 vs 人工感官评价一致性
- `mcnemar`：两模型预测差异显著性
- `confusion_stats`：混淆矩阵 + 精确率/召回率/F1
- `model_comparison_table`：四模型×三品类准确率矩阵

## 6. 稳定运行机制

- `start.bat`：一键重启（按端口 PID 精准停止，不影响其他 Python 程序）
- `scripts/backup.py`：代码+数据打包备份，`--full` 含模型权重
- `scripts/run_experiment.py`：在收集数据集上运行模型，生成显著性分析数据（output/experiment_results.json）
- `scripts/export_dataset.py`：数据集按 80/20 切分导出 train/val
- 日志轮转：`logs/app.log` 单文件 5MB，保留 3 份历史（RotatingFileHandler）
- 模型预热：启动时加载全部模型，避免首次请求卡顿

## 7. 配置（config.py）

| 配置 | 说明 |
|------|------|
| `HOST/PORT` | 0.0.0.0:5020，局域网可访问 |
| `MODEL_PATHS` | 各模型权重路径 |
| `PREMIUM_THRESHOLD` | 优等缺陷占比上限（5%） |
| `STANDARD_THRESHOLD` | 不合格缺陷占比阈值（15%） |
| `LOG_*` | 日志级别与格式 |