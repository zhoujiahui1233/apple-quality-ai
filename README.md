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
- **HACCP 品控视图**
- **手动框选导出 YOLO 标注**
- **智能批量导入**：批量选图→自动分割生成检测框→单张修正重新框选→批量/单张标签（合格与不合格+缺陷类型）→统一入库并可同时导出 YOLO 检测标注
：检测后拖动拓拳修正苹果框，直接导出为 YOLO 检测训练集标注（images/labels），可用于重训检测模型
：AI 检测结果映射到关键控制点（CCP），危害分析、CCP 状态监控、质检记录单生成与 Excel 导出（支持苹果/面包/饮料三类品流程）

- **移动端**：手机拍照上传、局域网访问、PWA 离线壳

## 模型评估工具（scripts/evaluate_models.py）

在收集的真实数据集上多维评估当前部署的四模型（SVM/RF/DL/YOLO）+ 缺陷分级管线。

```bash
# 1. 独立测试集评估：准确率 / Precision / Recall / F1 / 混淆矩阵 / ROC / 错误样本
python scripts/evaluate_models.py --mode test

# 2. 分层 K 折交叉验证（数据子集稳定性，均值±标准差）
python scripts/evaluate_models.py --mode cv --folds 5

# 3. 四模型一致性分析（分歧样本 = 待复核候选）
python scripts/evaluate_models.py --mode consistency

# 可选参数：--data 数据集目录（默认 dataset_collected/apple） --limit 每类最多张数
```

输出目录：`output/evaluation/{时间戳}/`，包含：

- **汇总图表**：四指标分组柱状图 `overview.png`、ROC 曲线 `roc_curves.png`
- **混淆矩阵与错误样本**：`cm_*.png`、`errors/`、逐样本明细 `predictions.csv`
- **重训前后对比**：test 模式自动与上次结果对比，生成 `compare_previous.png`（x 标记当前值）
- **交叉验证**：`cv_overview.png`（均值±标准差误差棒图）
- **一致性分析**：`consistency.csv` + `consistency_heatmap.png`（模型两两一致率热图）
- **单文件报告**：`eval_report.html`（图表内嵌，浏览器打开可直接打印/转 PDF）

注意：该工具评估的是当前已部署模型，交叉验证不重新训练模型。

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
├── scripts/run_experiment.py  # 实验脚本（显著性数据生成）
├── scripts/export_dataset.py  # 数据集切分导出（train/val）
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
---

## 登录系统（Supabase）

系统支持邮箱密码登录，用户账号数据保存在 Supabase（云端）。未配置时自动进入「本地模式」，不影响原有使用。

### 1. 创建 Supabase 项目
1. 打开 https://supabase.com 注册并创建新项目（免费档即可）。
2. 项目创建后，在左侧 **Project Settings → API** 中复制：
   - `Project URL`（形如 `https://xxxx.supabase.co`）→ 环境变量 `SUPABASE_URL`
   - `anon public` 密钥 → 环境变量 `SUPABASE_ANON_KEY`

### 2. 设置环境变量（Windows 推荐）
```powershell
$env:SUPABASE_URL="https://xxxx.supabase.co"
$env:SUPABASE_ANON_KEY="eyJ..."
python E:\apple\app.py
```
> 注意：每次新开终端都要重新设置。也可以写入系统环境变量（控制面板 → 系统 → 高级系统设置 → 环境变量）。

### 3.（可选）创建用户资料表 profiles
在 Supabase 的 **SQL Editor** 执行：
```sql
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  nickname text,
  role text default 'user',
  created_at timestamptz default now()
);
alter table public.profiles enable row level security;
create policy "profiles_select" on public.profiles for select using (true);
create policy "profiles_insert" on public.profiles for insert with check (true);
create policy "profiles_update" on public.profiles for update using (true);
```
不建表也不影响登录，只是不保存昵称/角色等扩展资料。

### 4. 邮箱验证说明
- Supabase 默认开启邮箱验证：注册后需点击邮件里的验证链接，再回来登录。
- 如需免验证（测试方便）：Supabase → **Authentication → Providers → Email**，关闭「Confirm email」。

### 5. 说明
- 登录后 Flask 使用会话（Cookie）保持登录状态，检测历史/数据集等仍保存在本机（`instance/database.db` 与本地目录），如需同步到 Supabase 可后续扩展。
- 退出登录：右上角头像旁的退出图标。
## 双模块架构（外观检测 / 切片检测）

系统顶层分为两大独立模块，由页面顶部「外观检测 / 切片检测」切换器切换，数据完全分流（切片模块品类固定为 `apple_slice`）。

- **外观模块（默认）**：/ 实时监测、/history、/stats、/dataset、/annotate、/haccp、/sensory、/enose、/significance
- **切片模块**：/slice 实时监测，其余页面同步支持 `?module=slice`（如 /history?module=slice）
- 切片检测结果自动入库（DetectionRecord.category='apple_slice'），历史/统计/数据集/批量标注按模块分流，互不干扰。
- 切片视觉检测引擎：`slice_detector.py`（自动分割、褐变占比/BI/等级）；嗅觉复用 `enose/` 管线。
- 实验方案：`enose/protocol.md` §9 鲜切苹果（切片）扩展实验。

## 切片分割模型（U-Net）训练流程

切片视觉检测支持两个引擎：**规则算法**（内置，无需模型）与 **U-Net 分割模型**（训练后自动优先使用，无模型时自动回退规则算法）。

1. **攒数据**：切片检测页拍照/上传切片照片并"存入切片数据集"（自动进 `dataset_collected/apple_slice/`）。
2. **生成伪标签**：侧边栏"掩膜标注"页点「生成伪标签」，用规则算法自动为每张图生成 5 类分割掩膜（背景/果肉/果核/褐变/果皮）。
3. **人工修正**：在"掩膜标注"页用画笔工具修正掩膜（1~4 切换类别、0 橡皮、[ ] 调笔刷、Ctrl+Z 撤销、Enter 保存），保存后自动重算训练/验证划分与类别权重。
4. **训练**：
   ```bash
   # 本地 CPU（离线、自定义轻量编码器）
   python scripts/train_slice_unet.py --data slice_seg_dataset --backbone none --epochs 60 --out models/slice_unet.pt
   # 或带预训练编码器（需联网下载权重）
   python scripts/train_slice_unet.py --data slice_seg_dataset --backbone mobilenet_v2 --epochs 80 --out models/slice_unet.pt
   ```
   Kaggle 用法：上传 `slice_seg_dataset` 文件夹与 `scripts/train_slice_unet.py`，GPU 加速器下运行相同命令，完成后把 `models/slice_unet.pt` 拿回本地。
5. **部署**：模型放到 `E:\apple\models\slice_unet.pt` 后重启服务，切片检测页"视觉引擎"选择 **U-Net 分割** 即可；结果区会显示引擎名与褐变占比/分环占比。
6. **效果评估**：训练结束会输出逐类 IoU / Dice、褐变占比 MAE，曲线图保存为 `slice_seg_dataset/train_curves.png`、指标保存为 `metrics.json`。

相关文件：`scripts/slice_unet_model.py`（模型定义，训练/推理共用）、`scripts/slice_seg_prepare.py`（伪标签与数据集构建）、`scripts/train_slice_unet.py`（训练）、`templates/slice_annotate.html`（掩膜标注工作台）。
