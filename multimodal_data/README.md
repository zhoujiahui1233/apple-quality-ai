# 四觉数据目录（multimodal_data/）

> 用法：按下面的规则把数据放进对应文件夹，系统页面/脚本会自动读取。
> 命名规则：所有文件用统一样本编号 `sample_id` 关联，格式建议 `YYYYMMDD_品种_序号`，如 `20260806_apple_001`。

## 目录结构

```
multimodal_data/
├── visual/             # 视觉：苹果/食品照片（jpg/png），命名 {sample_id}.jpg（系统保存时填 sample_id 自动写入）
├── odor/
│   ├── raw/            # 电子鼻原始采集 CSV，命名 {sample_id}.csv（collect.py 输出格式）
│   └── features.csv    # （自动生成）气味特征表，由 enose/preprocess.py 输出
├── touch/              # 触觉/理化：touch.csv 可含 sample_id,hardness,weight_g,brix,ph,ta,sugar_acid_ratio,lab_*,fruit_diameter_mm,shape_index,tpa_*
├── qda/                # 人工感官评分（多人模式在系统感官页录入存数据库，此 CSV 供脚本分析）：qda_scores.csv 可含 sample_id,qda_color,qda_odor,qda_taste,qda_texture,qda_overall,qda_sweet,qda_sour,qda_astringent,qda_aroma,qda_crispness,qda_juiciness
├── _mock_demo/         # 开发期示例/测试数据（可整体删除，正式采集前清空即可）
└── manifest.csv        # （可选）总表：sample_id,label,remark
```

## 各数据怎么来（对照采集流程）

1. **视觉**：用系统"实时监测"拍照检测，保存到数据集/保存选区时填 sample_id，原图自动写入 `visual/{sample_id}.jpg`；或直接把手机照片按 `{sample_id}.jpg` 放入 `visual/`
2. **嗅觉**：电子鼻采样 → `python enose/collect.py --port COM3 --label apple --out multimodal_data/odor/raw/{sample_id}.csv`
3. **触觉/理化**：硬度计测硬度、糖度计测 Brix、pH 笔测 pH、可滴定酸度 TA、质构仪测 TPA（弹性/内聚性/咀嚼性）、色差仪测 Lab → 在系统"感官数据"页面录入（或写进 `touch/touch.csv`）
4. **人工感官**：5~8 人 QDA 评分（色/香/味/质地 + 甜/酸/涩/果香/脆/多汁，1~9 分）→ 在系统"感官数据"页面录入，每样本每评鉴人一条（页面自动显示均值±标准差）；需要脚本分析时再汇总写进 `qda/qda_scores.csv`

## 一键流水线（数据填好后运行）

```bash
# 1) 气味预处理：raw 下所有 CSV -> 60 维特征表
python enose/preprocess.py multimodal_data/odor/raw --out multimodal_data/odor/features.csv

# 2) 电子鼻模型对比 + 训练
python enose/train.py multimodal_data/odor/features.csv --compare
python enose/predict_freshness.py --train-from multimodal_data/odor/features.csv

# 3) 四觉数据汇总导出（把 visual/odor/touch/qda 合成一张表）
python scripts/build_multimodal_table.py --out output/multimodal_table.csv

# 4) 多模态融合实验（单模态 vs 融合）
python scripts/multimodal_fusion_experiment.py --visual output/visual_features.csv \
    --odor multimodal_data/odor/features.csv --phys touch/touch.csv

# 5) 仪器-感官相关性分析
python scripts/sensory_correlation.py --scores multimodal_data/qda/qda_scores.csv \
    --feature output/multimodal_table.csv
```

## 说明
- 系统页面：顶部导航"感官数据"（四觉录入与总表）、"电子鼻"（气味上传/曲线/预测）
- 所有表以 `sample_id` 为主键对齐，缺某个模态不影响其他模态

## 关于 _mock_demo（示例数据）

`_mock_demo/` 里是开发时用于验证流程的模拟数据（气味 CSV、touch.csv、qda_scores.csv、
defect_blocks.csv、mock 训练的电子鼻模型等），**不是真实数据**。

- 作用：展示每种文件的格式，方便对照；也用来验证过整条流水线。
- 正式采集前：整个 `_mock_demo/` 文件夹可直接删除（或移走），不影响系统。
- 删除后从零开始采集，按上方规则放数据即可。
