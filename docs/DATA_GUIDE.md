# 数据填装指南（四觉一体，你只要填数据就能跑）

面向用户：所有代码已写完并通过测试（81 项）。下面告诉你：往哪里放数据、放什么格式、放好后运行什么命令。
系统入口：http://127.0.0.1:5020/ （顶部导航：实时监测 / 历史数据 / 数据统计 / 数据集 / 感官数据 / 电子鼻）

## 一、总原则

- 一个样本 = 一个 sample_id（建议 20260806_apple_001 这种格式：日期_品种_序号）。
- 四觉数据都用 sample_id 关联，缺哪个模态都不影响其他模态运行。
- 检测页保存到数据集/保存选区时填写 sample_id，原图会自动同步到 multimodal_data/visual/{sample_id}.jpg，照片与电子鼻/理化/QDA 自动对齐，无需手动拷贝。
- 示例/模拟数据在 multimodal_data/_mock_demo/，正式采集前可整体删除。

## 二、每个数据往哪放

模态：视觉（照片），放 multimodal_data/visual/，格式 {sample_id}.jpg；用系统"实时监测"拍照检测后保存到数据集/保存选区时填"样本编号 sample_id"，原图会自动存为 visual/{sample_id}.jpg。
模态：嗅觉（电子鼻），放 multimodal_data/odor/raw/，格式 {sample_id}.csv，用 enose/collect.py 采集。
模态：触觉/理化，放 multimodal_data/touch/touch.csv，格式 sample_id,hardness,weight_g,brix,ph。
模态：人工感官 QDA（多人评分）：感官数据页按"样本编号 + 评鉴人编号"录入，每样本可有多名评鉴人各一条评分，自动存数据库，列表显示各维度"均值±标准差（人数）"，可查看/删除单条。需要脚本分析时也可用 multimodal_data/qda/qda_scores.csv（sample_id,qda_color,...,1~9 分）。
模态：缺陷类型标注，检测页选缺陷块类型后自动写 multimodal_data/defect_blocks.csv。
模态：四觉汇总（可选），放 multimodal_data/manifest.csv，格式 sample_id,label,remark。

## 三、电子鼻硬件与采集

1. 烧录固件：enose/enose_sketch/enose_sketch.ino（ESP32 + MQ 阵列 + BME680 + SGP30，免焊杜邦线方案）。
2. 采集命令：python enose/collect.py --port COM3 --label apple_fresh（--label 写新鲜度档位，会进训练标签）。
3. 先采一次空罐基线（--label baseline），实验方案细节见 enose/protocol.md。

## 四、数据填好后跑什么（按顺序）

第 1 步 气味预处理：raw 下所有 CSV 转成 60 维特征表
   python enose/preprocess.py multimodal_data/odor/raw --out multimodal_data/odor/features.csv

第 2 步 电子鼻模型对比 + 训练（自动存 enose/models/freshness_rf.pkl）
   python enose/train.py multimodal_data/odor/features.csv --compare
   python enose/predict_freshness.py multimodal_data/odor/raw/你的样本.csv --train-from multimodal_data/odor/features.csv

第 3 步 四觉汇总宽表
   python scripts/build_multimodal_table.py --out output/multimodal_table.csv

## 六、系统页面里能直接录的

- 感官数据页：四觉录入（理化 + QDA 多人评分 + 备注），QDA 输入评鉴人编号后按人保存，自动聚合均值±标准差，支持导出 CSV。
- 电子鼻页：拖拽上传气味 CSV，显示曲线 + 60 维特征 + 新鲜度预测（模型训练后）。
- 检测页：拍照/上传，四模型检测，框选重建，保存到数据集，缺陷块类型标注。

## 七、启动服务

先结束旧的 python 进程，再以后台隐藏窗口方式启动系统服务，然后访问 http://127.0.0.1:5020/ 即可。
具体启动方式与项目根目录的 start.bat 一致（重启步骤：结束 python 进程，再启动 E:\apple\app.py）。

## 五、缺陷类型分类器（攒标注后启用）

- 检测页对每个缺陷块下拉选类型（碰伤/腐烂/虫眼/斑点/畸形/果萼果梗/其他），自动存 multimodal_data/defect_blocks.csv。
- 每类攒到约 300 条后训练：
   python train_defect_type.py --csv multimodal_data/defect_blocks.csv --out models/defect_type_model.pkl
- 训练后检测页即可自动显示缺陷类型（当前为人工标注阶段）。

## 八、多模态融合（训练 + 预测）

打开系统感官数据页 http://127.0.0.1:5020/sensory ，底部"多模态融合"卡片：

1. 训练：确认 multimodal_data 已放好视觉图（visual/）、气味 CSV（odor/raw/）、理化表（touch/touch.csv），
   标签放 manifest.csv（sample_id,label）或由样本编号自动推断（fresh/mid/rotten/合格/不合格等关键词）。
   点"开始训练模型"：自动提取视觉 170 维特征（带缓存）、电子鼻 60 维特征，训练视觉/气味/理化三个单模态 RF 模型，
   再做 concat 特征融合和 late 概率融合，页面显示各准确率，报告存 output/sensory_train_report.json，
   模型存 models/sensory_models/。

5. 训练报告明细：报告内每个模型都带 per_class（各类别样本数 n、precision/recall/F1）与混淆矩阵，融合 concat 报告同样带。哪一类 F1 明显偏低，就说明该类样本不足或特征难区分，优先补该类数据。

2. 预测：填样本编号，点"融合预测"，显示各模态预测 + 融合等级与置信度。
   缺少某个模态的数据会自动跳过，两个及以上模态即可融合。

3. 命令行等价操作：
   python services/fusion_service.py
   训练完在脚本内调用 predict()，或直接用系统页面。

4. 说明：单模态用 RF（数据量小、可解释、稳定），融合用 concat + late 两种都训练做对比；
   数据量低于 8 条或只有单类别时会提示样本不足，继续采集即可。
