# -*- coding: utf-8 -*-
"""Build new index.html head (keep JS section intact)."""
import io, os, re

TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates', 'index.html')
with io.open(TPL, 'r', encoding='utf-8') as f:
    old = f.read()

# JS section: from first <script> to end of file (keeps all logic byte-for-byte)
idx = old.find('<script>')
assert idx > 0, 'no <script> found'
js_part = old[idx:]

NEW_HEAD = r'''{% extends "base.html" %}
{% block title %}实时监测{% endblock %}
{% block content %}
<style>
.stream-area { position: relative; background: #000; border-radius: 12px; overflow: hidden; min-height: 420px; }
.stream-area img { width: 100%; display: block; }
.stream-badge { position: absolute; top: 12px; left: 12px; background: rgba(0,0,0,0.7); backdrop-filter: blur(8px); padding: 5px 12px; border-radius: 8px; font-size: 12px; border: 1px solid var(--border); z-index: 2; }
.stream-badge i { margin-right: 6px; }
.model-switch { position: absolute; top: 12px; right: 12px; display: flex; gap: 6px; z-index: 2; }
.stream-status { position: absolute; bottom: 12px; left: 12px; background: rgba(0,0,0,0.6); padding: 4px 12px; border-radius: 6px; font-size: 11px; z-index: 2; }
.dropzone { cursor: pointer; transition: all 0.3s; }
.dropzone:hover { border-color: var(--primary) !important; background: rgba(0,212,255,0.05) !important; }
#saveBarToggle, #quickSaveBtn { pointer-events: auto; }
/* 卡片面板头/体 */
.panel-head {
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  padding: 12px 16px; border-bottom: 1px solid var(--border);
}
.panel-head .t { display: flex; align-items: center; gap: 8px; font-size: 13px; font-weight: 600; color: var(--text); }
.panel-head .t i { color: var(--primary); }
.panel-body { padding: 14px 16px; }
.action-bar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.opt-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; font-size: 12px; color: var(--text-dim); }
.opt-row label { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; margin: 0; }
.opt-row input[type=checkbox], .opt-row input[type=radio] { accent-color: var(--primary); }
@media (max-width: 768px) {
  #saveToDatasetBar { max-height: 62%; overflow-y: auto; }
  #saveToDatasetBar button { white-space: nowrap; }
  .model-btn { padding: 4px 8px; font-size: 10px; }
}
</style>

<div class="row g-3">
  <!-- ============ 左：检测工作区 ============ -->
  <div class="col-lg-7">
    <div class="glass-card" style="overflow:hidden;">
      <div class="panel-head">
        <span class="t"><i class="fas fa-crosshairs"></i> 检测工作区</span>
        <span class="hint-text">拖拽 / 点击 / 拍照上传，实时 AI 检测</span>
      </div>
      <div class="stream-area" id="streamContainer">
        <div id="uploadZone" class="dropzone d-flex flex-column align-items-center justify-content-center" onclick="document.getElementById('fileInput').click()" style="min-height:420px;">
          <i class="fas fa-cloud-upload-alt" style="font-size:48px;color:var(--primary-dim);margin-bottom:12px;"></i>
          <p style="color:var(--text-dim);font-size:14px;margin-bottom:4px;">拖拽或点击上传图片</p>
          <p style="color:var(--text-dim);font-size:11px;margin-bottom:16px;">支持 JPG / PNG / BMP</p>
          <div class="action-bar" style="justify-content:center;">
            <button class="btn-p btn-sm2" onclick="event.stopPropagation();document.getElementById('fileInput').click()"><i class="fas fa-folder-open"></i> 选择图片</button>
            <button class="btn-p btn-sm2" onclick="event.stopPropagation();document.getElementById('fileInput').click()"><i class="fas fa-images"></i> 批量检测</button>
            <button class="btn-a btn-sm2" onclick="event.stopPropagation();toggleCamera()" id="camBtn"><i class="fas fa-camera"></i> 摄像头</button>
          </div>
          <input type="file" class="d-none" id="fileInput" accept="image/*" multiple>
        </div>
        <div id="cameraContainer" class="d-none" style="position:relative;min-height:420px;background:#000;">
          <video id="videoFeed" autoplay playsinline style="width:100%;max-height:420px;object-fit:contain;"></video>
          <canvas id="photoCanvas" class="d-none"></canvas>
          <div style="position:absolute;bottom:12px;left:50%;transform:translateX(-50%);display:flex;gap:10px;z-index:2;">
            <button class="btn-d btn-sm2" onclick="stopCamera()"><i class="fas fa-stop"></i> 关闭</button>
            <button class="btn-a btn-sm2" onclick="captureAndDetect()"><i class="fas fa-camera"></i> 拍照检测</button>
            <button class="btn-p btn-sm2" onclick="startContinuous()" id="contBtn"><i class="fas fa-sync-alt"></i> 连续</button>
          </div>
        </div>
        <img id="resultImage" class="d-none" style="width:100%;min-height:420px;object-fit:contain;" alt="">
        <div id="cropHint" class="d-none" style="position:absolute;top:12px;left:50%;transform:translateX(-50%);z-index:6;background:rgba(0,0,0,0.72);color:var(--primary);padding:4px 16px;border-radius:8px;font-size:12px;pointer-events:none;white-space:nowrap;">拖动框选苹果区域</div>
        <div id="cropConfirmBar" class="d-none" style="position:absolute;bottom:48px;left:50%;transform:translateX(-50%);z-index:6;display:flex;gap:8px;align-items:center;background:rgba(10,18,34,0.95);border:1px solid var(--warning);border-radius:10px;padding:6px 10px;">
          <span id="cropConfirmText" style="font-size:12px;color:var(--text);white-space:nowrap;"></span>
          <button class="btn-w btn-sm2" onclick="cropRedoSelect()"><i class="fas fa-redo"></i> 重新框选</button>
          <button id="cropConfirmOk" class="btn-a btn-sm2" onclick="cropConfirmOk()"><i class="fas fa-check"></i> 确认</button>
        </div>
        <div id="cropActionBar" class="d-none" style="position:absolute;bottom:48px;left:50%;transform:translateX(-50%);z-index:6;display:flex;flex-wrap:wrap;justify-content:center;gap:8px;max-width:96%;background:rgba(10,18,34,0.92);border:1px solid var(--primary);border-radius:10px;padding:6px 10px;">
          <button class="btn-w btn-sm2" onclick="cropRedoSelect()"><i class="fas fa-redo"></i> 重新框选</button>
          <button id="cropSaveBtn" class="btn-p btn-sm2" onclick="cropSaveToDataset()">💾 保存选区</button>
          <span style="border-left:1px solid var(--border);padding-left:10px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
            <span style="font-size:11px;color:var(--text-dim);">标签</span>
            <label style="font-size:12px;color:var(--text);cursor:pointer;margin:0;"><input type="radio" name="cropLabel" value="合格" checked style="accent-color:var(--accent);" onchange="onCropLabelChange()"> 合格</label>
            <label style="font-size:12px;color:var(--text);cursor:pointer;margin:0;"><input type="radio" name="cropLabel" value="不合格" style="accent-color:var(--danger);" onchange="onCropLabelChange()"> 不合格</label>
            <select id="cropDefectType" style="display:none;padding:4px 8px;border-radius:6px;border:1px solid var(--border);background:rgba(0,0,0,0.7);color:var(--text);font-size:12px;">
              <option value="">缺陷类型</option>
              <option value="碰伤">碰伤</option>
              <option value="腐烂">腐烂</option>
              <option value="虫眼">虫眼</option>
              <option value="斑点">斑点</option>
              <option value="畸形">畸形</option>
              <option value="其他">其他</option>
            </select>
          </span>
          <button class="btn-g btn-sm2" onclick="exitCropMode()">取消</button>
        </div>
        <div class="stream-overlay" style="position:absolute;inset:0;pointer-events:none;">
          <div class="stream-badge" id="streamBadge" style="pointer-events:none;"><i class="fas fa-circle" style="color:var(--danger);font-size:8px;"></i> LIVE · 等待上传</div>
          <div class="model-switch" style="pointer-events:auto;">
            <div class="model-btn active" onclick="switchModel('yolo')">YOLO</div>
            <div class="model-btn" onclick="switchModel('dl')">DL</div>
            <div class="model-btn" onclick="switchModel('rf')">RF</div>
            <div class="model-btn" onclick="switchModel('svm')">SVM</div>
          </div>
          <div class="stream-status" id="streamStatus"><i class="fas fa-arrow-circle-up"></i> 上传图片开始检测</div>
          <button id="quickSaveBtn" class="d-none" onclick="quickSave()" style="pointer-events:auto;position:absolute;bottom:12px;right:12px;z-index:4;padding:8px 16px;border-radius:20px;border:1px solid var(--accent);background:rgba(0,255,136,0.18);color:var(--accent);font-size:13px;cursor:pointer;backdrop-filter:blur(8px);"><i class="fas fa-save"></i> 保存</button>
          <button id="reviewBtn" class="d-none" onclick="location.href='/annotate'" style="pointer-events:auto;position:absolute;bottom:12px;right:100px;z-index:4;padding:8px 14px;border-radius:20px;border:1px solid var(--warning);background:rgba(255,193,7,0.16);color:var(--warning);font-size:13px;cursor:pointer;backdrop-filter:blur(8px);"><i class="fas fa-user-check"></i> 待复核</button>
          <button id="saveBarToggle" class="d-none" onclick="toggleSaveBar()" style="pointer-events:auto;position:absolute;bottom:56px;right:12px;z-index:4;padding:8px 16px;border-radius:20px;border:1px solid var(--primary);background:rgba(0,212,255,0.15);color:var(--primary);font-size:13px;cursor:pointer;backdrop-filter:blur(8px);"><i class="fas fa-list-alt"></i> 展开</button>
          <div id="saveToDatasetBar" class="d-none" style="position:absolute;bottom:48px;left:12px;right:12px;z-index:3;pointer-events:auto;background:rgba(10,18,34,0.88);backdrop-filter:blur(12px);border:1px solid var(--border);border-radius:10px;padding:10px 12px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
              <span style="font-size:12px;color:var(--text);font-weight:600;"><i class="fas fa-list-alt"></i> 检测结果 · 保存到数据集</span>
              <button class="btn-g btn-sm2" onclick="toggleSaveBar()">收起 ▲</button>
            </div>
            <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
              <span style="font-size:12px;color:var(--text-dim);">标签:</span>
              <label style="font-size:12px;color:var(--text);cursor:pointer;"><input type="radio" name="dsLabel" value="合格" checked style="accent-color:var(--accent);" onchange="onDsLabelChange()"> 合格</label>
              <label style="font-size:12px;color:var(--text);cursor:pointer;"><input type="radio" name="dsLabel" value="不合格" style="accent-color:var(--danger);" onchange="onDsLabelChange()"> 不合格</label>
              <select id="defectType" style="display:none;padding:4px 8px;border-radius:6px;border:1px solid var(--border);background:rgba(0,0,0,0.7);color:var(--text);font-size:12px;">
                <option value="">缺陷类型</option>
                <option value="碰伤">碰伤</option>
                <option value="腐烂">腐烂</option>
                <option value="虫眼">虫眼</option>
                <option value="斑点">斑点</option>
                <option value="畸形">畸形</option>
                <option value="其他">其他</option>
              </select>
              <button class="btn-p btn-sm2" onclick="toggleSmartPanel()"><i class="fas fa-sliders-h"></i> 智能选项</button>
              <button class="btn-p btn-sm2" onclick="smartPreview()"><i class="fas fa-search"></i> 智能分析</button>
              <button class="btn-w btn-sm2" onclick="enterCropMode()"><i class="fas fa-crop-alt"></i> 框选重检</button>
              <button class="btn-a btn-sm2" onclick="saveToDataset()"><i class="fas fa-save"></i> 保存到数据集</button>
            </div>
          </div>
        </div>
      </div>
      <div class="panel-body">
        <div class="action-bar">
          <button class="btn-p btn-sm2" onclick="toggleSmartPanel()"><i class="fas fa-sliders-h"></i> 智能选项</button>
          <button class="btn-p btn-sm2" onclick="smartPreview()"><i class="fas fa-search"></i> 智能分析</button>
          <button class="btn-w btn-sm2" onclick="enterCropMode()"><i class="fas fa-crop-alt"></i> 框选重检</button>
          <button class="btn-a btn-sm2" onclick="saveToDataset()"><i class="fas fa-save"></i> 保存到数据集</button>
        </div>
        <div id="fusionInfo" style="display:none;margin-top:10px;padding:9px 12px;border:1px solid var(--primary);border-radius:8px;background:rgba(0,212,255,0.06);font-size:12px;"></div>
        <div id="defectBlocksInfo" style="display:none;margin-top:10px;padding:9px 12px;border:1px solid var(--warning);border-radius:8px;background:rgba(255,193,7,0.05);font-size:12px;"></div>
        <div class="hint-text" style="margin-top:8px;"><i class="fas fa-sync-alt"></i> 提示：保存前若手动修改了「合格/不合格」标签，保存后会同步修正本条历史记录的等级。</div>
        <div id="smartPanel" style="display:none;margin-top:10px;padding-top:10px;border-top:1px solid var(--border);gap:14px;flex-wrap:wrap;font-size:12px;color:var(--text-dim);">
          <label style="cursor:pointer;" title="勾选后会额外保存 1 张去背景裁剪图（共 2 张）"><input type="checkbox" id="optCrop" style="accent-color:var(--primary);"> 自动裁剪主体(去背景)（另存1张）</label>
          <label style="cursor:pointer;"><input type="checkbox" id="optQuality" checked style="accent-color:var(--primary);"> 质量检查(模糊/曝光)</label>
          <label style="cursor:pointer;"><input type="checkbox" id="optDedup" checked style="accent-color:var(--primary);"> 去重检查</label>
          <label style="cursor:pointer;"><input type="checkbox" id="optAugment" style="accent-color:var(--primary);"> 数据增强(1变4)</label>
          <label style="cursor:pointer;"><input type="checkbox" id="optForce" style="accent-color:var(--danger);"> 强制保存</label>
        </div>
        <div id="smartResult" style="display:none;margin-top:10px;padding:9px 12px;border-radius:8px;background:rgba(0,0,0,0.4);font-size:12px;color:var(--text);line-height:1.8;"></div>
      </div>
    </div>

    <!-- ============ 批量检测 ============ -->
    <div class="glass-card mt-3">
      <div class="panel-head" onclick="toggleBatch()" style="cursor:pointer;">
        <span class="t"><i class="fas fa-images"></i> 批量检测 <span id="batchCount" style="color:var(--primary);font-weight:600;"></span></span>
        <span class="hint-text"><i class="fas fa-chevron-down" id="batchArrow"></i></span>
      </div>
      <div id="batchPanel" class="panel-body">
        <div class="opt-row">
          <label style="cursor:pointer;"><input type="checkbox" id="autoReviewCb" checked style="accent-color:var(--primary);"> 不确定结果自动进复核队列</label>
          <label style="cursor:pointer;"><input type="checkbox" id="batchSaveGood" checked style="accent-color:var(--accent);"> 导出时存为合格样本</label>
          <span style="flex:1;"></span>
          <button id="batchYoloBtn" class="btn-p btn-sm2" onclick="exportBatchYolo()"><i class="fas fa-crop-alt"></i> 一键导出YOLO标注</button>
          <button class="btn-p btn-sm2" onclick="document.getElementById('fileInput').click()"><i class="fas fa-folder-open"></i> 选择多张图片</button>
        </div>
        <div id="batchProgressWrap" class="d-none mt-2">
          <div style="height:6px;background:rgba(0,212,255,0.1);border-radius:4px;overflow:hidden;"><div id="batchProgressBar" style="height:100%;width:0%;background:linear-gradient(90deg,var(--primary),var(--accent));transition:width .3s;"></div></div>
          <div id="batchProgressText" style="font-size:12px;color:var(--text-dim);margin-top:6px;"></div>
        </div>
        <div id="batchTableWrap" class="d-none mt-2" style="max-height:260px;overflow-y:auto;">
          <table class="table table-sm table-dark mb-0" style="font-size:12px;color:var(--text);">
            <thead><tr><th>#</th><th>文件名</th><th>等级</th><th>严重度</th><th>复核</th><th>操作</th></tr></thead>
            <tbody id="batchTableBody"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ============ 参数调节 ============ -->
    <div class="glass-card mt-3">
      <div class="panel-head" onclick="toggleTuning()" style="cursor:pointer;">
        <span class="t"><i class="fas fa-sliders-h"></i> 参数调节</span>
        <span class="hint-text"><span id="tuningArrow">▶</span> 展开</span>
      </div>
      <div id="tuningPanel" class="panel-body" style="display:none;">
        <div class="row g-2">
          <div class="col-6"><label class="small text-muted">灵敏度: <strong id="sensVal">5</strong></label><input type="range" class="w-100" id="sensitivity" min="1" max="10" value="5" oninput="sensVal.textContent=this.value"></div>
          <div class="col-6"><label class="small text-muted">最小面积: <strong id="areaVal">200</strong></label><input type="range" class="w-100" id="minArea" min="50" max="500" value="200" step="10" oninput="areaVal.textContent=this.value"></div>
          <div class="col-6"><label class="small text-muted">优质阈值: <strong id="premVal">5.0</strong>%</label><input type="range" class="w-100" id="premiumTh" min="1" max="15" value="5" step="0.5" oninput="premVal.textContent=this.value"></div>
          <div class="col-6"><label class="small text-muted">合格阈值: <strong id="stdVal">15.0</strong>%</label><input type="range" class="w-100" id="standardTh" min="5" max="30" value="15" step="0.5" oninput="stdVal.textContent=this.value"></div>
          <div class="col-12"><label style="font-size:12px;color:var(--text-dim);cursor:pointer;margin:0;"><input type="checkbox" id="wbCb" style="accent-color:var(--primary);"> 颜色校正（偏色/灯光偏黄时启用，检测前自动白平衡）</label></div>
        </div>
        <button class="btn-g btn-sm2 mt-2" onclick="resetTuning()">恢复默认</button>
      </div>
    </div>
  </div>

  <!-- ============ 右：数据看板 ============ -->
  <div class="col-lg-5">
    <div class="row g-3">
      <div class="col-6">
        <div class="glass-card p-3" style="height:100%;">
          <div class="card-title"><i class="fas fa-chart-bar"></i> 模型准确率</div>
          <div class="chart-box"><canvas id="accChart"></canvas></div>
        </div>
      </div>
      <div class="col-6">
        <div class="glass-card p-3" style="height:100%;">
          <div class="card-title"><i class="fas fa-chart-pie"></i> 品质分布</div>
          <div class="chart-box-lg"><canvas id="donutChart"></canvas></div>
        </div>
      </div>
      <div class="col-6">
        <div class="glass-card p-3">
          <div class="card-title"><i class="fas fa-microchip"></i> 模型性能</div>
          <div class="row g-2">
            <div class="col-4"><div class="metric-card"><div class="value" id="metricLatency">--</div><div class="label">推理耗时</div></div></div>
            <div class="col-4"><div class="metric-card"><div class="value" id="metricMemory">--</div><div class="label">内存占用</div></div></div>
            <div class="col-4"><div class="metric-card"><div class="value" id="metricGPU">--</div><div class="label">GPU 利用率</div></div></div>
          </div>
        </div>
      </div>
      <div class="col-6">
        <div class="glass-card p-3">
          <div class="card-title"><i class="fas fa-list"></i> 检测日志</div>
          <div class="log-container" id="logContainer">
            <div style="text-align:center;padding:20px;color:var(--text-dim);font-size:12px;">等待检测...</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<div id="loadingOverlay" class="d-none" style="position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,0.6);display:flex;align-items:center;justify-content:center;">
  <div class="text-center"><div class="spinner-border" style="color:var(--primary);width:3rem;height:3rem;"></div><p class="mt-3" style="color:var(--text-dim);">AI 分析中...</p></div>
</div>
<script>
function toggleBatch() {
  var p = document.getElementById('batchPanel');
  var a = document.getElementById('batchArrow');
  if (!p || !a) return;
  var open = p.style.display !== 'none';
  if (open) { p.style.display = 'none'; a.className = 'fas fa-chevron-right'; }
  else { p.style.display = ''; a.className = 'fas fa-chevron-down'; }
}
</script>
'''

with io.open(TPL, 'w', encoding='utf-8', newline='\n') as f:
    f.write(NEW_HEAD + js_part)
print('index.html written:', os.path.getsize(TPL), 'bytes')
