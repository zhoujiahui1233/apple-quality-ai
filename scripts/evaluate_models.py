# -*- coding: utf-8 -*-
"""
模型多维度评估工具（独立测试集 / 交叉验证 / 一致性分析）+ 论文图表 + 重训前后对比

用法:
  python scripts/evaluate_models.py                      # 模式1：独立测试集评估
  python scripts/evaluate_models.py --mode test          # 同上
  python scripts/evaluate_models.py --mode cv --folds 5  # 模式2：分层K折交叉验证
  python scripts/evaluate_models.py --mode consistency   # 模式3：四模型一致性分析
  python scripts/evaluate_models.py --data 数据集目录 --limit 每类最多张数

输出:
  output/evaluation/{时间戳}/
    report.json / predictions.csv / eval_report.html（单文件报告，可打印/转PDF）
    cm_{model}.png / overview.png / roc_curves.png / compare_previous.png
    cv_summary.txt / cv_folds.csv / cv_overview.png
    consistency.csv / consistency_heatmap.png
"""
import argparse
import base64
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc)
from sklearn.model_selection import StratifiedKFold

import config
import run_experiment as rexp  # 复用: imread_unicode / run_apple / 评分函数

MODELS = ["svm", "rf", "dl", "yolo"]
MODELS_ACTIVE = list(MODELS)  # runtime-filtered via --models
DATASET_DEFAULT = config.ROOT_DIR / "dataset_collected" / "apple"
OUT_ROOT = config.OUTPUT_DIR / "evaluation"
M_COLORS = {"svm": "#4e79a7", "rf": "#f28e2b", "dl": "#59a14f", "yolo": "#e15759"}


def load_samples(data_dir: Path, limit=None):
    """返回 [(path, yb)]，yb: good/bad；排除 *_crop.jpg"""
    samples = []
    if not data_dir.is_dir():
        raise SystemExit(f"数据集目录不存在: {data_dir}")
    for label_dir in sorted(data_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        yb, _y3 = rexp.label_classes(label_dir.name)
        files = sorted(label_dir.glob("*"))
        if limit:
            files = files[:limit]
        for f in files:
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp") and "_crop." not in f.name:
                samples.append((f, yb))
    if not samples:
        raise SystemExit(f"目录 {data_dir} 下没有标注图片（需要 {data_dir}/合格 与 /不合格 子目录）")
    return samples


def run_one(path, yb):
    img = rexp.imread_unicode(path)
    if img is None:
        return None
    row = rexp.run_apple(path, yb, None)
    if row is None:
        return None
    row["file"] = str(path.relative_to(config.ROOT_DIR))
    row["y_true"] = yb
    row["y3_true"] = "reject" if yb == "bad" else "premium"
    return row


def active_models():
    return MODELS_ACTIVE


def collect_predictions(samples):
    rows, skipped = [], 0
    for path, yb in samples:
        row = run_one(path, yb)
        if row is None:
            skipped += 1
            continue
        # drop inactive models' predictions to keep metrics consistent
        for m in MODELS:
            if m not in MODELS_ACTIVE:
                row.pop(m, None)
        rows.append(row)
    return rows, skipped


def binary_metrics(y_true, y_pred):
    yt = np.asarray([1 if v == "bad" else 0 for v in y_true])
    yp = np.asarray([1 if v == "bad" else 0 for v in y_pred])
    return {"accuracy": round(accuracy_score(yt, yp), 4),
            "precision": round(precision_score(yt, yp, zero_division=0), 4),
            "recall": round(recall_score(yt, yp, zero_division=0), 4),
            "f1": round(f1_score(yt, yp, zero_division=0), 4)}


def make_confusion_matrix(y_true, y_pred, out_path, title, lang="zh", dpi=150):
    yt = [1 if v == "bad" else 0 for v in y_true]
    yp = [1 if v == "bad" else 0 for v in y_pred]
    cm = confusion_matrix(yt, yp)
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["good", "bad"]); ax.set_yticklabels(["good", "bad"])
    ax.set_xlabel("预测" if lang == "zh" else "Predicted")
    ax.set_ylabel("真实" if lang == "zh" else "Actual"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)
    return cm.tolist()


def make_overview(metrics, out_path, title=None, lang="zh", dpi=150):
    if title is None:
        title = "四模型真实数据评估对比" if lang == "zh" else "Real-data Evaluation of Four Models"
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(MODELS_ACTIVE))
    width = 0.2
    keys = [("accuracy", "准确率" if lang == "zh" else "Accuracy", "#4e79a7"),
            ("precision", "精确率" if lang == "zh" else "Precision", "#f28e2b"),
            ("recall", "召回率" if lang == "zh" else "Recall", "#59a14f"),
            ("f1", "F1", "#e15759")]
    for i, (k, label, color) in enumerate(keys):
        vals = [metrics.get(m, {}).get(k, 0) for m in MODELS_ACTIVE]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color)
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in MODELS_ACTIVE])
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9, ncol=4, loc="upper center")
    ax.set_title(title); ax.set_ylabel("得分" if lang == "zh" else "Score")
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


def make_roc(rows, out_path, lang="zh", dpi=150):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    y_true_all = [1 if r["y_true"] == "bad" else 0 for r in rows]
    drawn = 0
    for m in MODELS_ACTIVE:
        scores, ys = [], []
        for r in rows:
            sc = (r.get("scores") or {}).get(m)
            if sc is None:
                continue
            scores.append(sc); ys.append(1 if r["y_true"] == "bad" else 0)
        if len(scores) < 5 or len(set(ys)) < 2:
            continue
        fpr, tpr, _ = roc_curve(ys, scores)
        ax.plot(fpr, tpr, label=f"{m.upper()} (AUC={auc(fpr, tpr):.3f})", lw=1.6, color=M_COLORS[m])
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("假阳性率 (FPR)" if lang == "zh" else "False Positive Rate (FPR)")
    ax.set_ylabel("真阳性率 (TPR)" if lang == "zh" else "True Positive Rate (TPR)")
    ax.set_title("ROC 曲线（bad=positive）" if lang == "zh" else "ROC Curves (positive = defective)"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)
    return out_path


def make_cv_overview(summary, out_path, lang="zh", dpi=150):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(MODELS_ACTIVE))
    means, stds = [], []
    for m in MODELS_ACTIVE:
        v = summary.get(m)
        means.append(v["mean"] if v else 0)
        stds.append(v["std"] if v else 0)
    ax.bar(x, means, yerr=stds, capsize=5, color=[M_COLORS[m] for m in MODELS_ACTIVE], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in MODELS_ACTIVE])
    ax.set_ylim(0, 1.05); ax.set_ylabel("准确率" if lang == "zh" else "Accuracy")
    ax.set_title("分层 K 折交叉验证（均值 ± 标准差）" if lang == "zh"
                 else "Stratified K-Fold Cross-Validation (mean ± std)")
    for i, (me, sd) in enumerate(zip(means, stds)):
        ax.text(i, me + sd + 0.03, f"{me:.3f}\n±{sd:.3f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


def make_consistency_heatmap(pair_agree, out_path, lang="zh", dpi=150):
    mat = np.ones((len(MODELS_ACTIVE), len(MODELS_ACTIVE)))
    for i, a in enumerate(MODELS_ACTIVE):
        for j, b in enumerate(MODELS_ACTIVE):
            if a == b:
                mat[i, j] = 1.0
            else:
                k = f"{a}-{b}" if a < b else f"{b}-{a}"
                mat[i, j] = pair_agree.get(k) or 0
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(mat, cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(MODELS_ACTIVE))); ax.set_yticks(range(len(MODELS_ACTIVE)))
    ax.set_xticklabels([m.upper() for m in MODELS_ACTIVE]); ax.set_yticklabels([m.upper() for m in MODELS_ACTIVE])
    for i in range(len(MODELS_ACTIVE)):
        for j in range(len(MODELS_ACTIVE)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="black" if mat[i, j] > 0.6 else "white")
    ax.set_title("四模型两两预测一致率热图" if lang == "zh"
                 else "Pairwise Prediction Agreement Heatmap")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


def find_previous_test(current_dir):
    """在 OUT_ROOT 下找最近一次 test 模式的评估（排除当前目录）"""
    for d in sorted(OUT_ROOT.iterdir(), reverse=True):
        if d == current_dir or not d.is_dir():
            continue
        rp = d / "report.json"
        if not rp.exists():
            continue
        try:
            j = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if j.get("mode") == "test":
            return d, j
    return None, None


def make_compare(prev_metrics, cur_metrics, out_path, lang="zh", dpi=150):
    """重训/数据更新前后的指标对比图"""
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(MODELS_ACTIVE))
    width = 0.2
    keys = [("accuracy", "准确率" if lang == "zh" else "Accuracy", "#4e79a7"),
            ("precision", "精确率" if lang == "zh" else "Precision", "#f28e2b"),
            ("recall", "召回率" if lang == "zh" else "Recall", "#59a14f"),
            ("f1", "F1", "#e15759")]
    for i, (k, label, color) in enumerate(keys):
        prev_v = [prev_metrics.get(m, {}).get(k, 0) for m in MODELS_ACTIVE]
        cur_v = [cur_metrics.get(m, {}).get(k, 0) for m in MODELS_ACTIVE]
        ax.bar(x + (i - 1.5) * width, cur_v, width, color=color, alpha=0.85,
               label=f"{label} (current)" if lang != "zh" else f"{label}（本次）")
        ax.plot(x + (i - 1.5) * width, prev_v, marker="x", ls="", color=color,
                alpha=0.55, label=(f"{label} (previous)" if lang != "zh" else f"{label}（上次）") if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in MODELS_ACTIVE])
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8, ncol=4, loc="upper center")
    ax.set_title("重训/数据更新前后对比（×=上次，柱=本次）" if lang == "zh"
                 else "Before vs After Retraining/Data Update (x=previous, bar=current)")
    ax.set_ylabel("得分" if lang == "zh" else "Score")
    fig.tight_layout(); fig.savefig(out_path, dpi=dpi); plt.close(fig)


def make_html_report(out_dir, report, pngs, table_rows=None, compare_txt=None):
    """生成单文件 HTML 报告（内嵌 base64 图片，可打印/转 PDF）"""
    def b64(p):
        try:
            return base64.b64encode(Path(p).read_bytes()).decode()
        except Exception:
            return ""
    html = ["<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>",
            "<title>模型评估报告</title>",
            "<style>body{font-family:'Microsoft YaHei',sans-serif;margin:24px;color:#222;background:#fff}",
            "h1{font-size:22px;border-bottom:2px solid #1f4e79;padding-bottom:6px}",
            "h2{font-size:16px;color:#1f4e79;margin-top:22px}",
            "table{border-collapse:collapse;font-size:13px;margin:8px 0}",
            "th,td{border:1px solid #999;padding:5px 10px;text-align:center}",
            "th{background:#1f4e79;color:#fff}",
            "img{max-width:760px;border:1px solid #ddd;margin:6px 0}",
            ".note{color:#666;font-size:12px}</style></head><body>"]
    html.append(f"<h1>模型评估报告</h1>")
    html.append(f"<p class='note'>生成时间：{report.get('generated_at', '')}　数据源：{report.get('data_dir', '')}　样本数：{report.get('n_samples', '')}（跳过 {report.get('n_skipped', 0)}）</p>")
    dist = report.get("class_distribution", {})
    html.append(f"<p>类别分布：合格 {dist.get('good', 0)} 张，不合格 {dist.get('bad', 0)} 张</p>")
    if report.get("pipeline3_accuracy") is not None:
        html.append(f"<p>缺陷分级管线（三级）准确率：{report['pipeline3_accuracy'] * 100:.2f}%</p>")

    html.append("<h2>1. 四模型评估指标</h2><table><tr><th>模型</th><th>准确率</th><th>精确率</th><th>召回率</th><th>F1</th></tr>")
    for m in MODELS_ACTIVE:
        v = (report.get("models") or {}).get(m) or {}
        if not v:
            continue
        html.append(f"<tr><td><b>{m.upper()}</b></td><td>{v.get('accuracy', '-'):.2%}</td>"
                    f"<td>{v.get('precision', '-'):.2%}</td><td>{v.get('recall', '-'):.2%}</td>"
                    f"<td>{v.get('f1', '-'):.2%}</td></tr>")
    html.append("</table>")

    for name, path in pngs.items():
        data = b64(path)
        if data:
            html.append(f"<h2>{name}</h2><img src='data:image/png;base64,{data}'>")
    if table_rows:
        html.append("<h2>逐样本预测明细（前 20 行）</h2><table><tr><th>文件</th><th>真实</th><th>SVM</th><th>RF</th><th>DL</th><th>YOLO</th></tr>")
        for r in table_rows[:20]:
            html.append(f"<tr><td style='text-align:left;font-size:11px'>{r['file']}</td><td>{r['y_true']}</td>"
                        f"<td>{r.get('svm') or '-'}</td><td>{r.get('rf') or '-'}</td>"
                        f"<td>{r.get('dl') or '-'}</td><td>{r.get('yolo') or '-'}</td></tr>")
        html.append("</table>")
    if compare_txt:
        html.append(f"<h2>与上一次评估对比</h2><pre>{compare_txt}</pre>")
    html.append("</body></html>")
    (out_dir / "eval_report.html").write_text("\n".join(html), encoding="utf-8")


# ================= 模式 1：独立测试集 =================
def mode_test(args):
    out_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / "errors").mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.data, args.limit)
    rows, skipped = collect_predictions(samples)
    print(f"样本: {len(samples)}（跳过 {skipped}）")
    if not rows:
        raise SystemExit("没有可评估的样本")

    report = {"mode": "test", "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "data_dir": str(args.data), "n_samples": len(rows), "n_skipped": skipped,
              "class_distribution": {"good": sum(1 for r in rows if r["y_true"] == "good"),
                                     "bad": sum(1 for r in rows if r["y_true"] == "bad")},
              "models": {}}
    y_true = [r["y_true"] for r in rows]
    pngs = {}

    for m in MODELS_ACTIVE:
        y_pred = [r.get(m) or "unknown" for r in rows]
        valid = [i for i, v in enumerate(y_pred) if v in ("good", "bad")]
        if not valid:
            continue
        mtr = binary_metrics([y_true[i] for i in valid], [y_pred[i] for i in valid])
        cm_path = out_dir / f"cm_{m}.png"
        cm_title = f"{m.upper()} 混淆矩阵" if args.lang != "en" else f"{m.upper()} Confusion Matrix"
        mtr["confusion_matrix"] = make_confusion_matrix([y_true[i] for i in valid],
                                                        [y_pred[i] for i in valid],
                                                        cm_path, cm_title, args.lang, args.dpi)
        mtr["usable"] = len(valid)
        report["models"][m] = mtr
        pngs[f"{m.upper()} 混淆矩阵"] = cm_path
        print(f"  {m.upper():5s} acc={mtr['accuracy']:.4f} P={mtr['precision']:.4f} "
              f"R={mtr['recall']:.4f} F1={mtr['f1']:.4f}")

    y3_pred = [r.get("pipeline3") or "unknown" for r in rows]
    y3_true = [r["y3_true"] for r in rows]
    report["pipeline3_accuracy"] = round(sum(1 for a, b in zip(y3_true, y3_pred) if a == b) / len(rows), 4)

    ov = out_dir / "overview.png"
    make_overview(report["models"], ov, None, args.lang, args.dpi)
    pngs["四模型指标对比"] = ov

    roc = make_roc(rows, out_dir / "roc_curves.png", args.lang, args.dpi)
    if roc:
        pngs["ROC 曲线"] = roc

    # 对比上一次评估
    compare_txt = None
    prev_dir, prev = find_previous_test(out_dir)
    if prev:
        prev_models = prev.get("models") or {}
        cur_models = report["models"]
        cp = out_dir / "compare_previous.png"
        make_compare(prev_models, cur_models, cp, args.lang, args.dpi)
        pngs["重训/数据更新前后对比（×=上次，柱=本次）"] = cp
        lines = ["模型   上次acc → 本次acc    Δ", "-" * 40]
        for m in MODELS_ACTIVE:
            pv = (prev_models.get(m) or {}).get("accuracy")
            cv = (cur_models.get(m) or {}).get("accuracy")
            if pv is not None and cv is not None:
                lines.append(f"{m.upper():6s} {pv:.4f} → {cv:.4f}   {cv - pv:+.4f}")
        lines.append(f"上次数据: {prev.get('data_dir', '')}（{prev.get('n_samples', 0)} 张，{prev.get('generated_at', '')}）")
        compare_txt = "\n".join(lines)
        print("\n=== 与上次评估对比 ===")
        for l in lines:
            print(" ", l)

    # 明细 CSV
    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "y_true", "svm", "rf", "dl", "yolo", "pipeline3",
                    "svm_score", "rf_score", "dl_score", "yolo_score"])
        for r in rows:
            sc = r.get("scores") or {}
            w.writerow([r["file"], r["y_true"], r.get("svm", ""), r.get("rf", ""),
                        r.get("dl", ""), r.get("yolo", ""), r.get("pipeline3", ""),
                        round(sc.get("svm", 0) or 0, 3), round(sc.get("rf", 0) or 0, 3),
                        round(sc.get("dl", 0) or 0, 3), round(sc.get("yolo", 0) or 0, 3)])

    err_count = 0
    for r in rows:
        preds = [r.get(m) for m in MODELS_ACTIVE if r.get(m) in ("good", "bad")]
        if preds and r["y_true"] not in preds:
            src = config.ROOT_DIR / r["file"]
            if src.exists():
                try:
                    shutil.copy2(str(src), str(out_dir / "errors" / f"{Path(r['file']).stem}_err.jpg"))
                    err_count += 1
                except Exception:
                    pass
    report["error_samples"] = err_count
    report["compare"] = compare_txt

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_html_report(out_dir, report, pngs, rows, compare_txt)
    print(f"\n输出目录: {out_dir}")
    print(f"错误样本: {err_count} | 报告: eval_report.html（可打印/转PDF）")
    return out_dir


# ================= 模式 2：交叉验证 =================
def mode_cv(args):
    out_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.data, args.limit)
    y_all = [yb for _, yb in samples]
    n_good = sum(1 for v in y_all if v == "good")
    n_bad = sum(1 for v in y_all if v == "bad")
    if n_good < args.folds or n_bad < args.folds:
        print(f"[warn] 类别样本过少（good={n_good}, bad={n_bad}），结果仅供参考")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    per_model = {m: [] for m in MODELS_ACTIVE}
    fold_rows = []
    for fold, (_, test_idx) in enumerate(skf.split(samples, y_all), 1):
        rows, skipped = collect_predictions([samples[i] for i in test_idx])
        y_true = [r["y_true"] for r in rows]
        info = {"fold": fold, "n": len(rows),
                "good": sum(1 for v in y_true if v == "good"),
                "bad": sum(1 for v in y_true if v == "bad")}
        print(f"\n第 {fold} 折: {len(rows)} 张 (good={info['good']}, bad={info['bad']}, 跳过 {skipped})")
        for m in MODELS_ACTIVE:
            y_pred = [r.get(m) or "unknown" for r in rows]
            valid = [i for i, v in enumerate(y_pred) if v in ("good", "bad")]
            acc = accuracy_score([y_true[i] for i in valid], [y_pred[i] for i in valid]) if valid else None
            per_model[m].append(acc)
            info[f"{m}_acc"] = round(acc, 4) if acc is not None else None
            if acc is not None:
                print(f"    {m.upper():5s} acc={acc:.4f}")
        fold_rows.append(info)

    summary = {"mode": "cv", "folds": args.folds, "n_total": len(samples),
               "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    txt = ["交叉验证结果（当前模型 · 数据子集稳定性）", "=" * 40,
           f"总样本: {len(samples)} (good={n_good}, bad={n_bad})  折数: {args.folds}", ""]
    for m in MODELS_ACTIVE:
        vals = [v for v in per_model[m] if v is not None]
        if vals:
            mean, std = float(np.mean(vals)), float(np.std(vals))
            summary[m] = {"mean": round(mean, 4), "std": round(std, 4),
                          "per_fold": [round(v, 4) for v in vals]}
            txt.append(f"{m.upper():5s} acc = {mean:.4f} ± {std:.4f}")
        else:
            summary[m] = None
            txt.append(f"{m.upper():5s} 无法评估")
    make_cv_overview(summary, out_dir / "cv_overview.png", args.lang, args.dpi)
    (out_dir / "cv_summary.txt").write_text("\n".join(txt), encoding="utf-8")
    with open(out_dir / "cv_folds.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["fold", "n", "good", "bad"] + [f"{m}_acc" for m in MODELS_ACTIVE])
        for fr in fold_rows:
            w.writerow([fr["fold"], fr["n"], fr["good"], fr["bad"]] +
                       [fr.get(f"{m}_acc", "") for m in MODELS_ACTIVE])
    (out_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 交叉验证汇总 ===")
    for line in txt[2:]:
        print(" ", line)
    print(f"\n输出目录: {out_dir}（含 cv_overview.png 误差棒图）")
    return out_dir


# ================= 模式 3：一致性 =================
def mode_consistency(args):
    out_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.data, args.limit)
    rows, skipped = collect_predictions(samples)
    print(f"样本: {len(rows)}（跳过 {skipped}）")
    disagree = []
    agree_count = usable = 0
    pair_agree = {f"{a}-{b}": 0 for a in MODELS_ACTIVE for b in MODELS_ACTIVE if a < b}
    pair_total = {k: 0 for k in pair_agree}
    for r in rows:
        preds = {m: r.get(m) for m in MODELS_ACTIVE}
        pvals = [v for v in preds.values() if v in ("good", "bad")]
        if len(pvals) < 2:
            continue
        usable += 1
        if len(set(pvals)) == 1:
            agree_count += 1
        else:
            disagree.append({"file": r["file"], "y_true": r["y_true"], **preds})
        for k in pair_agree:
            a, b = k.split("-")
            if preds[a] in ("good", "bad") and preds[b] in ("good", "bad"):
                pair_total[k] += 1
                if preds[a] == preds[b]:
                    pair_agree[k] += 1
    full_ratio = agree_count / usable if usable else 0
    pair_summary = {k: round(pair_agree[k] / pair_total[k], 4) if pair_total[k] else None
                    for k in pair_agree}
    summary = {"mode": "consistency", "n_usable": usable, "n_skipped": skipped,
               "full_agree_ratio": round(full_ratio, 4), "pair_agree": pair_summary,
               "n_disagree": len(disagree),
               "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(out_dir / "consistency.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "y_true", "svm", "rf", "dl", "yolo"])
        for d in disagree:
            w.writerow([d["file"], d["y_true"], d["svm"], d["rf"], d["dl"], d["yolo"]])
    make_consistency_heatmap(pair_summary, out_dir / "consistency_heatmap.png", args.lang, args.dpi)
    (out_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 一致性分析 ===")
    print(f"可评估样本: {usable}  四模型完全一致率: {full_ratio:.4f}")
    for k, v in pair_summary.items():
        print(f"  {k:10s} 一致率: {v if v is not None else 'N/A'}")
    print(f"分歧样本: {len(disagree)}（见 consistency.csv + 热图 consistency_heatmap.png）")
    print(f"输出目录: {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="模型多维度评估工具")
    ap.add_argument("--mode", default="test", choices=["test", "cv", "consistency"])
    ap.add_argument("--data", default=str(DATASET_DEFAULT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--models", default=None,
                    help="comma-separated model subset, e.g. svm,rf "
                         "(default: all of svm,rf,dl,yolo)")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"],
                    help="图表语言：zh 中文（默认）/ en English")
    ap.add_argument("--dpi", type=int, default=150, help="图表分辨率（论文建议 300）")
    ap.add_argument("--paper", action="store_true",
                    help="论文模式：英文标注 + 300dpi，并复制图表到 output/evaluation/paper/")
    args = ap.parse_args()
    if args.paper:
        args.lang = "en"
        args.dpi = 300
    if args.models:
        want = [m.strip().lower() for m in args.models.split(",") if m.strip()]
        MODELS_ACTIVE[:] = [m for m in MODELS if m in want]
        if not MODELS_ACTIVE:
            raise SystemExit("no valid models in --models (choose from svm,rf,dl,yolo)")
        print("active models:", MODELS_ACTIVE)
    args.data = Path(args.data)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode == "test":
        out_dir = mode_test(args)
    elif args.mode == "cv":
        out_dir = mode_cv(args)
    else:
        out_dir = mode_consistency(args)
    if args.paper and out_dir is not None:
        paper_dir = OUT_ROOT / "paper"
        paper_dir.mkdir(parents=True, exist_ok=True)
        mapping = {"overview.png": "fig1_overview.png",
                   "roc_curves.png": "fig2_roc.png",
                   "compare_previous.png": "fig4_retrain_compare.png",
                   "cv_overview.png": "fig5_cv.png",
                   "consistency_heatmap.png": "fig6_consistency.png"}
        for src, dst in mapping.items():
            sp = out_dir / src
            if sp.exists():
                shutil.copy2(str(sp), str(paper_dir / dst))
        for m in MODELS_ACTIVE:
            sp = out_dir / f"cm_{m}.png"
            if sp.exists():
                shutil.copy2(str(sp), str(paper_dir / f"fig3_confusion_{m}.png"))
        print(f"论文图表目录: {paper_dir}（英文标注 · 300dpi）")


if __name__ == "__main__":
    main()
