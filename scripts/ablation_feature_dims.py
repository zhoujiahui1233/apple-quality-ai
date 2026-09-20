# -*- coding: utf-8 -*-
"""
消融实验 A：特征维度对 SVM / RF 分类准确率的影响（15 / 50 / 100 / 170 维）

用法:
  python scripts/ablation_feature_dims.py                  # 苹果，默认 15/50/100/170
  python scripts/ablation_feature_dims.py --category apple --dims "15,50,100,170"
  python scripts/ablation_feature_dims.py --test-ratio 0.3 --seed 42

原理:
  在收集的数据集(dataset_collected/{category}/{label})上提取 170 维特征，
  用"互信息 + 方差"对特征排序，取前 k 维分别训练 SVM/RF，
  在固定测试集(与训练完全隔离)上比较准确率，得到特征维度-准确率曲线。

输出:
  output/ablation_feature_dims.json   (论文可直接引用的结果)
  output/ablation_feature_dims.png    (若安装了 matplotlib，自动画曲线)

注意:
  当前真实收集数据量少时，结果仅供流程验证；按毕设计划扩充到每品类 200 张后重跑。
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import config

DATASET_DIR = config.ROOT_DIR / "dataset_collected"
OUTPUT_DIR = config.OUTPUT_DIR
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_DIMS = [15, 50, 100, 170]


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def label_classes(folder_name):
    low = folder_name.lower()
    if "不合格" in folder_name or "reject" in low:
        return 1
    if "合格" in folder_name or "good" in low or "premium" in low or "优质" in folder_name or "特级" in folder_name:
        return 0
    return 0


def iter_samples(category, limit=None):
    cat_dir = DATASET_DIR / category
    if not cat_dir.is_dir():
        return
    for label_dir in sorted(cat_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        cls = label_classes(label_dir.name)
        files = sorted(label_dir.glob("*"))
        if limit:
            files = files[:limit]
        for f in files:
            if f.suffix.lower() in IMG_EXTS:
                yield f, cls


def _safe_n_jobs():
    """容器/受限环境下避免 loky 物理核检测报错"""
    try:
        import os
        return max(1, min(int(os.cpu_count() or 1), 8))
    except Exception:
        return 1

def rank_features(X, y, seed):
    """互信息为主、方差为辅的特征排序；返回从优到劣的特征下标"""
    from sklearn.feature_selection import mutual_info_classif
    n_feat = X.shape[1]
    try:
        mi = mutual_info_classif(X, y, random_state=seed, n_neighbors=3)
    except Exception:
        mi = np.zeros(n_feat)
    var = np.var(X, axis=0)
    var_n = var / (var.max() + 1e-9)
    score = mi + 1e-3 * var_n
    return np.argsort(score)[::-1].tolist()


def build_dataset(category, limit):
    """返回 (X 170维, y 二分类) 与 (paths, labels) 明细"""
    from apple_defect_detector import extract_features_v2
    X, y, paths = [], [], []
    skipped = 0
    for f, cls in iter_samples(category, limit):
        img = imread_unicode(f)
        if img is None:
            skipped += 1
            continue
        feats = extract_features_v2(img)
        if feats is None or feats.shape[0] != 170:
            skipped += 1
            continue
        X.append(feats)
        y.append(cls)
        paths.append(str(f.relative_to(ROOT)))
    if not X:
        raise RuntimeError(
            f"没有可用的 {category} 数据，请在 dataset_collected/{category}/{{合格,不合格}}/ 放入图片")
    return np.array(X, dtype=np.float32), np.array(y), paths, skipped


def run_ablation(X, y, dims, test_ratio, seed):
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y)
    order = rank_features(Xtr, ytr, seed)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    results = {"svm": {}, "rf": {}}
    for k in dims:
        k = int(k)
        if k > X.shape[1]:
            k = X.shape[1]
        idx = np.array(sorted(order[:k]))
        Xtr_k, Xte_k = Xtr[:, idx], Xte[:, idx]

        scaler = StandardScaler().fit(Xtr_k)
        Xtr_s, Xte_s = scaler.transform(Xtr_k), scaler.transform(Xte_k)

        svm = SVC(kernel="rbf", C=50.0, gamma=0.01, class_weight="balanced",
                  random_state=seed)
        svm.fit(Xtr_s, ytr)
        results["svm"][str(k)] = round(float(np.mean(svm.predict(Xte_s) == yte)) * 100, 2)

        rf = RandomForestClassifier(n_estimators=500, max_depth=30,
                                    min_samples_leaf=2, random_state=42,
                                    n_jobs=_safe_n_jobs(), class_weight="balanced")
        rf.fit(Xtr_k, ytr)
        results["rf"][str(k)] = round(float(np.mean(rf.predict(Xte_k) == yte)) * 100, 2)

    return {
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "order": order[:20],
        "results": results,
    }


def try_plot(payload, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    dims = list(payload["dims"])
    for m in ("svm", "rf"):
        vals = [payload["results"][m][str(k)] for k in dims]
        plt.plot(dims, vals, marker="o", label=m.upper())
    plt.xlabel("Feature dimensions")
    plt.ylabel("Test accuracy (%)")
    plt.title(f"Feature dimension ablation ({payload['category']})")
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=150)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="消融实验 A：特征维度影响")
    parser.add_argument("--category", default="apple")
    parser.add_argument("--dims", default=",".join(map(str, DEFAULT_DIMS)))
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="每个标签最多读取的图片数")
    args = parser.parse_args()

    dims = [int(x) for x in args.dims.split(",") if x.strip()]
    t0 = time.time()
    X, y, paths, skipped = build_dataset(args.category, args.limit)
    print(f"[{args.category}] 有效样本 {len(y)} 张（跳过 {skipped}）")
    if len(y) < 12:
        print("提示: 样本过少，结果仅供流程验证；毕设数据扩充后请重跑。")
    if len(set(y.tolist())) < 2:
        raise SystemExit("错误: 标签只有一类，无法做二分类消融实验")

    payload = run_ablation(X, y, dims, args.test_ratio, args.seed)
    payload.update({
        "source": "ablation_feature_dims",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category": args.category,
        "dims": dims,
        "n": len(y),
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "top_features": payload.pop("order"),
    })
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_json = OUTPUT_DIR / "ablation_feature_dims.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    out_png = OUTPUT_DIR / "ablation_feature_dims.png"
    has_png = try_plot(payload, out_png)

    print(f"\n特征维度消融结果（测试集 {payload['n_test']} 张）:  ")
    header = "维度    " + "".join(f"{str(k):>10}" for k in dims)
    print(header)
    for m in ("svm", "rf"):
        row = f"{m.upper():<8}" + "".join(
            f"{payload['results'][m][str(k)]:>9.2f}%" for k in dims)
        print(row)
    print(f"\n耗时 {time.time() - t0:.1f}s")
    print(f"已保存: {out_json}" + ("  + 曲线图 " + str(out_png) if has_png else ""))


if __name__ == "__main__":
    main()