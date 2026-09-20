# -*- coding: utf-8 -*-
"""
fusion_service.py - Multimodal sensory fusion service.

Two-level structure:
  Level 1: single-modality models (visual 170-d / odor 60-d / phys numeric, one RF each)
  Level 2: fusion models (concat feature-level + late decision-level)

Data sources (multimodal_data/):
  visual/*.jpg                 -> image -> 170-d visual features (extract_features_v2)
  odor/raw/*.csv               -> e-nose -> 60-d odor features (predict_freshness.extract_single)
  touch/touch.csv              -> phys/touch numeric columns
  manifest.csv (optional)      -> label; fallback infer from sample_id

Outputs:
  models/sensory_models/{visual,odor,phys}_rf.pkl + fusion_concat.pkl
  report output/sensory_train_report.json
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "enose"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

MODEL_DIR = ROOT / "models" / "sensory_models"

LABEL_KEYWORDS = [
    ("rotten", "rotten"), ("fresh", "fresh"), ("mid", "mid"),
    ("bad", "bad"), ("good", "good"),
    ("\u4e0d\u5408\u683c", "bad"), ("\u5408\u683c", "good"),
    ("\u7279\u7ea7", "premium"), ("\u4e00\u7ea7", "grade1"), ("\u4e8c\u7ea7", "grade2"),
]


def infer_label(sample_id, manifest_map=None):
    """Label priority: manifest.csv > sample_id keywords."""
    if manifest_map and sample_id in manifest_map and manifest_map.get(sample_id):
        return str(manifest_map[sample_id]).strip()
    s = str(sample_id).lower()
    for kw, lab in LABEL_KEYWORDS:
        if kw in s:
            return lab
    return None


class FusionService:
    def __init__(self, root=None):
        self.root = Path(root) if root else ROOT
        self.md = self.root / "multimodal_data"

    # ---------------- data collection ----------------
    def collect(self):
        """Aggregate 4-modality samples. Returns (samples, phys_df, odor_df, manifest)."""
        manifest = {}
        mf = self.md / "manifest.csv"
        if mf.exists():
            m = pd.read_csv(mf, encoding="utf-8")
            manifest = dict(zip(m["sample_id"].astype(str), m.get("label", "").fillna("")))

        phys_df = None
        tcsv = self.md / "touch" / "touch.csv"
        if tcsv.exists():
            phys_df = pd.read_csv(tcsv, encoding="utf-8")
            phys_df["sample_id"] = phys_df["sample_id"].astype(str)

        odor_df = None
        ofeat = self.md / "odor" / "features.csv"
        if ofeat.exists():
            odor_df = pd.read_csv(ofeat, encoding="utf-8")
            odor_df["sample_id"] = odor_df["source"].astype(str).str.replace(r"\.[a-zA-Z]+$", "", regex=True)

        vis_cache = self._load_visual_cache()
        vids = set(vis_cache.keys())
        vdir = self.md / "visual"
        if vdir.exists():
            vids |= {p.stem for p in vdir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}

        oids = set()
        odor_dir = self.md / "odor" / "raw"
        if odor_dir.exists():
            oids = {p.stem for p in odor_dir.glob("*.csv")}
        if odor_df is not None:
            oids |= set(odor_df["sample_id"])

        pids = set(phys_df["sample_id"]) if phys_df is not None else set()
        all_ids = vids | oids | pids | set(manifest.keys())
        samples = {}
        for sid in sorted(all_ids):
            rec = {"label": infer_label(sid, manifest)}
            if sid in vids:
                rec["visual"] = vis_cache.get(sid)
                rec["has_visual"] = True
            if sid in oids:
                rec["has_odor"] = True
            if sid in pids:
                rec["has_phys"] = True
            samples[sid] = rec
        return samples, phys_df, odor_df, manifest

    def _cache_path(self):
        return self.root / "multimodal_data" / "visual" / "features.csv"

    def _load_visual_cache(self):
        if not self._cache_path().exists():
            return {}
        try:
            df = pd.read_csv(self._cache_path(), encoding="utf-8")
            return {str(r["sample_id"]): np.array([float(x) for x in r.drop("sample_id").values], dtype=np.float32)
                    for _, r in df.iterrows()}
        except Exception:
            return {}

    def _save_visual_cache(self, cache):
        if not cache:
            return
        cp = self._cache_path()
        cp.parent.mkdir(parents=True, exist_ok=True)
        rows = [[sid] + [float(x) for x in vec] for sid, vec in sorted(cache.items())]
        cols = ["sample_id"] + [f"f{i}" for i in range(len(next(iter(cache.values()))))]
        pd.DataFrame(rows, columns=cols).to_csv(cp, index=False, encoding="utf-8-sig")

    def _extract_visual(self, sid, cache):
        if sid in cache and cache[sid] is not None:
            return cache[sid]
        import cv2
        import numpy as _np
        import apple_defect_detector as D
        for ext in (".jpg", ".jpeg", ".png"):
            p = self.md / "visual" / f"{sid}{ext}"
            if p.exists():
                # imdecode 支持含中文的路径
                img = cv2.imdecode(_np.fromfile(str(p), dtype=_np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    return None
                vec = D.extract_features_v2(img)
                if vec is None or len(vec) == 0:
                    return None
                cache[sid] = np.asarray(vec, dtype=np.float32)
                return cache[sid]
        return None

    def _odor_vec(self, sid, odor_df=None):
        if odor_df is not None and sid in set(odor_df["sample_id"]):
            row = odor_df[odor_df["sample_id"] == sid].iloc[0]
            cols = [c for c in odor_df.columns if c not in ("sample_id", "source", "label")]
            return np.array([float(x) if pd.notna(x) else 0.0 for x in row[cols].values], dtype=np.float32)
        csvp = self.md / "odor" / "raw" / f"{sid}.csv"
        if csvp.exists():
            try:
                import predict_freshness as PF
                return PF.extract_single(str(csvp))
            except Exception:
                return None
        return None

    def _phys_vec(self, sid, phys_df=None, phys_values=None):
        if phys_values is not None:
            return np.array([float(v) if v is not None else 0.0 for v in phys_values], dtype=np.float32)
        if phys_df is not None:
            row = phys_df[phys_df["sample_id"] == sid]
            if len(row):
                return np.array([float(x) if pd.notna(x) else 0.0
                                 for x in row.iloc[0].drop("sample_id").values], dtype=np.float32)
        tcsv = self.md / "touch" / "touch.csv"
        if tcsv.exists():
            df = pd.read_csv(tcsv, encoding="utf-8")
            row = df[df["sample_id"].astype(str) == sid]
            if len(row):
                return np.array([float(x) if pd.notna(x) else 0.0
                                 for x in row.iloc[0].drop("sample_id").values], dtype=np.float32)
        return None

    # ---------------- training ----------------
    def train(self, cv=5):
        """Train single-modality + fusion models, save and return report."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import joblib

        samples, phys_df, odor_df, manifest = self.collect()

        # extract visual features (with cache)
        cache = self._load_visual_cache()
        changed = False
        for sid, rec in samples.items():
            if rec.get("has_visual") and rec.get("visual") is None:
                vec = self._extract_visual(sid, cache)
                if vec is not None:
                    rec["visual"] = vec
                    changed = True
        if changed:
            self._save_visual_cache(cache)

        report = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  "cv": cv, "model_dir": str(MODEL_DIR), "n_labels": 0, "labels": []}
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        def labeled(mat_selector):
            """Return (X, y, ids) for one modality among labeled samples."""
            Xs, ys, ids = [], [], []
            for sid, rec in samples.items():
                if not rec.get("label"):
                    continue
                vec = mat_selector(sid, rec)
                if vec is not None:
                    Xs.append(vec)
                    ys.append(str(rec["label"]))
                    ids.append(sid)
            return Xs, ys, ids

        def sel_visual(sid, rec):
            return rec.get("visual")

        def sel_odor(sid, rec):
            return self._odor_vec(sid, odor_df)

        def sel_phys(sid, rec):
            return self._phys_vec(sid, phys_df)

        all_labels = set()
        for sid, rec in samples.items():
            if rec.get("label"):
                all_labels.add(str(rec["label"]))
        report["n_labels"] = len(all_labels)
        report["labels"] = sorted(all_labels)

        for mat, sel in (("visual", sel_visual), ("odor", sel_odor), ("phys", sel_phys)):
            Xs, ys, ids = labeled(sel)
            if len(Xs) < 8 or len(set(ys)) < 2:
                report[mat] = {"n": len(Xs), "note": "too few samples or single class"}
                print(f"[{mat}] skipped, n={len(Xs)}")
                continue
            X = np.array(Xs, dtype=np.float32)
            y = np.array(ys)
            clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
            k = min(cv, len(set(y)))
            scores = cross_val_score(clf, X, y, cv=k, scoring="accuracy")
            cv_pred = cross_val_predict(clf, X, y, cv=k)
            clf.fit(X, y)
            joblib.dump({"model": clf, "feat_len": int(X.shape[1]), "classes": sorted(set(y)),
                         "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         "n_samples": int(len(X)), "acc": float(scores.mean())},
                        MODEL_DIR / f"{mat}_rf.pkl")
            rep = classification_report(y, cv_pred, output_dict=True, zero_division=0)
            per_class = {str(k): {"precision": round(float(v["precision"]), 3),
                                  "recall": round(float(v["recall"]), 3),
                                  "f1": round(float(v["f1-score"]), 3),
                                  "support": int(v["support"])}
                         for k, v in rep.items() if k not in ("accuracy", "macro avg", "weighted avg")}
            conf = [[int(v) for v in row] for row in confusion_matrix(y, cv_pred, labels=sorted(set(y)))]
            report[mat] = {"n": int(len(X)), "acc": round(float(scores.mean()) * 100, 2),
                           "std": round(float(scores.std()) * 100, 2),
                           "classes": sorted(set(y)), "per_class": per_class, "confusion": conf}
            print(f"[{mat}] n={len(X)} acc={scores.mean()*100:.2f}% (+/-{scores.std()*100:.2f})")

        # fusion on aligned samples (all 3 modalities present + labeled)
        align_ids = sorted([sid for sid, rec in samples.items()
                            if rec.get("visual") is not None and rec.get("has_odor") and rec.get("has_phys") and rec.get("label")])
        if len(align_ids) >= 8 and len(set(str(samples[s]["label"]) for s in align_ids)) >= 2:
            phys_cols = [c for c in phys_df.columns if c != "sample_id"]
            Xc, yc = [], []
            for sid in align_ids:
                v = samples[sid]["visual"]
                o = self._odor_vec(sid, odor_df)
                p = self._phys_vec(sid, phys_df)
                if v is None or o is None or p is None:
                    continue
                Xc.append(np.concatenate([v, o, p]))
                yc.append(str(samples[sid]["label"]))
            Xc = np.array(Xc, dtype=np.float32)
            yc = np.array(yc)
            k = min(cv, len(set(yc)))
            clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
            scores = cross_val_score(clf, Xc, yc, cv=k, scoring="accuracy")
            cv_pred_c = cross_val_predict(clf, Xc, yc, cv=k)
            clf.fit(Xc, yc)
            visual_dim = int(len(samples[align_ids[0]]["visual"]))
            odor_dim = int(len(self._odor_vec(align_ids[0], odor_df)))
            joblib.dump({"model": clf, "feat_len": int(Xc.shape[1]), "classes": sorted(set(yc)),
                         "visual_dim": visual_dim, "odor_dim": odor_dim,
                         "phys_cols": phys_cols,
                         "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         "n_samples": int(len(Xc)), "acc": float(scores.mean())},
                        MODEL_DIR / "fusion_concat.pkl")
            rep_c = classification_report(yc, cv_pred_c, output_dict=True, zero_division=0)
            per_class_c = {str(k): {"precision": round(float(v["precision"]), 3),
                                    "recall": round(float(v["recall"]), 3),
                                    "f1": round(float(v["f1-score"]), 3),
                                    "support": int(v["support"])}
                           for k, v in rep_c.items() if k not in ("accuracy", "macro avg", "weighted avg")}
            conf_c = [[int(v) for v in row] for row in confusion_matrix(yc, cv_pred_c, labels=sorted(set(yc)))]
            report["fusion"] = {"concat": {"n": int(len(Xc)), "acc": round(float(scores.mean()) * 100, 2),
                                           "std": round(float(scores.std()) * 100, 2),
                                           "classes": sorted(set(yc)), "per_class": per_class_c,
                                           "confusion": conf_c}}
            print(f"[fusion_concat] n={len(Xc)} acc={scores.mean()*100:.2f}%")

            # late fusion: per-modality models on the same KFold, probability average
            classes = sorted(set(yc))
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            accs_late = []
            Xmats = {}
            for mat, sel in (("visual", sel_visual), ("odor", sel_odor), ("phys", sel_phys)):
                Xmats[mat] = np.array([self._pick(sid, mat, samples, odor_df, phys_df) for sid in align_ids], dtype=np.float32)
            for tr, te in skf.split(Xc, yc):
                proba_sum = np.zeros((len(te), len(classes)))
                for mat in Xmats:
                    Xm = Xmats[mat]
                    m = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
                    m.fit(Xm[tr], yc[tr])
                    p = m.predict_proba(Xm[te])
                    for ci, c in enumerate(classes):
                        if c in m.classes_:
                            proba_sum[:, ci] += p[:, list(m.classes_).index(c)]
                pred = np.array([classes[np.argmax(r)] for r in proba_sum])
                accs_late.append(accuracy_score(yc[te], pred))
            report["fusion"]["late"] = {"n": int(len(Xc)), "acc": round(float(np.mean(accs_late)) * 100, 2),
                                        "std": round(float(np.std(accs_late)) * 100, 2)}
            print(f"[fusion_late] n={len(Xc)} acc={np.mean(accs_late)*100:.2f}%")
        else:
            report["fusion"] = {"note": "aligned samples < 8 (need visual+odor+phys together)"}
            print("[fusion] skipped, aligned samples < 8")

        out = self.root / "output" / "sensory_train_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def _pick(self, sid, mat, samples, odor_df, phys_df):
        if mat == "visual":
            return samples[sid]["visual"]
        if mat == "odor":
            return self._odor_vec(sid, odor_df)
        return self._phys_vec(sid, phys_df)

    # ---------------- prediction ----------------
    def predict(self, sample_id, phys_values=None):
        """Predict grade for one sample: per-modality + fusion."""
        import joblib
        sid = str(sample_id)
        out = {"sample_id": sid, "modalities": {}, "fusion": None, "ok": True}
        cache = self._load_visual_cache()
        mods = {}
        for mat in ("visual", "odor", "phys"):
            pkl = MODEL_DIR / f"{mat}_rf.pkl"
            if not pkl.exists():
                out["modalities"][mat] = {"available": False, "note": "model not trained"}
                continue
            vec = None
            if mat == "visual":
                vec = self._extract_visual(sid, cache)
            elif mat == "odor":
                vec = self._odor_vec(sid)
            elif mat == "phys":
                vec = self._phys_vec(sid, phys_values=phys_values)
            if vec is None:
                out["modalities"][mat] = {"available": False, "note": "data missing"}
                continue
            bundle = joblib.load(pkl)
            clf = bundle["model"]
            if len(vec) != bundle["feat_len"]:
                out["modalities"][mat] = {"available": False, "note": "feature length mismatch"}
                continue
            proba = clf.predict_proba(vec.reshape(1, -1))[0]
            pred = clf.classes_[int(np.argmax(proba))]
            out["modalities"][mat] = {"available": True, "prediction": str(pred),
                                      "confidence": round(float(np.max(proba)), 4)}
            mods[mat] = vec

        concat_pkl = MODEL_DIR / "fusion_concat.pkl"
        if concat_pkl.exists() and len(mods) >= 2:
            bundle = joblib.load(concat_pkl)
            clf = bundle["model"]
            vecs = []
            for mat in ("visual", "odor", "phys"):
                if mat in mods:
                    vecs.append(mods[mat])
                else:
                    dim = {"visual": bundle.get("visual_dim", 170), "odor": bundle.get("odor_dim", 60)}.get(mat, 0)
                    vecs.append(np.zeros(dim, dtype=np.float32))
            X = np.concatenate(vecs).reshape(1, -1)
            if X.shape[1] == bundle["feat_len"]:
                proba = clf.predict_proba(X)[0]
                pred = clf.classes_[int(np.argmax(proba))]
                out["fusion"] = {"type": "concat", "prediction": str(pred),
                                 "confidence": round(float(np.max(proba)), 4)}
        return out


if __name__ == "__main__":
    svc = FusionService()
    report = svc.train()
    print(json.dumps(report, ensure_ascii=False, indent=2))
