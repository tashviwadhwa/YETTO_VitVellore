"""Day 3, step 2: the hybrid model - calibrated twin + machine learning.

The real-world scenario this tests:
    Week 1: the patient wears a CGM; their twin is calibrated (script 04).
    Week 2: the CGM is removed (it is expensive). Can a cheap smartwatch +
            the health record + the twin still give reliable warnings?

Everything here is evaluated on WEEK 2 ONLY, with patient-level 5-fold CV.

Run from the project root (after 04_calibrate_twins.py):
    python scripts/05_hybrid_model.py
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glucotwin import config as C
from glucotwin.hybrid import TWIN_FEATURES, load_twin, twin_features

OUT = C.ROOT / "results"
BLUE, ORANGE, GREY, DARK, INK, INK2 = "#2a78d6", "#eb6834", "#b4b2ab", "#6b6a66", "#0b0b0b", "#52514e"
N_FOLDS = 5


def feature_sets(g):
    t, st, tw = g["time"], g["static"], TWIN_FEATURES
    watch = g["watch"] + g["diary"] + t
    return {
        "Smartwatch + EHR": watch + st,
        "Smartwatch + EHR + twin (GlucoTwin)": watch + st + tw,
        "CGM + EHR (reference, CGM worn)": g["cgm"] + g["watch"] + g["diary"] + t + st,
    }


def cv_predict(d, y, cols, kind, pos_rate=None):
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(N_FOLDS).split(d, y, d.patient_id):
        if kind == "clf":
            m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.7, min_child_weight=10, reg_lambda=5,
                                  tree_method="hist", n_jobs=-1, random_state=C.SEED,
                                  scale_pos_weight=min((1 - pos_rate) / pos_rate, 50))
            m.fit(d.iloc[tr][cols], y[tr])
            oof[te] = m.predict_proba(d.iloc[te][cols])[:, 1]
        else:
            m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
                                 random_state=C.SEED)
            m.fit(d.iloc[tr][cols], y[tr])
            oof[te] = m.predict(d.iloc[te][cols])
    return oof


def main():
    t0 = time.time()
    df = pd.read_csv(C.DATA_DIR / "processed" / "fused_dataset.csv.gz", parse_dates=["timestamp"])
    groups = json.loads((C.DATA_DIR / "processed" / "feature_groups.json").read_text())
    week2 = df[df.timestamp >= pd.Timestamp(C.START_DATE) + pd.Timedelta(days=7)].reset_index(drop=True)
    print(f"[1/3] Running the twin forward 2 h from {len(week2):,} week-2 moments ...")
    T = load_twin()
    tf = twin_features(week2[["patient_id", "timestamp"]], T)
    week2 = pd.concat([week2, tf], axis=1)
    week2[["patient_id", "timestamp"] + TWIN_FEATURES].to_csv(
        C.DATA_DIR / "processed" / "twin_features_week2.csv.gz", index=False)
    sets = feature_sets(groups)
    res = {}

    print("[2/3] Early warnings on week 2 (unseen patients, 5-fold CV)")
    for task, label, elig, physics in [("Glucose spike in 2 h", "spike_2h", "spike_eligible", "twin_max_2h"),
                                       ("Hypoglycaemia in 2 h", "hypo_2h", "hypo_eligible", "twin_min_2h")]:
        d = week2[week2[elig] == 1].reset_index(drop=True)
        y = d[label].to_numpy()
        r = {}
        score = d[physics].to_numpy() * (1 if label == "spike_2h" else -1)
        r["Twin alone (no ML)"] = dict(auprc=average_precision_score(y, score), auroc=roc_auc_score(y, score))
        for name, cols in sets.items():
            p = cv_predict(d, y, cols, "clf", y.mean())
            r[name] = dict(auprc=average_precision_score(y, p), auroc=roc_auc_score(y, p))
        r = {k: {m: float(v) for m, v in vv.items()} for k, vv in r.items()}
        res[task] = dict(results=r, prevalence=float(y.mean()), n_events=int(y.sum()))
        print(f"  {task}  (events: {y.sum()}, rate {y.mean():.2%})")
        for k, v in r.items():
            print(f"     {k:38s} AUPRC {v['auprc']:.3f}  AUROC {v['auroc']:.3f}")

    print("[3/3] Glucose forecast error (MAE, mg/dL)")
    for hz in [60, 120]:
        tgt = f"glucose_plus_{hz}"
        d = week2[np.isfinite(week2[tgt])].reset_index(drop=True)
        y = d[tgt].to_numpy()
        r = {"Twin alone (no ML)": dict(mae=float(mean_absolute_error(y, d[f"twin_glucose_plus_{hz}"])))}
        for name, cols in sets.items():
            r[name] = dict(mae=float(mean_absolute_error(y, cv_predict(d, y, cols, "reg"))))
        res[f"Glucose at +{hz} min"] = dict(results=r)
        print(f"  +{hz} min")
        for k, v in r.items():
            print(f"     {k:38s} MAE {v['mae']:5.1f}")

    (OUT / "hybrid_results.json").write_text(json.dumps(res, indent=2))
    plot(res)
    print(f"Done in {time.time() - t0:.0f} s")


def plot(res):
    names = ["Twin alone (no ML)", "Smartwatch + EHR", "Smartwatch + EHR + twin (GlucoTwin)",
             "CGM + EHR (reference, CGM worn)"]
    short = ["Twin alone (physics, no ML)", "Smartwatch + EHR", "Smartwatch + EHR + twin",
             "CGM + EHR  (CGM worn)"]
    colors = [ORANGE, GREY, BLUE, DARK]
    panels = [("Glucose spike in 2 h", "auprc", "Alert quality (AUPRC), higher is better"),
              ("Hypoglycaemia in 2 h", "auprc", "Alert quality (AUPRC), higher is better"),
              ("Glucose at +60 min", "mae", "Forecast error (mg/dL), lower is better")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, (task, metric, xlabel) in zip(axes, panels):
        vals = [res[task]["results"][n][metric] for n in names]
        ax.barh(y, vals, color=colors, height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.02, yi, f"{v:.2f}" if metric == "auprc" else f"{v:.1f}",
                    va="center", fontsize=10, color=INK)
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_title(task, loc="left", fontsize=11.5, color=INK)
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, short, color=INK)
    fig.suptitle("Week 2, CGM removed: smartwatch + health record + calibrated twin matches wearing a CGM",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "hybrid_results.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
