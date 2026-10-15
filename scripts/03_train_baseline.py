"""Day 2, step 2: train XGBoost models and prove that data fusion helps.

For each warning (glucose spike, hypoglycaemia) we test two real-world
deployment modes, each with and without the health record (EHR):

    CGM mode             - patient wears a CGM sensor + smartwatch + uses the app diary
    Smartwatch-only mode - no CGM (CGM sensors cost ~Rs 4,000-5,000 per 14 days in
                           India, so most patients cannot wear one all the time);
                           only smartwatch + app diary
    EHR only             - reference: what the health record alone can tell

Comparing "with EHR" against "without EHR" in each mode measures exactly
how much data fusion adds.

Evaluation uses 5-fold cross-validation split BY PATIENT: the model is always
tested on patients it has never seen, like a new patient walking into a clinic.

Run from the project root:
    python scripts/03_train_baseline.py
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
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glucotwin import config as C

OUT = C.ROOT / "results"
N_FOLDS = 5

# Colours (validated categorical palette, light mode)
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b4b2ab"
INK, INK2 = "#0b0b0b", "#52514e"


SETS_ORDER = [("CGM mode", "without EHR"), ("CGM mode", "with EHR"),
              ("Smartwatch-only mode", "without EHR"), ("Smartwatch-only mode", "with EHR")]


def feature_sets(groups):
    t, st = groups["time"], groups["static"]
    cgm_mode = groups["cgm"] + groups["watch"] + groups["diary"] + t
    watch_mode = groups["watch"] + groups["diary"] + t
    return {
        "EHR only": st + t,
        "CGM mode | without EHR": cgm_mode,
        "CGM mode | with EHR": cgm_mode + st,
        "Smartwatch-only mode | without EHR": watch_mode,
        "Smartwatch-only mode | with EHR": watch_mode + st,
    }


def clf_params(pos_rate):
    return dict(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.7, min_child_weight=10, reg_lambda=5, tree_method="hist",
                scale_pos_weight=min((1 - pos_rate) / max(pos_rate, 1e-6), 50),
                eval_metric="aucpr", n_jobs=-1, random_state=C.SEED)


def alarm_stats(y, p, groups_pid, sensitivity=0.80):
    """Pick the threshold that catches `sensitivity` of events, report the cost."""
    fpr, tpr, thr = roc_curve(y, p)
    i = np.searchsorted(tpr, sensitivity)
    th = thr[min(i, len(thr) - 1)]
    pred = p >= th
    tp = (pred & (y == 1)).sum()
    fp = (pred & (y == 0)).sum()
    n_days = len(np.unique(groups_pid)) * 13      # 13 evaluated days per patient
    return dict(threshold=float(th), sensitivity=float(tp / max(y.sum(), 1)),
                specificity=float(1 - fp / max((y == 0).sum(), 1)),
                ppv=float(tp / max(tp + fp, 1)),
                false_alert_checks_per_patient_day=float(fp / n_days))  # checks run every 15 min


def run_classification(df, label, eligible, sets):
    d = df[df[eligible] == 1].reset_index(drop=True)
    y = d[label].to_numpy()
    pid = d.patient_id.to_numpy()
    gkf = GroupKFold(n_splits=N_FOLDS)
    res, oof_store = {}, {}
    for name, cols in sets.items():
        oof = np.zeros(len(d))
        for tr, te in gkf.split(d, y, pid):
            m = xgb.XGBClassifier(**clf_params(y[tr].mean()))
            m.fit(d.loc[tr, cols], y[tr])
            oof[te] = m.predict_proba(d.loc[te, cols])[:, 1]
        fold_auc = [roc_auc_score(y[te], oof[te]) for _, te in gkf.split(d, y, pid)
                    if y[te].sum() > 0]
        res[name] = dict(auroc=float(roc_auc_score(y, oof)),
                         auroc_fold_std=float(np.std(fold_auc)),
                         auprc=float(average_precision_score(y, oof)),
                         prevalence=float(y.mean()), n_moments=int(len(y)),
                         n_events=int(y.sum()), **alarm_stats(y, oof, pid))
        oof_store[name] = oof
        print(f"  {name:36s} AUROC {res[name]['auroc']:.3f} (±{res[name]['auroc_fold_std']:.3f})"
              f"  AUPRC {res[name]['auprc']:.3f}  sens {res[name]['sensitivity']:.0%}"
              f" spec {res[name]['specificity']:.1%}")
    return res, oof_store, y


def run_regression(df, sets, horizons=(30, 60, 120)):
    """Predict the actual glucose value at +30, +60, +120 minutes."""
    out = {}
    for hz in horizons:
        tgt = f"glucose_plus_{hz}"
        d = df[np.isfinite(df[tgt])].reset_index(drop=True)
        y = d[tgt].to_numpy(); pid = d.patient_id.to_numpy()
        gkf = GroupKFold(n_splits=N_FOLDS)
        r = {"Persistence (no model)": dict(mae=float(mean_absolute_error(y, d.glucose_now)),
                                             rmse=float(np.sqrt(np.mean((y - d.glucose_now) ** 2))))}
        for name in [k for k in sets if k != "EHR only"]:
            cols = sets[name]
            oof = np.zeros(len(d))
            for tr, te in gkf.split(d, y, pid):
                m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                     n_jobs=-1, random_state=C.SEED)
                m.fit(d.loc[tr, cols], y[tr])
                oof[te] = m.predict(d.loc[te, cols])
            r[name] = dict(mae=float(mean_absolute_error(y, oof)),
                           rmse=float(np.sqrt(np.mean((y - oof) ** 2))))
        out[f"+{hz} min"] = r
        print(f"  +{hz:3d} min")
        for k, v in r.items():
            print(f"     {k:38s} MAE {v['mae']:5.1f}  RMSE {v['rmse']:5.1f}")
    return out


def plot_ablation(results, fname):
    """Grouped bars: the gain from adding the EHR, in each deployment mode."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, (task, res) in zip(axes, results.items()):
        modes = ["CGM mode", "Smartwatch-only mode"]
        x = np.arange(len(modes))
        w = 0.34
        for j, (variant, color) in enumerate([("without EHR", GREY), ("with EHR", BLUE)]):
            vals = [res[f"{m} | {variant}"]["auprc"] for m in modes]
            bars = ax.bar(x + (j - 0.5) * (w + 0.02), vals, w, color=color,
                          label=f"Wearables {variant}" if variant == "without EHR" else "Fused: wearables + EHR")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                        ha="center", fontsize=10, color=INK)
        ehr = res["EHR only"]["auprc"]
        ax.axhline(ehr, color=ORANGE, lw=1.5, ls="--")
        ax.text(1.62, ehr, f"EHR only\n{ehr:.2f}", va="center", fontsize=8.5, color=INK2)
        prev = res["EHR only"]["prevalence"]
        ax.set_xticks(x, modes, color=INK)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.6, 1.9)
        ax.set_ylabel("Alert quality (AUPRC)", color=INK2)
        ax.set_title(f"{task}  (event rate {prev:.1%})", loc="left", fontsize=12, color=INK)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9.5, ncol=2, loc="upper left",
               bbox_to_anchor=(0.01, 0.93))
    fig.suptitle("The health record matters most when there is no CGM  "
                 "(tested on unseen patients, 5-fold CV)", x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def plot_roc(oofs, ys, fname):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    style = {"EHR only": (ORANGE, "--"),
             "CGM mode | without EHR": (GREY, "-"), "CGM mode | with EHR": (BLUE, "-"),
             "Smartwatch-only mode | without EHR": (GREY, ":"),
             "Smartwatch-only mode | with EHR": (AQUA, "-")}
    for ax, task in zip(axes, oofs):
        y = ys[task]
        for name, p in oofs[task].items():
            fpr, tpr, _ = roc_curve(y, p)
            c, ls = style[name]
            ax.plot(fpr, tpr, color=c, ls=ls, lw=2, label=f"{name} ({roc_auc_score(y, p):.3f})")
        ax.set_xlim(0, 0.3)
        ax.set_title(f"{task} (zoomed: false-alarm rate 0-30%)", loc="left", color=INK, fontsize=11)
        ax.set_xlabel("False alarm rate", color=INK2); ax.set_ylabel("Events caught", color=INK2)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(C.DATA_DIR / "processed" / "fused_dataset.csv.gz", parse_dates=["timestamp"])
    groups = json.loads((C.DATA_DIR / "processed" / "feature_groups.json").read_text())
    sets = feature_sets(groups)

    print("\n[1/3] Glucose SPIKE (>180 mg/dL) within 2 hours")
    spike_res, spike_oof, y_sp = run_classification(df, "spike_2h", "spike_eligible", sets)
    print("\n[2/3] HYPOGLYCAEMIA (<70 mg/dL) within 2 hours")
    hypo_res, hypo_oof, y_hy = run_classification(df, "hypo_2h", "hypo_eligible", sets)
    print("\n[3/3] Glucose value forecast (mean absolute error, mg/dL)")
    reg_res = run_regression(df, sets)

    results = {"Glucose spike in 2 h": spike_res, "Hypoglycaemia in 2 h": hypo_res}
    plot_ablation(results, "fusion_ablation.png")
    plot_roc({"Glucose spike in 2 h": spike_oof, "Hypoglycaemia in 2 h": hypo_oof},
             {"Glucose spike in 2 h": y_sp, "Hypoglycaemia in 2 h": y_hy}, "roc_curves.png")
    (OUT / "baseline_results.json").write_text(json.dumps(
        {"classification": results, "forecast": reg_res}, indent=2))
    print(f"\nSaved results/ in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
