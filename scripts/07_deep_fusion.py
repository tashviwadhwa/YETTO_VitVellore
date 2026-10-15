"""Day 3, step 4: deep fusion network vs XGBoost (and their ensemble).

Same held-out patients for every model (25% of patients never seen in training),
CGM-worn mode.

Run from the project root:
    python scripts/07_deep_fusion.py        (~10-20 min on a laptop CPU)
"""
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glucotwin import config as C
from glucotwin import deep
from glucotwin.features import STATIC_FEATURES, load_arrays

OUT = C.ROOT / "results"
MODELS = C.ROOT / "models"
GREY, BLUE, AQUA, INK, INK2 = "#b4b2ab", "#2a78d6", "#1baf7a", "#0b0b0b", "#52514e"


def xgb_fit_predict(dtr, dte, cols, target, kind, eligible=None):
    if eligible:
        dtr = dtr[dtr[eligible] == 1]
    if kind == "clf":
        pos = dtr[target].mean()
        m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.7, min_child_weight=10, reg_lambda=5,
                              tree_method="hist", n_jobs=-1, random_state=C.SEED,
                              scale_pos_weight=min((1 - pos) / pos, 50))
        m.fit(dtr[cols], dtr[target])
        return m.predict_proba(dte[cols])[:, 1]
    dtr = dtr[np.isfinite(dtr[target])]
    m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=C.SEED)
    m.fit(dtr[cols], dtr[target])
    return m.predict(dte[cols])


def evaluate(dte, p_spike, p_hypo, p_reg):
    out = {}
    e = dte.spike_eligible.to_numpy() == 1
    out["Spike AUPRC"] = average_precision_score(dte.spike_2h[e], p_spike[e])
    out["Spike AUROC"] = roc_auc_score(dte.spike_2h[e], p_spike[e])
    e = dte.hypo_eligible.to_numpy() == 1
    out["Hypo AUPRC"] = average_precision_score(dte.hypo_2h[e], p_hypo[e])
    out["Hypo AUROC"] = roc_auc_score(dte.hypo_2h[e], p_hypo[e])
    for j, tgt in enumerate(deep.TARGETS_REG):
        f = np.isfinite(dte[tgt].to_numpy())
        out[f"MAE +{tgt.split('_')[-1]} min"] = mean_absolute_error(dte[tgt][f], p_reg[f, j])
    return {k: float(v) for k, v in out.items()}


def main():
    t0 = time.time()
    torch.set_num_threads(os.cpu_count() or 2)
    MODELS.mkdir(exist_ok=True)
    df = pd.read_csv(C.DATA_DIR / "processed" / "fused_dataset.csv.gz", parse_dates=["timestamp"])
    g = json.loads((C.DATA_DIR / "processed" / "feature_groups.json").read_text())
    cols = g["cgm"] + g["watch"] + g["diary"] + g["time"] + g["static"]

    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=C.SEED)
                  .split(df, groups=df.patient_id))
    dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    print(f"Train: {dtr.patient_id.nunique()} patients, {len(dtr):,} moments | "
          f"Test: {dte.patient_id.nunique()} unseen patients, {len(dte):,} moments")

    print("[1/3] Building 3-hour sensor sequences ...")
    A = load_arrays()
    seq_tr, seq_te = deep.build_sequences(A, dtr), deep.build_sequences(A, dte)
    sc = deep.StaticScaler().fit(dtr[STATIC_FEATURES].to_numpy(float))
    st_tr, st_te = sc.transform(dtr[STATIC_FEATURES].to_numpy(float)), sc.transform(dte[STATIC_FEATURES].to_numpy(float))
    tm_tr = dtr[g["time"]].to_numpy(np.float32); tm_te = dte[g["time"]].to_numpy(np.float32)

    print("[2/3] Training the deep fusion network (GRU + MLP, multi-task) ...")
    net = deep.train_net(seq_tr, st_tr, tm_tr, deep.make_targets(dtr), epochs=8)
    pc, pr = deep.predict(net, seq_te, st_te, tm_te)
    torch.save(dict(state=net.state_dict(), mu=sc.mu, sd=sc.sd), MODELS / "deep_fusion.pt")

    print("[3/3] Training XGBoost on the same patients for comparison ...")
    xs = xgb_fit_predict(dtr, dte, cols, "spike_2h", "clf", "spike_eligible")
    xh = xgb_fit_predict(dtr, dte, cols, "hypo_2h", "clf", "hypo_eligible")
    xr = np.column_stack([xgb_fit_predict(dtr, dte, cols, t, "reg") for t in deep.TARGETS_REG])

    res = {"XGBoost": evaluate(dte, xs, xh, xr),
           "Deep fusion net": evaluate(dte, pc[:, 0], pc[:, 1], pr),
           "Ensemble (average)": evaluate(dte, (xs + pc[:, 0]) / 2, (xh + pc[:, 1]) / 2, (xr + pr) / 2)}
    print()
    print(pd.DataFrame(res).round(3).to_string())
    (OUT / "deep_results.json").write_text(json.dumps(res, indent=2))
    plot(res)
    print(f"\nDone in {time.time() - t0:.0f} s")


def plot(res):
    names = list(res)
    colors = [GREY, AQUA, BLUE]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, metrics, ylabel in [(ax[0], ["Spike AUPRC", "Hypo AUPRC"], "AUPRC (higher is better)"),
                               (ax[1], ["MAE +30 min", "MAE +60 min", "MAE +120 min"],
                                "Forecast error, mg/dL (lower is better)")]:
        x = np.arange(len(metrics)); w = 0.26
        for j, (n, c) in enumerate(zip(names, colors)):
            v = [res[n][m] for m in metrics]
            bars = a.bar(x + (j - 1) * (w + 0.015), v, w, color=c, label=n)
            for b, val in zip(bars, v):
                a.text(b.get_x() + b.get_width() / 2, val, f"{val:.2f}" if "AUPRC" in metrics[0] else f"{val:.1f}",
                       ha="center", va="bottom", fontsize=8.5, color=INK)
        a.set_xticks(x, metrics, color=INK)
        a.set_ylabel(ylabel, color=INK2)
        for sp in ["top", "right"]:
            a.spines[sp].set_visible(False)
        a.grid(axis="y", alpha=0.25)
    ax[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Deep fusion network vs XGBoost on unseen patients (CGM worn)",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "deep_vs_xgboost.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
