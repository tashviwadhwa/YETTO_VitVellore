"""Day 3, step 3: train the final models used by the doctor dashboard.

GlucoTwin's alert engines:
    Spike alert, CGM worn     - XGBoost on CGM + smartwatch + diary + EHR
    Spike alert, no CGM       - XGBoost on smartwatch + diary + EHR + twin forecast
    Hypo alert (both modes)   - the calibrated physics twin: alert when its
                                2-hour forecast dips below a safety threshold
                                (hypos are too rare for ML to learn well; see
                                results/hybrid_results.json)

Every spike alert comes with its reasons (exact TreeSHAP contributions).

Run from the project root (after 05_hybrid_model.py):
    python scripts/06_train_final_models.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glucotwin import config as C
from glucotwin.explain import contributions, source_of, top_reasons
from glucotwin.hybrid import TWIN_FEATURES

MODELS = C.ROOT / "models"
OUT = C.ROOT / "results"
INK, INK2 = "#0b0b0b", "#52514e"
SOURCE_COLORS = {"Twin": "#2a78d6", "EHR": "#eb6834", "CGM": "#1baf7a",
                 "Smartwatch": "#eda100", "Diary": "#e87ba4", "Time": "#008300"}


def train(d, cols, label):
    m = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.7, min_child_weight=10, reg_lambda=5,
                          tree_method="hist", n_jobs=-1, random_state=C.SEED)
    m.fit(d[cols], d[label])
    return m


def hypo_threshold_table(d):
    """How the twin's hypo alert performs at different safety thresholds."""
    y = d.hypo_2h.to_numpy()
    n_days = d.patient_id.nunique() * 7
    rows = []
    for th in [70, 75, 80, 85, 90, 95, 100]:
        alert = d.twin_min_2h.to_numpy() < th
        tp, fp = (alert & (y == 1)).sum(), (alert & (y == 0)).sum()
        rows.append(dict(threshold_mgdl=th, sensitivity=tp / max(y.sum(), 1),
                         precision=tp / max(tp + fp, 1),
                         false_alert_checks_per_patient_day=fp / n_days))
    return pd.DataFrame(rows)


def importance_by_source(contrib: pd.DataFrame) -> pd.Series:
    imp = contrib.drop(columns="_baseline").abs().mean()
    return imp.groupby(imp.index.map(source_of)).sum() / imp.sum()


def main():
    t0 = time.time()
    MODELS.mkdir(exist_ok=True)
    df = pd.read_csv(C.DATA_DIR / "processed" / "fused_dataset.csv.gz", parse_dates=["timestamp"])
    g = json.loads((C.DATA_DIR / "processed" / "feature_groups.json").read_text())
    tf = pd.read_csv(C.DATA_DIR / "processed" / "twin_features_week2.csv.gz", parse_dates=["timestamp"])
    week2 = df.merge(tf, on=["patient_id", "timestamp"])

    cols_cgm = g["cgm"] + g["watch"] + g["diary"] + g["time"] + g["static"]
    cols_nocgm = g["watch"] + g["diary"] + g["time"] + g["static"] + TWIN_FEATURES

    print("[1/4] Training spike model, CGM worn (all 13 days) ...")
    d_cgm = df[df.spike_eligible == 1]
    m_cgm = train(d_cgm, cols_cgm, "spike_2h")
    m_cgm.save_model(MODELS / "spike_cgm.json")

    print("[2/4] Training spike model, no CGM (smartwatch + EHR + twin, week 2) ...")
    d_no = week2[week2.spike_eligible == 1]
    m_no = train(d_no, cols_nocgm, "spike_2h")
    m_no.save_model(MODELS / "spike_nocgm.json")

    print("[3/4] Choosing the twin's hypo safety threshold ...")
    tab = hypo_threshold_table(week2[week2.hypo_eligible == 1])
    print(tab.round(3).to_string(index=False))
    chosen = tab[tab.sensitivity >= 0.8].iloc[0] if (tab.sensitivity >= 0.8).any() else tab.iloc[-1]
    hypo_th = float(chosen.threshold_mgdl)
    print(f"   -> alert when the twin forecasts glucose below {hypo_th:.0f} mg/dL within 2 h "
          f"(catches {chosen.sensitivity:.0%} of hypos)")

    print("[4/4] Explaining the models (TreeSHAP) ...")
    rng = np.random.default_rng(C.SEED)
    s_cgm = d_cgm.iloc[rng.choice(len(d_cgm), 20000, replace=False)]
    s_no = d_no.iloc[rng.choice(len(d_no), 20000, replace=False)]
    c_cgm = contributions(m_cgm, s_cgm[cols_cgm])
    c_no = contributions(m_no, s_no[cols_nocgm])
    share = pd.DataFrame({"CGM worn": importance_by_source(c_cgm),
                          "No CGM": importance_by_source(c_no)}).fillna(0)

    # Example alert explanation
    p = m_no.predict_proba(s_no[cols_nocgm])[:, 1]
    i = int(np.argmax(np.where(s_no.spike_2h.to_numpy() == 1, p, 0) * (p < 0.95)))
    ex = s_no.iloc[i]
    print(f"\n   Example alert: {ex.patient_id} at {ex.timestamp:%d %b %H:%M}, "
          f"spike risk {p[i]:.0%} (spike did happen: {bool(ex.spike_2h)})")
    for r in top_reasons(c_no.iloc[i], ex):
        print(f"     {r['name']:32s} {r['value']:>12s}  [{r['source']}]  {r['direction']}")

    plot_importance(c_no, share)
    card = dict(
        spike_cgm=dict(file="spike_cgm.json", features=cols_cgm, trained_rows=int(len(d_cgm))),
        spike_nocgm=dict(file="spike_nocgm.json", features=cols_nocgm, trained_rows=int(len(d_no))),
        hypo_twin=dict(rule="twin_min_2h < threshold", threshold_mgdl=hypo_th,
                       threshold_table=tab.round(4).to_dict(orient="records")),
        explanation_share_by_source=share.round(4).to_dict())
    (MODELS / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"\nSaved models/ in {time.time() - t0:.0f} s")


def plot_importance(c_no, share):
    imp = c_no.drop(columns="_baseline").abs().mean()
    imp["hour_sin"] = imp.get("hour_sin", 0) + imp.get("hour_cos", 0)
    imp = imp.drop("hour_cos", errors="ignore").sort_values(ascending=False).head(14)[::-1]
    from glucotwin.explain import FEATURES
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    cols = [SOURCE_COLORS.get(source_of(f), "#999") for f in imp.index]
    labels = [FEATURES.get(f, (f,))[0] for f in imp.index]
    ax[0].barh(range(len(imp)), imp.values, color=cols, height=0.65)
    ax[0].set_yticks(range(len(imp)), labels, color=INK, fontsize=9.5)
    ax[0].set_xlabel("Average impact on spike risk (|SHAP|, log-odds)", color=INK2)
    ax[0].set_title("No-CGM model: what drives a spike alert", loc="left", fontsize=11.5, color=INK)
    order = [s for s in SOURCE_COLORS if s in share.index]
    y = np.arange(2)
    left = np.zeros(2)
    for s in order:
        v = share.loc[s, ["CGM worn", "No CGM"]].to_numpy()
        ax[1].barh(y, v, left=left, color=SOURCE_COLORS[s], height=0.5, label=s,
                   edgecolor="white", linewidth=2)
        for yi, (l, w) in enumerate(zip(left, v)):
            if w > 0.06:
                ax[1].text(l + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=9, color=INK)
        left += v
    ax[1].set_yticks(y, ["CGM worn", "No CGM"], color=INK)
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("Share of the explanation", color=INK2)
    ax[1].set_title("Which data source the model relies on", loc="left", fontsize=11.5, color=INK)
    ax[1].legend(frameon=False, ncol=3, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    for a in ax:
        for sp in ["top", "right"]:
            a.spines[sp].set_visible(False)
    ax[0].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "explainability.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
