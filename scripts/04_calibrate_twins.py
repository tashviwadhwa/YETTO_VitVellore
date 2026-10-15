"""Day 3, step 1: build a calibrated digital twin for every patient.

Protocol (like a real clinic):
    Week 1 (days 1-6): patient wears a CGM; the twin is calibrated on it.
    Week 2 (days 7-13): CGM removed. The twin must reproduce the patient's
                        glucose using ONLY the app diary + smartwatch.

Three twins are compared on week 2:
    Population twin  - an "average Indian T2D patient" (no personalisation)
    EHR twin         - personalised from the health record only
    Calibrated twin  - health record + one week of CGM (GlucoTwin)

Run from the project root:
    python scripts/04_calibrate_twins.py
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glucotwin import config as C
from glucotwin import twin

TWIN_DIR = C.DATA_DIR / "twins"
OUT = C.ROOT / "results"
BLUE, ORANGE, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#b4b2ab", "#0b0b0b", "#52514e"
FIT_DAYS = (1, 7)      # calibrate on days 1-6 (day 0 is warm-up)
VAL_DAYS = (7, 14)     # validate on days 7-13


def week2_error(sim_g5, cgm):
    s0, s1 = VAL_DAYS[0] * 288, VAL_DAYS[1] * 288
    return twin.fit_error(sim_g5, cgm, s0, s1)


def main():
    t0 = time.time()
    TWIN_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    pids = pd.read_csv(C.EHR_DIR / "patients.csv").patient_id.to_numpy()
    N = len(pids)

    print("[1/4] Rebuilding each patient's day from the app diary + smartwatch ...")
    inp = twin.reconstruct_inputs(pids)
    cgm = twin.load_cgm(pids)
    pri = twin.ehr_priors(pids)

    print("[2/4] Calibrating 200 twins on week-1 CGM (Cross-Entropy Method) ...")
    est, history = twin.calibrate(pids, inp, cgm, pri, fit_days=FIT_DAYS, n_iter=12)

    print("[3/4] Validating on week 2 (twin sees NO glucose readings) ...")
    T = C.N_DAYS * 1440
    rows = np.arange(N)
    pop = pri[twin.FIT_NAMES].copy()
    for c in twin.FIT_NAMES:
        pop[c] = pri[c].median()
    pop["log_drug_gain"] = 0.0
    errs, runs = {}, {}
    for name, th in [("Population twin", pop), ("EHR twin", pri[twin.FIT_NAMES]),
                     ("Calibrated twin (GlucoTwin)", est)]:
        sim = twin.run_twin(twin.theta_of(th), pri, inp, rows, T, out_every=5,
                            save_states_every=15 if "Calibrated" in name else None)
        errs[name] = week2_error(sim["G"], cgm)
        runs[name] = sim
        print(f"   {name:28s} week-2 error: median {np.median(errs[name]):5.1f} mg/dL")

    # Parameter recovery vs the hidden ground truth
    truth = pd.read_csv(C.TRUTH_DIR / "twin_parameters.csv").set_index("patient_id").loc[pids]
    # Glucose data alone cannot separate SI from beta (a well-known property of
    # the minimal model), but their product - the clinical "disposition index",
    # a measure of overall diabetes severity - is identifiable.
    pri["log_DI"] = pri.log_SI + pri.log_beta
    est["log_DI"] = est.log_SI + est.log_beta
    rec = {}
    for nm, col, f in [("Insulin sensitivity (SI)", "log_SI", lambda t: np.log(t.SI)),
                       ("Disposition index (SI x beta)", "log_DI", lambda t: np.log(t.SI * t.beta)),
                       ("Liver glucose drive (Gb)", "gb", lambda t: t.gb_untreated)]:
        rec[nm] = dict(ehr=spearmanr(pri[col], f(truth))[0], cal=spearmanr(est[col], f(truth))[0])
        print(f"   {nm:28s} rank correlation with truth: EHR {rec[nm]['ehr']:.2f} -> calibrated {rec[nm]['cal']:.2f}")

    print("[4/4] Saving twins and figures ...")
    cal = runs["Calibrated twin (GlucoTwin)"]
    params = est.drop(columns="log_DI").copy()
    params["SI"] = np.exp(est.log_SI); params["beta"] = np.exp(est.log_beta)
    params["drug_gain"] = np.exp(est.log_drug_gain)
    params["disposition_index"] = params.SI * params.beta
    params["prior_SI"] = np.exp(pri.log_SI); params["prior_beta"] = np.exp(pri.log_beta)
    params["prior_gb"] = pri.gb
    params["fit_error_week1"] = twin.fit_error(cal["G"], cgm, FIT_DAYS[0] * 288, FIT_DAYS[1] * 288)
    params["error_week2"] = errs["Calibrated twin (GlucoTwin)"]
    params.index.name = "patient_id"
    params.to_csv(TWIN_DIR / "twin_params.csv")
    np.savez_compressed(TWIN_DIR / "twin_run.npz", patient_ids=pids,
                        glucose_5min=cal["G"].astype(np.float32),
                        **{f"state_{k}": v.astype(np.float32) for k, v in cal["states"].items()})

    # Figure 1: one patient's week 2, twin vs CGM; plus error comparison
    k = int(np.argsort(errs["Calibrated twin (GlucoTwin)"])[N // 2])        # a typical patient
    days = (4.0 / 24 + 7, 7 + 2 + 4.0 / 24)
    s = np.arange(int(days[0] * 288), int(days[1] * 288))
    ts = pd.Timestamp(C.START_DATE) + pd.to_timedelta(s * 5, unit="min")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[2.3, 1]))
    ax[0].axhspan(70, 180, color="#1baf7a", alpha=0.07)
    ax[0].plot(ts, cgm[k, s], color=GREY, lw=1.6, label="Real CGM (hidden from the twin)")
    ax[0].plot(ts, runs["Population twin"]["G"][k, s - twin.CGM_LAG], color=ORANGE, lw=1.4,
               ls="--", label="Population twin")
    ax[0].plot(ts, cal["G"][k, s - twin.CGM_LAG], color=BLUE, lw=2, label="Calibrated twin (GlucoTwin)")
    ax[0].set_ylabel("Glucose (mg/dL)", color=INK2)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax[0].set_title(f"Week 2, patient {pids[k]}: the twin reproduces glucose without a CGM",
                    loc="left", fontsize=11.5, color=INK)
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper left", ncol=3)
    ax[0].set_ylim(40, max(320, np.nanmax(cgm[k, s]) + 40))
    names = list(errs)
    vals = [np.median(errs[n]) for n in names]
    yb = np.arange(len(names))[::-1]
    ax[1].barh(yb, vals, color=[GREY, GREY, BLUE], height=0.55)
    for yi, v in zip(yb, vals):
        ax[1].text(v + 0.6, yi, f"{v:.1f}", va="center", color=INK)
    ax[1].set_yticks(yb, [n.replace(" (GlucoTwin)", "") for n in names], color=INK)
    ax[1].set_xlabel("Median error vs real CGM, week 2 (mg/dL)", color=INK2)
    ax[1].set_title("Lower is better", loc="left", fontsize=11, color=INK)
    for a in ax:
        for sp in ["top", "right"]:
            a.spines[sp].set_visible(False)
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "twin_validation.png", dpi=150)
    plt.close(fig)

    # Figure 2: hidden physiology recovered
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, (nm, col, f, logscale) in zip(ax, [
            ("Insulin sensitivity (SI)", "log_SI", lambda t: t.SI, True),
            ("Disposition index (SI x beta)", "log_DI", lambda t: t.SI * t.beta, True),
            ("Liver glucose drive (Gb)", "gb", lambda t: t.gb_untreated, False)]):
        tr = f(truth).to_numpy()
        conv = np.exp if logscale else (lambda v: v)
        a.scatter(tr, conv(pri[col]), s=12, color=GREY, label=f"From health record (ρ={rec[nm]['ehr']:.2f})")
        a.scatter(tr, conv(est[col]), s=12, color=BLUE, label=f"Calibrated twin (ρ={rec[nm]['cal']:.2f})")
        lo, hi = np.nanmin(tr) * 0.8, np.nanmax(tr) * 1.2
        a.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls=":")
        if logscale:
            a.set_xscale("log"); a.set_yscale("log")
        a.set_title(nm, loc="left", fontsize=11, color=INK)
        a.set_xlabel("True hidden value", color=INK2); a.set_ylabel("Twin's estimate", color=INK2)
        a.legend(frameon=False, fontsize=8, loc="upper left")
        for sp in ["top", "right"]:
            a.spines[sp].set_visible(False)
    fig.suptitle("The twin recovers each patient's hidden physiology from one week of data",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "twin_recovery.png", dpi=150)
    plt.close(fig)

    summary = dict(week2_median_error={n: float(np.median(v)) for n, v in errs.items()},
                   parameter_recovery_spearman=rec, calibration_history=history)
    pd.Series(summary).to_json(OUT / "twin_results.json", indent=2)
    print(f"Done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
