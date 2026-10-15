"""The personal digital twin: calibrate each patient's physiology from their own data.

Every patient's body has hidden numbers that the doctor cannot measure directly:
    SI        - insulin sensitivity ("how rusty are the locks")
    beta      - how well the pancreas still makes insulin
    Gb        - the glucose level the liver drives towards without medicine
    drug_gain - how strongly this patient responds to glimepiride / insulin

Calibration = finding the values that make the simulator reproduce the
patient's real CGM trace, given what they ate, how they walked and slept, and
which medicines they took (all from the app diary and smartwatch).

Data fusion happens here at the physiology level too: the search STARTS from
values estimated from the health record (fasting insulin -> SI, fasting
glucose -> Gb, diabetes duration -> beta), and the CGM refines them.

Search method: the Cross-Entropy Method, a simple evolutionary optimiser.
Each round it tries 24 candidate bodies per patient, keeps the 6 that best
match the CGM, and moves its search towards them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .lifestyle import KA_SCALE, MENU
from .physiology import drug_effects, simulate

DRUG_CLASSES = ["metformin", "sulfonylurea", "dpp4", "sglt2", "basal_insulin"]
P1_POP, DAWN_POP, SSENS_POP = 0.022, 0.08, 0.05     # population values, not fitted
CGM_LAG = 2                                          # CGM trails blood by ~10 min (2 samples)
FIT_NAMES = ["log_SI", "log_beta", "gb", "log_drug_gain"]

KA_BY_FOOD = {name: ka for meal in MENU.values() for opts in meal.values()
              for name, _, ka in opts}


# ---------------------------------------------------------------------------
# 1. Rebuild the simulator inputs from what the apps recorded
# ---------------------------------------------------------------------------
def reconstruct_inputs(pids, n_days: int = C.N_DAYS):
    """1-minute input arrays rebuilt ONLY from logged data (diary + smartwatch)."""
    N, T = len(pids), n_days * 1440
    start = pd.Timestamp(C.START_DATE)
    idx = {p: i for i, p in enumerate(pids)}

    def minute(ts):
        return ((ts - start).dt.total_seconds() // 60).astype(int).to_numpy()

    carbs = np.zeros((N, T))
    kabs = np.full((N, T), 0.04 * KA_SCALE)
    meals = pd.read_csv(C.LOG_DIR / "meals.csv", parse_dates=["timestamp"])
    meals = meals[meals.patient_id.isin(idx)].sort_values("timestamp")
    for pi, m, g, food, kind in zip(meals.patient_id.map(idx), minute(meals.timestamp),
                                    meals.carbs_g, meals.food, meals.meal):
        if 0 <= m < T:
            carbs[pi, m] += g
            kabs[pi, m:m + (240 if kind == "sweet" else 300)] = KA_BY_FOOD.get(food, 0.04) * KA_SCALE

    watch = pd.read_csv(C.WEARABLE_DIR / "smartwatch.csv.gz", usecols=["patient_id", "steps"])
    watch = watch[watch.patient_id.isin(idx)]
    S = len(watch) // N
    steps5 = watch.steps.fillna(0).to_numpy().reshape(N, S)[:, : T // 5]
    steps = np.repeat(steps5 / 5.0, 5, axis=1)

    sleep = pd.read_csv(C.WEARABLE_DIR / "sleep_nightly.csv", parse_dates=["date"])
    sleep = sleep[sleep.patient_id.isin(idx)]
    sleep_hours = np.full((N, n_days), np.nan)
    d = ((sleep.date - start).dt.days).to_numpy()
    ok = (d >= 0) & (d < n_days)
    sleep_hours[sleep.patient_id.map(idx).to_numpy()[ok], d[ok]] = sleep.total_sleep_min.to_numpy()[ok] / 60
    pmean = np.nanmean(sleep_hours, axis=1, keepdims=True)
    sleep_hours = np.where(np.isnan(sleep_hours), pmean, sleep_hours)

    # Doses actually taken, and the full prescribed schedule (taken or not)
    dose = {c: np.zeros((N, T)) for c in DRUG_CLASSES}
    planned = {c: np.zeros((N, T)) for c in DRUG_CLASSES}
    meds = pd.read_csv(C.LOG_DIR / "medications.csv", parse_dates=["timestamp"])
    meds = meds[meds.patient_id.isin(idx) & meds.drug_class.isin(DRUG_CLASSES)]
    for pi, m, cls, taken in zip(meds.patient_id.map(idx), minute(meds.timestamp),
                                 meds.drug_class, meds.taken):
        if 0 <= m < T:
            planned[cls][pi, m] = 1.0
            if taken:
                dose[cls][pi, m] = 1.0
    effects = drug_effects(dose, T)
    effects_raw = drug_effects(dose, T, clip=False)
    return dict(carbs=carbs, kabs=kabs, steps=steps, sleep_hours=sleep_hours, effects=effects,
                effects_raw=effects_raw, dose=dose, planned=planned)


def load_cgm(pids):
    cgm = pd.read_csv(C.WEARABLE_DIR / "cgm.csv.gz")
    cgm = cgm[cgm.patient_id.isin(set(pids))]
    order = {p: i for i, p in enumerate(pids)}
    cgm = cgm.assign(_o=cgm.patient_id.map(order)).sort_values(["_o", "timestamp"], kind="stable")
    return cgm.glucose_mgdl.to_numpy().reshape(len(pids), -1)


# ---------------------------------------------------------------------------
# 2. Starting guesses from the health record (EHR priors)
# ---------------------------------------------------------------------------
def ehr_priors(pids) -> pd.DataFrame:
    p = pd.read_csv(C.EHR_DIR / "patients.csv").set_index("patient_id").loc[pids]
    labs = pd.read_csv(C.EHR_DIR / "observations.csv").sort_values("date")
    latest = labs.groupby(["patient_id", "test"]).value.last().unstack().loc[pids]
    meds = pd.read_csv(C.EHR_DIR / "medications.csv")
    pr = pd.DataFrame(index=pids)
    # HOMA-style: high fasting insulin = the body needs more insulin = low sensitivity
    pr["log_SI"] = np.log(3e-4 * (8.0 / latest["Fasting insulin"].clip(2, 45)) ** (1 / 0.6))
    # Pancreas function declines with years of diabetes
    pr["log_beta"] = np.log(0.028 * np.exp(-0.035 * p.diabetes_duration_yrs))
    # Untreated liver glucose drive, from fasting glucose
    pr["gb"] = (latest["Fasting plasma glucose"] / 0.93).clip(100, 260)
    pr["log_drug_gain"] = 0.0
    pr["body_weight_kg"] = p.weight_kg
    for cls in ["sulfonylurea", "basal_insulin"]:
        pr[f"on_{cls}"] = pr.index.isin(meds[meds.drug_class == cls].patient_id)
    return pr


def to_sim_params(theta: dict, weight, on_su, on_ins) -> dict:
    """Convert fitted values into the simulator's parameter dictionary."""
    SI = np.exp(theta["log_SI"]); gain = np.exp(theta["log_drug_gain"])
    n = len(SI)
    return dict(SI=SI, beta=np.exp(theta["log_beta"]), p1=np.full(n, P1_POP),
                gb_untreated=theta["gb"], dawn_amp=np.full(n, DAWN_POP),
                sleep_sensitivity=np.full(n, SSENS_POP), body_weight_kg=weight,
                su_push=np.where(on_su, np.minimum(gain * 1.7 * P1_POP / SI * 0.12, 9.0), 0.0),
                basal_dose=np.where(on_ins, np.minimum(gain * 0.9 * P1_POP / SI, 75.0), 0.0))


def _slice(inp, rows, t1):
    return dict(carbs=inp["carbs"][rows, :t1], kabs=inp["kabs"][rows, :t1],
                steps=inp["steps"][rows, :t1], sleep_hours=inp["sleep_hours"][rows],
                effects={k: v[rows, :t1] for k, v in inp["effects"].items()})


def run_twin(theta: dict, pri: pd.DataFrame, inp: dict, rows, t1: int, **kw):
    """Simulate the twin for patients `rows` (may repeat) over minutes [0, t1)."""
    x = _slice(inp, rows, t1)
    params = to_sim_params(theta, pri.body_weight_kg.to_numpy()[rows],
                           pri.on_sulfonylurea.to_numpy()[rows],
                           pri.on_basal_insulin.to_numpy()[rows])
    init = dict(G=pri.gb.to_numpy()[rows] * 0.95)
    return simulate(params, x["carbs"], x["kabs"], x["steps"], x["sleep_hours"],
                    x["effects"], init=init, **kw)


def fit_error(sim_g5: np.ndarray, cgm: np.ndarray, s0: int, s1: int) -> np.ndarray:
    """Mean absolute difference between twin and CGM over samples [s0, s1)."""
    pred = sim_g5[:, s0 - CGM_LAG: s1 - CGM_LAG]
    obs = cgm[:, s0:s1]
    return np.nanmean(np.abs(pred - obs), axis=1)


# ---------------------------------------------------------------------------
# 3. Calibration (Cross-Entropy Method, all patients in parallel)
# ---------------------------------------------------------------------------
def calibrate(pids, inp, cgm, pri, fit_days=(1, 7), n_cand=24, n_elite=6,
              n_iter=10, chunk=25, seed=C.SEED, verbose=True):
    """Fit each patient's twin on CGM days [fit_days[0], fit_days[1])."""
    rng = np.random.default_rng(seed)
    N = len(pids)
    t1 = fit_days[1] * 1440
    s0, s1 = fit_days[0] * 288, fit_days[1] * 288
    mean = pri[FIT_NAMES].to_numpy().astype(float)
    std = np.tile([0.6, 0.6, 30.0, 0.5], (N, 1))
    # drug gain only matters for patients on glimepiride / insulin
    has_drug = (pri.on_sulfonylurea | pri.on_basal_insulin).to_numpy()
    std[~has_drug, 3] = 0.0
    history = []
    for it in range(n_iter):
        new_mean = mean.copy(); new_std = std.copy(); errs = np.zeros(N)
        for c0 in range(0, N, chunk):
            rows = np.arange(c0, min(c0 + chunk, N))
            cand = mean[rows, None, :] + std[rows, None, :] * rng.normal(size=(len(rows), n_cand, 4))
            cand[:, 0, :] = mean[rows]                       # always re-test the current best guess
            cand[:, :, 2] = np.clip(cand[:, :, 2], 90, 300)
            cand[:, :, 0] = np.clip(cand[:, :, 0], np.log(2e-5), np.log(2e-3))
            cand[:, :, 1] = np.clip(cand[:, :, 1], np.log(1e-3), np.log(0.1))
            cand[:, :, 3] = np.clip(cand[:, :, 3], -1.5, 1.5)
            flat = cand.reshape(-1, 4)
            rep = np.repeat(rows, n_cand)
            theta = {n: flat[:, j] for j, n in enumerate(FIT_NAMES)}
            sim = run_twin(theta, pri, inp, rep, t1, out_every=5)
            e = fit_error(sim["G"], cgm[rep], s0, s1).reshape(len(rows), n_cand)
            best = np.argsort(e, axis=1)[:, :n_elite]
            elite = np.take_along_axis(cand, best[:, :, None], axis=1)
            new_mean[rows] = elite[:, 0, :] * 0.4 + elite.mean(1) * 0.6
            new_std[rows] = np.maximum(elite.std(1), [0.02, 0.02, 1.0, 0.02]) * (std[rows] > 0)
            errs[rows] = e[:, 0] if it == 0 else e.min(1)
        mean, std = new_mean, new_std
        history.append(float(np.median(errs)))
        if verbose:
            print(f"   round {it + 1:2d}/{n_iter}: median fit error {history[-1]:.1f} mg/dL")
    return pd.DataFrame(mean, index=pids, columns=FIT_NAMES), history


def theta_of(df: pd.DataFrame) -> dict:
    return {n: df[n].to_numpy().astype(float) for n in FIT_NAMES}


# ---------------------------------------------------------------------------
# 4. Forecasting forward
# ---------------------------------------------------------------------------
def future_effects(inp: dict, rows, start_min, horizon: int) -> dict:
    """Medicine effects over the forecast window, using only what is known now.

    Doses already taken keep acting. Doses due inside the window are assumed
    to be taken as prescribed (the twin cannot know if the patient will skip).
    """
    from .physiology import _kernel
    T = inp["carbs"].shape[1]
    tt = start_min[:, None] + np.arange(1, horizon + 1)[None, :]
    ttc = np.minimum(tt, T - 1)
    lag = np.arange(horizon)
    out = {}
    for cls, raw in inp["effects_raw"].items():
        base = raw[rows[:, None], ttc]
        delta = inp["planned"][cls][rows[:, None], ttc] - inp["dose"][cls][rows[:, None], ttc]
        if np.any(delta):
            k = _kernel(cls)[:horizon]
            K = np.where(lag[None, :] >= lag[:, None], k[np.clip(lag[None, :] - lag[:, None], 0, None)], 0.0)
            base = base + delta @ K
        out[cls] = np.clip(base, 0, 1.0)
    return out


# ---------------------------------------------------------------------------
# 5. Forecasting forward from any moment ("what happens next if ...")
# ---------------------------------------------------------------------------
def forecast_from_states(theta: dict, pri: pd.DataFrame, states: dict, rows, start_min,
                         inp: dict, horizon: int = 120, carbs=None, steps=None,
                         sleep_hours=None, effects=None, kabs=None):
    """Run the twin forward from saved states.

    rows: patient index for each forecast column; start_min: absolute minute
    (from day 0) each forecast starts at. Future meals and walks are unknown,
    so by default the forecast assumes none (what-if scenarios add them).
    Medicines follow the patient's schedule.
    """
    rows = np.asarray(rows); start_min = np.asarray(start_min)
    n = len(rows)
    tt = start_min[:, None] + np.arange(1, horizon + 1)[None, :]
    T = inp["carbs"].shape[1]
    ttc = np.minimum(tt, T - 1)
    if carbs is None:
        carbs = np.zeros((n, horizon))
    if steps is None:
        steps = np.zeros((n, horizon))
    if kabs is None:
        kabs = np.repeat(inp["kabs"][rows, np.minimum(start_min, T - 1)][:, None], horizon, 1)
    if effects is None:
        effects = future_effects(inp, rows, start_min, horizon)
    if sleep_hours is None:
        sleep_hours = inp["sleep_hours"][rows]
    params = to_sim_params({k: v[rows] for k, v in theta.items()},
                           pri.body_weight_kg.to_numpy()[rows],
                           pri.on_sulfonylurea.to_numpy()[rows],
                           pri.on_basal_insulin.to_numpy()[rows])
    init = {k: v for k, v in states.items()}
    return simulate(params, carbs, kabs, steps, sleep_hours, effects, init=init,
                    start_minute_of_day=start_min + 1)
