"""Physics-informed features: what the calibrated twin expects to happen next.

At every prediction moment, the twin starts from its current internal state
(glucose, insulin action, food still in the gut, recent exercise) and
simulates the next 2 hours. Its forecast becomes extra input for the ML model.

This combines two kinds of intelligence:
    Physiology (the twin) - understands HOW the body works, needs no training data
    Machine learning      - learns patterns the equations miss
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import twin

TWIN_FEATURES = ["twin_glucose_now", "twin_glucose_plus_30", "twin_glucose_plus_60",
                 "twin_glucose_plus_120", "twin_max_2h", "twin_min_2h", "twin_slope_30",
                 "twin_gut_glucose", "twin_insulin_action",
                 "twin_log_SI", "twin_log_DI", "twin_gb", "twin_drug_gain"]
STATE_KEYS = ["G", "X", "I", "Q1", "Q2", "load"]


def load_twin():
    """Calibrated parameters, simulator inputs and the saved twin run."""
    params = pd.read_csv(C.DATA_DIR / "twins" / "twin_params.csv").set_index("patient_id")
    pids = params.index.to_numpy()
    run = np.load(C.DATA_DIR / "twins" / "twin_run.npz", allow_pickle=True)
    states = {k: run[f"state_{k}"].astype(float) for k in STATE_KEYS}
    inp = twin.reconstruct_inputs(pids)
    pri = twin.ehr_priors(pids)
    theta = twin.theta_of(params)
    return dict(params=params, pids=pids, states=states, inp=inp, pri=pri, theta=theta,
                glucose_5min=run["glucose_5min"])


def twin_features(moments: pd.DataFrame, T: dict, chunk: int = 20000) -> pd.DataFrame:
    """Twin forecast features for each (patient_id, timestamp) row."""
    idx = {p: i for i, p in enumerate(T["pids"])}
    rows = moments.patient_id.map(idx).to_numpy()
    minute = ((moments.timestamp - pd.Timestamp(C.START_DATE)).dt.total_seconds() // 60).astype(int).to_numpy()
    j = minute // 15
    out = np.zeros((len(moments), 9))
    for c0 in range(0, len(moments), chunk):
        sl = slice(c0, c0 + chunk)
        st = {k: T["states"][k][rows[sl], j[sl]] for k in STATE_KEYS}
        sim = twin.forecast_from_states(T["theta"], T["pri"], st, rows[sl], minute[sl], T["inp"])
        G = sim["G"]
        g0 = st["G"]
        out[sl] = np.column_stack([g0, G[:, 29], G[:, 59], G[:, 119], G.max(1), G.min(1),
                                   G[:, 29] - g0, (st["Q1"] + st["Q2"]) / 1000, st["X"] * 1000])
    df = pd.DataFrame(out, columns=TWIN_FEATURES[:9], index=moments.index)
    p = T["params"].loc[moments.patient_id]
    df["twin_log_SI"] = p.log_SI.to_numpy()
    df["twin_log_DI"] = (p.log_SI + p.log_beta).to_numpy()
    df["twin_gb"] = p.gb.to_numpy()
    df["twin_drug_gain"] = p.drug_gain.to_numpy()
    return df
