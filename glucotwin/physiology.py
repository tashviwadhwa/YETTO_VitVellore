"""The physiology engine of the digital twin: a Type 2 Diabetes glucose-insulin model.

Core: the Bergman Minimal Model (Bergman et al., 1979), the classic model of
how glucose and insulin interact, extended for Type 2 Diabetes and daily life:

    Gut  (2-compartment meal absorption, Dalla Man-style):
        dQ1/dt = -ka * Q1 + meal_carbs
        dQ2/dt =  ka * (Q1 - Q2)
        Ra     =  f * ka * Q2 / (BW * Vg)            glucose entering blood

    Glucose:
        dG/dt = -(p1 + X) * G + p1 * Gb  + Ra        Bergman glucose equation
                - E_ex * G                           walking burns glucose
                - SGLT2 excretion + counter-regulation (glucagon when low)

    Remote insulin action (how strongly insulin is working):
        dX/dt = -p2 * X + p2 * SI * (I + I_exo - Ib)

    Insulin (pancreas, impaired in T2D):
        dI/dt = -n * (I - Ib) + beta * (G - Gb)+  + sulfonylurea push

Type 2 Diabetes lives in the parameters: low SI = insulin resistance
("rusty locks"), low beta = tired pancreas. Each patient gets their own
values, which is what makes the twin *personal*.

Daily-life effects:
    * short sleep lowers SI the next day
    * walking adds insulin-independent glucose uptake and raises SI for hours
    * dawn phenomenon: liver releases extra glucose around 4-8 AM
    * medicines: metformin (less liver glucose, better SI), sulfonylurea
      (forces insulin release -> hypo risk), DPP-4 inhibitor (boosts
      meal-time insulin), SGLT2 inhibitor (glucose lost in urine),
      basal insulin (background insulin)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

# Fixed physiological constants (population values)
VG = 1.9        # glucose distribution volume, dL/kg
F_ABS = 0.9     # fraction of meal carbohydrate reaching blood
P2 = 0.02       # insulin action decay rate, 1/min
N_INS = 0.12    # insulin clearance, 1/min
IB = 10.0       # basal plasma insulin, uU/mL
G_MIN, G_MAX = 39.0, 450.0


def _kernel(profile: str, length: int = 1440) -> np.ndarray:
    t = np.arange(length, dtype=float)
    if profile == "metformin":      # ~12 h action, onset ~1 h
        k = (1 - np.exp(-t / 60)) * (t < 720)
    elif profile == "sulfonylurea":  # peak ~3 h, lasts ~20 h
        k = (t / 180) ** 2 * np.exp(2 * (1 - t / 180))
    elif profile == "basal_insulin":  # nearly flat for 24 h
        k = (1 - np.exp(-t / 90)) * np.exp(-t / 3000)
    else:                             # dpp4, sglt2: once daily, ~24 h
        k = (1 - np.exp(-t / 60)) * np.exp(-t / 2400)
    return k / k.max()


def drug_effects(dose: dict, T: int, clip: bool = True) -> dict:
    """Turn dose events (1 = dose taken that minute) into effect curves 0..1."""
    eff = {}
    for cls, arr in dose.items():
        k = _kernel(cls)
        conv = fftconvolve(arr, k[None, :], axes=1)[:, :T]
        if clip:
            eff[cls] = np.clip(conv, 0, 1.0)
        else:
            conv[np.abs(conv) < 1e-9] = 0.0     # remove FFT round-off
            eff[cls] = conv
    return eff


def simulate(params: dict, carbs: np.ndarray, kabs: np.ndarray, steps: np.ndarray,
             sleep_hours: np.ndarray, effects: dict, init: dict | None = None,
             start_minute_of_day=0, save_states_every: int | None = None,
             out_every: int = 1) -> dict:
    """Simulate many patients at once, 1-minute steps.

    params: arrays of shape (N,) - SI, beta, p1, gb_untreated, dawn_amp,
            sleep_sensitivity, body_weight_kg
    carbs/kabs/steps: (N, T) per-minute inputs
    sleep_hours: (N, n_days) hours slept the night before each day
    effects: drug effect curves (N, T) from drug_effects()
    start_minute_of_day: minute (from the start of sleep_hours day 0) at which
            each column starts; a number or an (N,) array
    save_states_every: if set, also return the full internal state every k
            minutes, so the twin can later forecast forward from any moment
            (state saved at minutes 0, k, 2k, ... after that minute's update)
    out_every: store the glucose/insulin outputs every k minutes (saves memory)
    """
    N, T = carbs.shape
    SI0 = params["SI"]; beta = params["beta"]; p1 = params["p1"]
    gbu = params["gb_untreated"]; dawn = params["dawn_amp"]
    ssens = params["sleep_sensitivity"]; bw = params["body_weight_kg"]
    su_push = params.get("su_push", np.zeros(N)); basal_dose = params.get("basal_dose", np.zeros(N))
    zeros = np.zeros((N, T))
    met = effects.get("metformin", zeros); su = effects.get("sulfonylurea", zeros)
    dpp = effects.get("dpp4", zeros); sglt = effects.get("sglt2", zeros)
    bas = effects.get("basal_insulin", zeros)

    # State
    init = init or {}
    G = init.get("G", gbu * 0.95).astype(float).copy()
    X = init.get("X", np.zeros(N)).copy()
    I = init.get("I", np.full(N, IB)).copy()
    Q1 = init.get("Q1", np.zeros(N)).copy()
    Q2 = init.get("Q2", np.zeros(N)).copy()
    load = init.get("load", np.zeros(N)).copy()

    n_out = (T + out_every - 1) // out_every
    out_G = np.empty((N, n_out)); out_I = np.empty((N, n_out)); out_X = np.empty((N, n_out))
    decay = np.exp(-1 / 480)      # exercise memory ~8 h
    n_days = sleep_hours.shape[1]
    rows = np.arange(N)
    t0 = np.broadcast_to(np.asarray(start_minute_of_day), (N,))
    keep = {}
    if save_states_every:
        n_keep = (T + save_states_every - 1) // save_states_every
        keep = {k: np.empty((N, n_keep)) for k in ["G", "X", "I", "Q1", "Q2", "load"]}

    for t in range(T):
        mod = (t0 + t) % 1440
        day = np.minimum((t0 + t) // 1440, n_days - 1)
        hour = mod / 60

        # Gut absorption
        ka = kabs[:, t]
        Q1 = Q1 + carbs[:, t] * 1000 * F_ABS - ka * Q1
        Q2 = Q2 + ka * (Q1 - Q2)
        Ra = ka * Q2 / (bw * VG)

        # Time-varying modifiers
        sleep_debt = np.maximum(0, 7 - sleep_hours[rows, day])
        si_mult = np.clip(1 - ssens * sleep_debt, 0.65, 1.0)
        load = load * decay + steps[:, t]
        ex_boost = 1 + 0.3 * np.minimum(load / 4000, 1)
        SI = SI0 * si_mult * ex_boost * (1 + 0.12 * met[:, t])
        dawn_curve = np.exp(-((hour - 6.5) / 1.3) ** 2)
        Gb = gbu * (1 - 0.13 * met[:, t]) * (1 + dawn * dawn_curve)

        # Insulin: glucose-stimulated (DPP-4 boosts it) + sulfonylurea push
        secretion = beta * (1 + 0.35 * dpp[:, t]) * np.maximum(G - Gb, 0) + su_push * su[:, t]
        I = I + (-N_INS * (I - IB) + secretion)
        I_exo = basal_dose * bas[:, t]
        X = X + (-P2 * X + P2 * SI * (I + I_exo - IB))

        # Glucose
        E_ex = 0.0075 * steps[:, t] / 100
        counter = 0.06 * np.maximum(68 - G, 0)
        renal = 0.004 * np.maximum(G - 160, 0) * sglt[:, t]
        dG = -(p1 + X) * G + p1 * Gb + Ra - E_ex * G + counter - renal
        G = np.clip(G + dG, G_MIN, G_MAX)

        if t % out_every == 0:
            j = t // out_every
            out_G[:, j] = G; out_I[:, j] = I; out_X[:, j] = X
        if save_states_every and t % save_states_every == 0:
            j = t // save_states_every
            for k, v in zip(["G", "X", "I", "Q1", "Q2", "load"], [G, X, I, Q1, Q2, load]):
                keep[k][:, j] = v

    final = dict(G=G, X=X, I=I, Q1=Q1, Q2=Q2, load=load)
    return dict(G=out_G, I=out_I, X=out_X, final=final, states=keep)


def cgm_readings(G: np.ndarray, rng: np.random.Generator, interval: int = 5,
                 sensor_days: int = 10) -> np.ndarray:
    """Turn true blood glucose into what a CGM sensor would report every 5 min.

    Realism: interstitial-fluid lag (~8 min), noise (~9% MARD, like Libre/Dexcom),
    calibration drift that resets when the sensor is changed every 10 days,
    and occasional signal dropouts (NaN).
    """
    N, T = G.shape
    ig = np.empty_like(G)
    ig[:, 0] = G[:, 0]
    a = 1 / 8
    for t in range(1, T):
        ig[:, t] = ig[:, t - 1] + a * (G[:, t] - ig[:, t - 1])
    sampled = ig[:, ::interval]
    S = sampled.shape[1]
    per_sensor = sensor_days * 1440 // interval
    drift = np.zeros((N, S))
    for s0 in range(0, S, per_sensor):
        s1 = min(s0 + per_sensor, S)
        bias = rng.normal(0, 0.03, size=(N, 1))
        walk = np.cumsum(rng.normal(0, 0.0004, size=(N, s1 - s0)), axis=1)
        drift[:, s0:s1] = bias + walk
    noise = rng.normal(0, 1, size=(N, S)) * (3 + 0.035 * sampled)
    cgm = np.clip(sampled * (1 + drift) + noise, 40, 400).round()
    # dropouts: Bluetooth gaps / compression lows
    for k in range(N):
        n_gaps = rng.poisson(S * 0.002)
        for _ in range(n_gaps):
            s = rng.integers(S)
            cgm[k, s:s + rng.integers(2, 24)] = np.nan
    return cgm
