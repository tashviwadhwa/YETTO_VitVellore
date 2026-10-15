"""Deep fusion network: one neural network that reads both data streams directly.

Architecture (two branches + multi-task heads):

    Sensor branch   last 3 hours, every 5 min (36 steps x 10 channels):
                    glucose, heart rate, HRV, steps, asleep, carbs eaten,
                    glimepiride dose, insulin dose + 2 "missing reading" masks
                    -> 2-layer GRU (a recurrent network that reads time order)
    Record branch   31 health-record features -> 2-layer MLP
    Fusion          [sensor summary | record summary | time of day] -> MLP
    Heads           spike in 2 h, hypo in 2 h, glucose at +30/+60/+120 min

Learning all five targets together (multi-task) lets the rare hypo task
borrow what the network learns from the common spike and forecast tasks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .features import STATIC_FEATURES, ffill_limited

SEQ_LEN = 36                       # 3 hours of 5-minute samples
CHANNELS = ["glucose", "hr", "hrv", "steps", "asleep", "carbs", "su_dose", "ins_dose",
            "glucose_missing", "hrv_missing"]
TARGETS_REG = ["glucose_plus_30", "glucose_plus_60", "glucose_plus_120"]


def build_sequences(A: dict, df: pd.DataFrame, use_cgm: bool = True) -> np.ndarray:
    """(rows, SEQ_LEN, channels) float32 tensor of the last 3 h for each moment."""
    pid_idx = {p: i for i, p in enumerate(A["pids"])}
    r = df.patient_id.map(pid_idx).to_numpy()
    t = ((df.timestamp - A["start"]).dt.total_seconds() // 300).astype(int).to_numpy()
    win = t[:, None] + np.arange(-SEQ_LEN + 1, 1)[None, :]
    g = ffill_limited(A["glucose"], 6)
    hr = ffill_limited(A["hr"], 24)
    hrv = A["hrv"]
    ch = [
        (np.nan_to_num(g, nan=150.0) - 150) / 50 if use_cgm else np.zeros_like(g),
        (np.nan_to_num(hr, nan=72.0) - 72) / 12,
        (np.nan_to_num(hrv, nan=35.0) - 35) / 12,
        A["steps"] / 200,
        A["asleep"],
        A["carbs"] / 50,
        A["dose_sulfonylurea"],
        A["dose_basal_insulin"],
        np.isnan(g).astype(float) if use_cgm else np.ones_like(g),
        np.isnan(hrv).astype(float),
    ]
    X = np.stack([c[r[:, None], win] for c in ch], axis=-1).astype(np.float32)
    return X


class FusionNet(nn.Module):
    def __init__(self, n_ch=len(CHANNELS), n_static=len(STATIC_FEATURES), n_time=3,
                 hidden=64, static_dim=32, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_ch, hidden, num_layers=2, batch_first=True, dropout=dropout)
        self.static = nn.Sequential(nn.Linear(n_static, 64), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(64, static_dim), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(hidden + static_dim + n_time, 128), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU())
        self.head_cls = nn.Linear(64, 2)          # spike, hypo (logits)
        self.head_reg = nn.Linear(64, len(TARGETS_REG))

    def forward(self, seq, static, time):
        _, h = self.gru(seq)
        z = torch.cat([h[-1], self.static(static), time], dim=1)
        z = self.fuse(z)
        return self.head_cls(z), self.head_reg(z)


class StaticScaler:
    def fit(self, X):
        self.mu = np.nanmean(X, 0); self.sd = np.nanstd(X, 0) + 1e-6
        return self

    def transform(self, X):
        return np.nan_to_num((X - self.mu) / self.sd).astype(np.float32)


def make_targets(df):
    y_cls = df[["spike_2h", "hypo_2h"]].to_numpy(np.float32)
    m_cls = df[["spike_eligible", "hypo_eligible"]].to_numpy(np.float32)
    y_reg = df[TARGETS_REG].to_numpy(np.float32)
    m_reg = np.isfinite(y_reg).astype(np.float32)
    y_reg = np.nan_to_num((y_reg - 150) / 50)
    return y_cls, m_cls, y_reg, m_reg


def train_net(seq, static, time, targets, epochs=8, batch=512, lr=2e-3, hypo_weight=30.0,
              seed=42, verbose=True, val=None):
    torch.manual_seed(seed); np.random.seed(seed)
    net = FusionNet()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(seq) // batch + 1))
    y_cls, m_cls, y_reg, m_reg = [torch.from_numpy(a) for a in targets]
    S, Z, Tm = torch.from_numpy(seq), torch.from_numpy(static), torch.from_numpy(time)
    pos_w = torch.tensor([1.0, hypo_weight])
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(S))
        tot = 0.0
        for i in range(0, len(S), batch):
            b = perm[i:i + batch]
            lc, lr_ = net(S[b], Z[b], Tm[b])
            bce = nn.functional.binary_cross_entropy_with_logits(lc, y_cls[b], pos_weight=pos_w,
                                                                 reduction="none")
            loss_c = (bce * m_cls[b]).sum() / m_cls[b].sum().clamp(min=1)
            hub = nn.functional.huber_loss(lr_, y_reg[b], reduction="none", delta=1.0)
            loss_r = (hub * m_reg[b]).sum() / m_reg[b].sum().clamp(min=1)
            loss = loss_c + loss_r
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(b)
        if verbose:
            print(f"     epoch {ep + 1}/{epochs}  training loss {tot / len(S):.4f}")
    return net


@torch.no_grad()
def predict(net, seq, static, time, batch=4096):
    net.eval()
    pc, pr = [], []
    for i in range(0, len(seq), batch):
        lc, lr_ = net(torch.from_numpy(seq[i:i + batch]), torch.from_numpy(static[i:i + batch]),
                      torch.from_numpy(time[i:i + batch]))
        pc.append(torch.sigmoid(lc).numpy()); pr.append(lr_.numpy() * 50 + 150)
    return np.concatenate(pc), np.concatenate(pr)
