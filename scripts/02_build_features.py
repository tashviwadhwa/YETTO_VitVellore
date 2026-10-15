"""Day 2, step 1: fuse the EHR and wearable streams into one training table.

Run from the project root:
    python scripts/02_build_features.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glucotwin import config as C
from glucotwin.features import build_dataset


def main():
    t0 = time.time()
    print("Fusing EHR + CGM + smartwatch + app diary ...")
    df, groups = build_dataset()
    out = C.DATA_DIR / "processed"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "fused_dataset.csv.gz", index=False)
    (out / "feature_groups.json").write_text(json.dumps(groups, indent=2))

    sp = df[df.spike_eligible == 1]
    hy = df[df.hypo_eligible == 1]
    print(f"Done in {time.time() - t0:.0f} s")
    print(f"Rows (patient-moments): {len(df):,} | features: {sum(len(v) for v in groups.values())}")
    for k, v in groups.items():
        print(f"  {k:7s} {len(v):3d} features")
    print(f"Spike-in-2h rate: {sp.spike_2h.mean():.1%} of {len(sp):,} moments")
    print(f"Hypo-in-2h rate:  {hy.hypo_2h.mean():.2%} of {len(hy):,} moments")


if __name__ == "__main__":
    main()
