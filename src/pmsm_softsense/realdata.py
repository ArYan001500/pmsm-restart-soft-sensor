"""Real-data access with a test-lock firewall (literature protocol v2, 2026-10-06).

Locked profiles (TNN generalization set G) are refused unless the caller passes
``final_evaluation=True`` AND the freeze file ``data/manifests/FREEZE_FINAL_EVALUATION.json`` exists.
Observation columns and evaluation targets are returned separately.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data/raw/electric_motor_temperature/measures_v2.csv"
CACHE = ROOT / "data/processed/measures_v2.pkl"
MANIFEST = ROOT / "data/manifests/profile_splits_literature_v2.json"
FREEZE = ROOT / "data/manifests/FREEZE_FINAL_EVALUATION.json"

OBS_COLS = ["u_q", "u_d", "i_d", "i_q", "motor_speed", "coolant", "ambient",
            "stator_winding", "stator_tooth", "stator_yoke"]
TARGET_COLS = ["pm"]
POLE_PAIRS = 8
TS = 0.5  # nominal sample period [s]


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _frame() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    df = pd.read_csv(CSV)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(CACHE)
    return df


def load(profile_ids, final_evaluation: bool = False):
    """Return dict pid -> (obs DataFrame, target DataFrame), row order preserved."""
    locked = set(manifest()["locked_profiles_never_loaded_before_freeze"])
    ids = [int(p) for p in profile_ids]
    hit = locked.intersection(ids)
    if hit and not (final_evaluation and FREEZE.exists()):
        raise PermissionError(f"Locked test profiles requested before freeze: {sorted(hit)}")
    df = _frame()
    out = {}
    for pid in ids:
        b = df[df["profile_id"] == pid].reset_index(drop=True)
        if b.empty:
            raise ValueError(f"profile {pid} not found")
        out[pid] = (b[OBS_COLS].copy(), b[TARGET_COLS].copy())
    return out


def protocol_ids(role: str) -> list[int]:
    p = manifest()["primary_protocol"]
    return {"train": p["train"], "V1": p["validation_V1"], "V2": p["validation_V2"],
            "val": p["validation_V1"] + p["validation_V2"]}[role]
