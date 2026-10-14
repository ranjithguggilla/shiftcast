"""Loading and validating the half-hourly trip series, and cutting it into week blocks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SLOTS_PER_DAY = 48
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "nyc_taxi.csv"


def load_series(path: str | Path = DEFAULT_PATH) -> pd.Series:
    """Read the csv and check it is one regular 30 minute grid with no holes."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    s = df.set_index("timestamp")["value"].astype(float).sort_index()
    s.name = "trips"
    validate_grid(s)
    return s


def validate_grid(s: pd.Series) -> None:
    if s.index.has_duplicates:
        raise ValueError("duplicate timestamps")
    steps = s.index.to_series().diff().dropna().unique()
    if len(steps) != 1 or steps[0] != pd.Timedelta(minutes=30):
        raise ValueError(f"series is not a regular 30 minute grid: {steps[:3]}")
    if s.isna().any():
        raise ValueError("missing values")
    if (s < 0).any():
        raise ValueError("negative counts")


def week_blocks(s: pd.Series, n_blocks: int, end_offset_weeks: int = 0) -> list[pd.DatetimeIndex]:
    """Return n_blocks consecutive 7 day windows ending `end_offset_weeks` before the last slot.

    Blocks are end-aligned so the final one finishes on the last observed half hour.
    """
    idx = s.index
    stop = len(idx) - end_offset_weeks * SLOTS_PER_WEEK
    blocks = []
    for k in range(n_blocks, 0, -1):
        lo = stop - k * SLOTS_PER_WEEK
        if lo < 0:
            raise ValueError("not enough history for the requested number of blocks")
        blocks.append(idx[lo : lo + SLOTS_PER_WEEK])
    return blocks


def daily_totals(s: pd.Series) -> pd.Series:
    return s.resample("D").sum()


def describe(s: pd.Series) -> dict:
    d = daily_totals(s)
    return {
        "n_slots": int(len(s)),
        "start": str(s.index[0]),
        "end": str(s.index[-1]),
        "mean_per_slot": float(np.round(s.mean(), 1)),
        "max_per_slot": float(s.max()),
        "min_daily_total": float(d.min()),
        "min_daily_date": str(d.idxmin().date()),
    }
