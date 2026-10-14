"""Calendar features, including US holidays observed in New York."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def holiday_flags(index: pd.DatetimeIndex) -> pd.Series:
    """1 on a federal or New York holiday, and 0.5 on the eve of one (the evening runs hot)."""
    years = sorted(set(index.year) | {index.year.min() - 1, index.year.max() + 1})
    cal = holidays.US(state="NY", years=years)
    days = pd.Series(index.normalize(), index=index)
    is_hol = days.isin([pd.Timestamp(d) for d in cal]).astype(float)
    nxt = (days + pd.Timedelta("1D")).isin([pd.Timestamp(d) for d in cal]).astype(float)
    return np.maximum(is_hol, 0.5 * nxt).rename("holiday")


def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    slot = index.hour * 2 + index.minute // 30
    return pd.DataFrame(
        {
            "slot": slot.to_numpy(),
            "dow": index.dayofweek.to_numpy(),
            "holiday": holiday_flags(index).to_numpy(),
        },
        index=index,
    )
