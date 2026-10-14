import numpy as np
import pandas as pd
import pytest

from shiftcast.data import SLOTS_PER_DAY, SLOTS_PER_WEEK


@pytest.fixture(scope="session")
def toy_series():
    """Ten weeks of a clean daily and weekly pattern with mild noise."""
    rng = np.random.default_rng(3)
    n = 10 * SLOTS_PER_WEEK
    idx = pd.date_range("2014-09-01", periods=n, freq="30min")
    slot = np.arange(n) % SLOTS_PER_DAY
    dow = (np.arange(n) // SLOTS_PER_DAY) % 7
    base = 8000 + 6000 * np.sin(2 * np.pi * (slot - 12) / SLOTS_PER_DAY) + 1500 * (dow >= 5)
    return pd.Series(base + rng.normal(0, 150, n), index=idx, name="trips").clip(lower=0).round()
