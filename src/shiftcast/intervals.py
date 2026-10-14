"""Multiplicative prediction intervals from backtest residuals, split by part of the day."""

from __future__ import annotations

import numpy as np
import pandas as pd

# slot index (0..47) -> part of day. Residual spread differs a lot between 4am and 6pm.
_EDGES = [0, 12, 20, 38, 48]  # 00:00-06:00, 06:00-10:00, 10:00-19:00, 19:00-24:00
N_PARTS = len(_EDGES) - 1


def part_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    slot = index.hour * 2 + index.minute // 30
    return np.searchsorted(_EDGES, np.asarray(slot), side="right") - 1


def log_residuals(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    return np.log1p(actual) - np.log1p(forecast)


class ResidualQuantiles:
    """Empirical quantiles of log(actual/forecast) per part of the day, fitted on a calibration set."""

    def __init__(self, min_per_part: int = 50):
        self.min_per_part = min_per_part
        self.resid_: dict[int, np.ndarray] = {}
        self.pooled_: np.ndarray | None = None

    def fit(self, index: pd.DatetimeIndex, actual: np.ndarray, forecast: np.ndarray):
        r = log_residuals(np.asarray(actual, float), np.asarray(forecast, float))
        parts = part_of_day(index)
        self.pooled_ = np.sort(r)
        for p in range(N_PARTS):
            rp = r[parts == p]
            self.resid_[p] = np.sort(rp if len(rp) >= self.min_per_part else r)
        return self

    def quantile(self, index: pd.DatetimeIndex, level: float) -> np.ndarray:
        """Per-slot additive shift on the log scale for the requested quantile level."""
        if self.pooled_ is None:
            raise RuntimeError("fit first")
        parts = part_of_day(index)
        q = {p: _conformal_quantile(self.resid_[p], level) for p in range(N_PARTS)}
        return np.array([q[p] for p in parts])

    def predict_quantile(self, index: pd.DatetimeIndex, forecast: np.ndarray, level: float):
        shift = self.quantile(index, level)
        return np.clip(np.expm1(np.log1p(forecast) + shift), 0, None)

    def interval(self, index: pd.DatetimeIndex, forecast: np.ndarray, coverage: float = 0.9):
        a = (1 - coverage) / 2
        return (self.predict_quantile(index, forecast, a), self.predict_quantile(index, forecast, 1 - a))


def _conformal_quantile(sorted_resid: np.ndarray, level: float) -> float:
    """Finite-sample corrected empirical quantile (split conformal style)."""
    n = len(sorted_resid)
    k = int(np.ceil((n + 1) * level))
    k = min(max(k, 1), n)
    return float(sorted_resid[k - 1])


def empirical_coverage(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(((actual >= lo) & (actual <= hi)).mean())
