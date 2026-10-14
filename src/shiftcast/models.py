"""Week-ahead forecasters.

Every forecaster has the same signature: it gets the history up to the forecast origin and the
future timestamps, and returns one non-negative value per future timestamp. Nothing after the
origin is ever visible, so a week-ahead horizon can only use lags of 336 slots or more.
"""

from __future__ import annotations

from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd

from .calendar import calendar_frame
from .data import SLOTS_PER_WEEK

Forecaster = Callable[[pd.Series, pd.DatetimeIndex], np.ndarray]


def _lag_matrix(values: np.ndarray, positions: np.ndarray, weeks: int) -> np.ndarray:
    """Values at positions - k*336 for k = 1..weeks. Positions may point past the history end."""
    cols = [values[positions - k * SLOTS_PER_WEEK] for k in range(1, weeks + 1)]
    return np.column_stack(cols)


def seasonal_naive(history: pd.Series, future: pd.DatetimeIndex) -> np.ndarray:
    """Same half hour, one week earlier."""
    h = len(future)
    if h > SLOTS_PER_WEEK:
        raise ValueError("horizon longer than one week is not supported")
    return history.to_numpy()[-SLOTS_PER_WEEK:][:h].copy()


def make_window_average(weeks: int = 4, robust: bool = True) -> Forecaster:
    """Mean or median of the same half hour over the last `weeks` weeks."""

    def forecast(history: pd.Series, future: pd.DatetimeIndex) -> np.ndarray:
        v = history.to_numpy()
        blocks = np.stack(
            [v[len(v) - k * SLOTS_PER_WEEK : len(v) - (k - 1) * SLOTS_PER_WEEK] for k in range(1, weeks + 1)]
        )
        agg = np.median(blocks, axis=0) if robust else blocks.mean(axis=0)
        return agg[: len(future)]

    return forecast


def make_mstl(history_weeks: int = 12) -> Forecaster:
    """MSTL with daily and weekly seasonality; an exponential smoothing model carries the trend."""
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL

    def forecast(history: pd.Series, future: pd.DatetimeIndex) -> np.ndarray:
        h = history.iloc[-history_weeks * SLOTS_PER_WEEK :]
        df = pd.DataFrame({"unique_id": "city", "ds": h.index, "y": np.log1p(h.to_numpy())})
        sf = StatsForecast(models=[MSTL(season_length=[48, 336])], freq="30min", n_jobs=1)
        out = sf.forecast(df=df, h=len(future))
        return np.clip(np.expm1(out["MSTL"].to_numpy()), 0, None)

    return forecast


def _gbm_features(
    values: np.ndarray,
    positions: np.ndarray,
    index: pd.DatetimeIndex,
    hol_all: np.ndarray,
    cal: pd.DataFrame,
    weeks: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    lags = _lag_matrix(values, positions, weeks)
    base = lags.mean(axis=1)
    feats = pd.DataFrame(
        {
            "slot": cal["slot"].to_numpy(),
            "dow": cal["dow"].to_numpy(),
            "holiday": cal["holiday"].to_numpy(),
            "log_lag1": np.log1p(lags[:, 0]),
            "log_base": np.log1p(base),
            "lag1_over_base": np.log1p(lags[:, 0]) - np.log1p(base),
            "hol_lag1": hol_all[positions - SLOTS_PER_WEEK],
            "hol_lag2": hol_all[positions - 2 * SLOTS_PER_WEEK],
        }
    )
    return feats, base


def make_gbm(
    weeks: int = 4, n_estimators: int = 300, learning_rate: float = 0.04, num_leaves: int = 15, seed: int = 7
) -> Forecaster:
    """Gradient boosting on the log ratio between a slot and its trailing same-slot average.

    Predicting a ratio means the model learns shape (time of day, weekday, holiday effects) while
    the level comes from the recent weeks, which is where a short, drifting series needs it.
    """

    def forecast(history: pd.Series, future: pd.DatetimeIndex) -> np.ndarray:
        full_index = history.index.append(future)
        cal_all = calendar_frame(full_index)
        hol_all = cal_all["holiday"].to_numpy()
        values = np.concatenate([history.to_numpy(), np.zeros(len(future))])
        n = len(history)
        train_pos = np.arange(weeks * SLOTS_PER_WEEK, n)
        x_tr, base_tr = _gbm_features(values, train_pos, full_index, hol_all, cal_all.iloc[train_pos], weeks)
        y_tr = np.log1p(values[train_pos]) - np.log1p(base_tr)
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=seed,
            verbose=-1,
            n_jobs=1,
        )
        model.fit(x_tr, y_tr)
        fut_pos = np.arange(n, n + len(future))
        x_f, base_f = _gbm_features(values, fut_pos, full_index, hol_all, cal_all.iloc[fut_pos], weeks)
        pred = np.expm1(model.predict(x_f) + np.log1p(base_f))
        return np.clip(pred, 0, None)

    return forecast


def registry() -> dict[str, Forecaster]:
    return {
        "seasonal_naive": seasonal_naive,
        "window_median_4w": make_window_average(4, robust=True),
        "mstl": make_mstl(),
        "gbm_ratio": make_gbm(),
    }
