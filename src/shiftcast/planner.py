"""One function the pipeline, the UI and the tests all call: forecast a week, size it, schedule it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import SLOTS_PER_WEEK
from .intervals import ResidualQuantiles
from .models import Forecaster
from .staffing import CostModel, Outcome, Schedule, evaluate, required_cabs, solve_cpsat


@dataclass
class WeekPlan:
    index: pd.DatetimeIndex
    point: np.ndarray
    sized_demand: np.ndarray
    required: np.ndarray
    schedule: Schedule
    outcome: Outcome | None


def future_week_index(series: pd.Series) -> pd.DatetimeIndex:
    """The 336 half hours that follow the last observation."""
    start = series.index[-1] + pd.Timedelta("30min")
    return pd.date_range(start, periods=SLOTS_PER_WEEK, freq="30min")


def plan_block(
    history: pd.Series,
    block: pd.DatetimeIndex,
    forecaster: Forecaster,
    rq: ResidualQuantiles | None,
    level: float | None,
    cost: CostModel,
    actual: np.ndarray | None = None,
    time_limit: float = 20.0,
) -> WeekPlan:
    """Forecast `block` from `history`, size cabs at `level` of the residual distribution, schedule."""
    point = np.asarray(forecaster(history, block), dtype=float)
    if rq is not None and level is not None:
        sized = rq.predict_quantile(block, point, level)
    else:
        sized = point
    req = required_cabs(sized, cost)
    sched = solve_cpsat(req, cost, time_limit=time_limit)
    outcome = evaluate(sched, np.asarray(actual, float), cost) if actual is not None else None
    return WeekPlan(block, point, sized, req, sched, outcome)
