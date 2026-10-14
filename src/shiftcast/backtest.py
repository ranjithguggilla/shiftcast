"""Rolling-origin backtest over whole-week blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import Forecaster


def run_backtest(
    series: pd.Series, blocks: list[pd.DatetimeIndex], forecasters: dict[str, Forecaster]
) -> pd.DataFrame:
    """Forecast each block from the history before it. Returns a long frame of actual/forecast."""
    rows = []
    for b, block in enumerate(blocks):
        history = series.loc[: block[0]].iloc[:-1]
        actual = series.loc[block].to_numpy()
        for name, fn in forecasters.items():
            pred = np.asarray(fn(history, block), dtype=float)
            if pred.shape != actual.shape or not np.isfinite(pred).all():
                raise ValueError(f"{name}: bad forecast for block {b}")
            rows.append(
                pd.DataFrame({"ts": block, "block": b, "model": name, "actual": actual, "forecast": pred})
            )
    return pd.concat(rows, ignore_index=True)


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.abs(actual - forecast).sum() / actual.sum())


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float((forecast.sum() - actual.sum()) / actual.sum())


def mase(actual: np.ndarray, forecast: np.ndarray, scale: float) -> float:
    return float(np.abs(actual - forecast).mean() / scale)


def summarise(bt: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, g in bt.groupby("model", sort=False):
        per_block = g.groupby("block").apply(
            lambda x: wape(x["actual"].to_numpy(), x["forecast"].to_numpy()), include_groups=False
        )
        out.append(
            {
                "model": name,
                "wape": wape(g["actual"].to_numpy(), g["forecast"].to_numpy()),
                "bias": bias(g["actual"].to_numpy(), g["forecast"].to_numpy()),
                "rmse": float(np.sqrt(((g["actual"] - g["forecast"]) ** 2).mean())),
                "worst_block_wape": float(per_block.max()),
                "best_block_wape": float(per_block.min()),
            }
        )
    return pd.DataFrame(out).sort_values("wape").reset_index(drop=True)


def paired_block_bootstrap(bt: pd.DataFrame, a: str, b: str, n_boot: int = 4000, seed: int = 0) -> dict:
    """Bootstrap over blocks for WAPE(a) - WAPE(b). Negative means a is better."""
    rng = np.random.default_rng(seed)
    ga = bt[bt.model == a].sort_values(["block", "ts"])
    gb = bt[bt.model == b].sort_values(["block", "ts"])
    blocks = sorted(ga["block"].unique())

    def stats(g):
        return {k: (np.abs(x.actual - x.forecast).sum(), x.actual.sum()) for k, x in g.groupby("block")}

    sa, sb = stats(ga), stats(gb)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        wa = sum(sa[k][0] for k in pick) / sum(sa[k][1] for k in pick)
        wb = sum(sb[k][0] for k in pick) / sum(sb[k][1] for k in pick)
        diffs.append(wa - wb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = wape(ga.actual.to_numpy(), ga.forecast.to_numpy()) - wape(
        gb.actual.to_numpy(), gb.forecast.to_numpy()
    )
    return {"a": a, "b": b, "diff": float(point), "ci_low": float(lo), "ci_high": float(hi)}
