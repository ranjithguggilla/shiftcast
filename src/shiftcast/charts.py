"""README and report charts. Every figure is drawn from the run's own numbers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, GREY, BLUE, ORANGE, RED, GREEN = "#1f2933", "#9aa5b1", "#2b6cb0", "#dd6b20", "#c53030", "#2f855a"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GREY,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.dpi": 130,
        "savefig.bbox": "tight",
    }
)


def _save(fig, path: Path):
    fig.savefig(path)
    plt.close(fig)


def chart_model_comparison(path: Path, metrics: dict):
    val = {r["model"]: r["wape"] for r in metrics["forecast"]["validation"]}
    test = {r["model"]: r["wape"] for r in metrics["forecast"]["test"]}
    names = list(val)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x - 0.2, [val[n] * 100 for n in names], 0.4, color=BLUE, label="validation (9 weeks)")
    ax.bar(x + 0.2, [test[n] * 100 for n in names], 0.4, color=ORANGE, label="test (8 weeks)")
    for i, n in enumerate(names):
        ax.text(i - 0.2, val[n] * 100 + 0.3, f"{val[n] * 100:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.2, test[n] * 100 + 0.3, f"{test[n] * 100:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, names)
    ax.set_ylim(0, max(test.values()) * 100 * 1.3)
    ax.set_ylabel("week-ahead WAPE (%)")
    ax.set_title("Week-ahead error by model")
    ax.legend(frameon=False)
    _save(fig, path)


def chart_test_forecast(path: Path, tm: pd.DataFrame, notable: dict):
    ts = pd.DatetimeIndex(tm.ts)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(ts, tm.lo90, tm.hi90, color=BLUE, alpha=0.15, label="90% band (fitted on validation)")
    ax.plot(ts, tm.actual, color=INK, lw=0.8, label="actual")
    ax.plot(ts, tm.forecast, color=ORANGE, lw=0.8, label="forecast")
    for d, label in notable.items():
        if label in {"Christmas Day", "New Year's Day", "blizzard, second day"}:
            ax.annotate(
                label,
                (pd.Timestamp(d) + pd.Timedelta(hours=12), tm.actual.max() * 0.97),
                ha="center",
                fontsize=8,
                color=RED,
            )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel("trips per half hour")
    ax.set_title("Eight test weeks, forecast made one week ahead")
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    _save(fig, path)


def chart_blizzard(path: Path, s: pd.Series, tm: pd.DataFrame, block, schedule_cov: np.ndarray):
    sub = tm[tm.block == tm.block.max()]
    ts = pd.DatetimeIndex(sub.ts)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(ts, sub.actual, color=INK, lw=1.2, label="actual trips")
    ax.plot(ts, sub.forecast, color=ORANGE, lw=1.2, label="forecast")
    ax.plot(ts, schedule_cov, color=BLUE, lw=1.0, ls="--", label="cabs scheduled (1 trip per cab per slot)")
    ax.set_ylabel("per half hour")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax.set_title("Final test week: the schedule was built before the storm")
    ax.legend(frameon=False)
    _save(fig, path)


def chart_level_curve(path: Path, curve: dict, chosen: float):
    lv = sorted(curve, key=float)
    lab = [curve[k]["labour"] / 1e6 for k in lv]
    unm = [curve[k]["unmet"] / 1e6 for k in lv]
    x = np.arange(len(lv))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x, lab, color=BLUE, label="shift labour")
    ax.bar(x, unm, bottom=lab, color=RED, label="unmet-trip cost")
    for i in range(len(lv)):
        ax.text(i, lab[i] + unm[i] + 4, f"{(lab[i] + unm[i]):.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{float(k):.2f}" for k in lv])
    ax.set_xlabel("quantile of the residual distribution used to size demand")
    ax.set_ylabel("nine validation weeks, $ millions")
    ax.set_ylim(min(a + b for a, b in zip(lab, unm, strict=True)) * 0.9, None)
    ax.set_title(f"Safety level chosen on validation: {chosen:.2f}")
    ax.legend(frameon=False)
    _save(fig, path)


def chart_strategies(path: Path, metrics: dict):
    st = metrics["staffing"]["test_strategies"]
    order = [
        "oracle_perfect_foresight",
        "shipped",
        "point_forecast_no_margin",
        "seasonal_naive_plan",
        "greedy_on_shipped_demand",
        "flat_fleet_last_week_peak",
    ]
    labels = [
        "perfect\nforesight",
        "shipped",
        "point\nforecast",
        "seasonal\nnaive",
        "greedy\nscheduler",
        "flat fleet",
    ]
    lab = [st[k]["labour_cost"] / 1e6 for k in order]
    unm = [st[k]["unmet_cost"] / 1e6 for k in order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x, lab, color=BLUE, label="shift labour")
    ax.bar(x, unm, bottom=lab, color=RED, label="unmet-trip cost")
    for i in range(len(order)):
        ax.text(i, lab[i] + unm[i] + 6, f"{lab[i] + unm[i]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("eight test weeks, $ millions")
    ax.set_title("Total cost by planning approach")
    ax.legend(frameon=False)
    _save(fig, path)


def chart_week_regret(path: Path, metrics: dict, test_blocks):
    reg = [r * 100 for r in metrics["staffing"]["regret_by_week"]]
    labels = [b[0].strftime("%b %d") for b in test_blocks]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colours = [RED if r > np.median(reg) * 1.6 else BLUE for r in reg]
    ax.bar(labels, reg, color=colours)
    for i, r in enumerate(reg):
        ax.text(i, r + 0.6, f"{r:.0f}%", ha="center", fontsize=8)
    ax.set_ylabel("cost above perfect foresight (%)")
    ax.set_title("Where the shipped plan loses: week starting")
    _save(fig, path)


def chart_heatmap(path: Path, fwd):
    cov = pd.Series(fwd.schedule.coverage, index=fwd.index)
    grid = cov.groupby([cov.index.dayofweek, cov.index.hour]).max().unstack() / 1000
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(grid.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_title("Cabs to have on the road in the week after the data ends (thousands)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, path)


def make_all(assets: Path, metrics, s, bt, tm, test_blocks, cache, level, rq_all, fwd):
    assets.mkdir(parents=True, exist_ok=True)
    notable = {d: v["label"] for d, v in metrics["event_days"].items() if not d.startswith("_")}
    chart_model_comparison(assets / "model_comparison.png", metrics)
    chart_test_forecast(assets / "test_forecast.png", tm, notable)
    last = len(test_blocks) - 1
    chart_blizzard(assets / "storm_week.png", s, tm, test_blocks[-1], cache[("test", last, level)].coverage)
    chart_level_curve(assets / "safety_level.png", metrics["staffing"]["validation_curve"], level)
    chart_strategies(assets / "cost_by_strategy.png", metrics)
    chart_week_regret(assets / "regret_by_week.png", metrics, test_blocks)
    chart_heatmap(assets / "next_week_cabs.png", fwd)
