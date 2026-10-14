"""End-to-end run: backtest, choose, calibrate, schedule, price. Writes everything to outputs/."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt_mod
from .data import DEFAULT_PATH, describe, load_series, week_blocks
from .intervals import ResidualQuantiles, empirical_coverage
from .models import make_gbm, make_mstl, make_window_average, seasonal_naive
from .planner import future_week_index, plan_block
from .staffing import (
    CostModel,
    Schedule,
    constant_fleet,
    evaluate,
    required_cabs,
    solve_cpsat,
    solve_greedy,
)

N_VAL, N_TEST = 9, 8
LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SIMPLICITY = ["seasonal_naive", "window_median_4w", "mstl", "gbm_ratio"]
TOLERANCE = 0.003  # a simpler model within this much validation WAPE wins


def pick_model(val_summary: pd.DataFrame) -> str:
    best = val_summary.iloc[0]
    close = val_summary[val_summary.wape <= best.wape + TOLERANCE]
    return sorted(close.model, key=SIMPLICITY.index)[0]


def _outcome_row(o):
    return {k: float(v) for k, v in asdict(o).items()}


def run(
    out_dir: Path, data_path: Path = DEFAULT_PATH, quick: bool = False, assets_dir: Path | None = None
) -> dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = assets_dir or out_dir / "charts"
    assets_dir.mkdir(parents=True, exist_ok=True)
    s = load_series(data_path)
    val_blocks = week_blocks(s, N_VAL, N_TEST)
    test_blocks = week_blocks(s, N_TEST, 0)
    forecasters = {
        "seasonal_naive": seasonal_naive,
        "window_median_4w": make_window_average(4, robust=True),
        "mstl": make_mstl(),
        "gbm_ratio": make_gbm(),
    }
    bt = bt_mod.run_backtest(s, val_blocks + test_blocks, forecasters)
    bt["split"] = np.where(bt.block < N_VAL, "validation", "test")
    val_bt, test_bt = bt[bt.split == "validation"], bt[bt.split == "test"]
    val_sum, test_sum = bt_mod.summarise(val_bt), bt_mod.summarise(test_bt)
    chosen = pick_model(val_sum)

    boots = [
        bt_mod.paired_block_bootstrap(test_bt.assign(block=test_bt.block - N_VAL), chosen, other)
        for other in forecasters
        if other != chosen
    ]

    # intervals fitted on validation residuals only
    vm = val_bt[val_bt.model == chosen]
    tm = test_bt[test_bt.model == chosen].reset_index(drop=True)
    rq = ResidualQuantiles().fit(pd.DatetimeIndex(vm.ts), vm.actual.to_numpy(), vm.forecast.to_numpy())
    cover = {}
    for c in (0.8, 0.9):
        lo, hi = rq.interval(pd.DatetimeIndex(tm.ts), tm.forecast.to_numpy(), c)
        per_block = [
            empirical_coverage(g.actual.to_numpy(), lo[g.index], hi[g.index]) for _, g in tm.groupby("block")
        ]
        cover[str(c)] = {
            "test": empirical_coverage(tm.actual.to_numpy(), lo, hi),
            "per_week": [float(x) for x in per_block],
        }
    lo9, hi9 = rq.interval(pd.DatetimeIndex(tm.ts), tm.forecast.to_numpy(), 0.9)
    tm = tm.assign(lo90=lo9, hi90=hi9)

    # daily error on notable days
    daily = tm.set_index("ts")[["actual", "forecast"]].resample("D").sum()
    daily["ape"] = (daily.forecast - daily.actual).abs() / daily.actual
    daily["signed"] = (daily.forecast - daily.actual) / daily.actual
    notable = {
        "2014-12-24": "Christmas Eve",
        "2014-12-25": "Christmas Day",
        "2014-12-31": "New Year's Eve",
        "2015-01-01": "New Year's Day",
        "2015-01-19": "MLK Day",
        "2015-01-26": "blizzard, first day",
        "2015-01-27": "blizzard, second day",
    }
    notable = {d: lab for d, lab in notable.items() if pd.Timestamp(d) in daily.index}
    event_days = {
        d: {
            "label": lab,
            "actual": float(daily.loc[d, "actual"]),
            "forecast": float(daily.loc[d, "forecast"]),
            "signed_error": float(daily.loc[d, "signed"]),
        }
        for d, lab in notable.items()
    }
    normal = daily.drop(index=pd.to_datetime(list(notable)))
    event_days["_ordinary_days_median_ape"] = float(normal.ape.median())
    ordinary_mask = ~pd.DatetimeIndex(tm.ts).normalize().isin(pd.to_datetime(list(notable)))
    cover["0.9_ordinary_days"] = empirical_coverage(
        tm.actual.to_numpy()[ordinary_mask], lo9[ordinary_mask], hi9[ordinary_mask]
    )
    cover["0.9_event_days"] = (
        empirical_coverage(tm.actual.to_numpy()[~ordinary_mask], lo9[~ordinary_mask], hi9[~ordinary_mask])
        if (~ordinary_mask).any()
        else None
    )

    # staffing: schedules per (block, level), cached so any cost assumption can reprice them
    base = CostModel()
    cache: dict[tuple[str, int, object], Schedule] = {}
    levels = LEVELS[:3] if quick else LEVELS
    all_blocks = [("validation", i, b) for i, b in enumerate(val_blocks)] + [
        ("test", i, b) for i, b in enumerate(test_blocks)
    ]
    for split, i, block in all_blocks:
        sub = bt[
            (bt.model == chosen)
            & (bt.split == split)
            & (bt.block == (i if split == "validation" else i + N_VAL))
        ]
        point = sub.forecast.to_numpy()
        for lv in levels:
            sized = rq.predict_quantile(block, point, lv)
            cache[(split, i, lv)] = solve_cpsat(required_cabs(sized, base), base)
        cache[(split, i, "point")] = solve_cpsat(required_cabs(point, base), base)
        snaive = bt[
            (bt.model == "seasonal_naive")
            & (bt.split == split)
            & (bt.block == (i if split == "validation" else i + N_VAL))
        ].forecast.to_numpy()
        cache[(split, i, "snaive")] = solve_cpsat(required_cabs(snaive, base), base)

    val_curve = {}
    for lv in levels:
        parts = {"labour": 0.0, "unmet": 0.0}
        for i, block in enumerate(val_blocks):
            actual = s.loc[block].to_numpy()
            cov = cache[("validation", i, lv)]
            o = evaluate(cov, actual, base)
            parts["labour"] += o.labour_cost
            parts["unmet"] += o.unmet_cost
        val_curve[lv] = {**parts, "total": parts["labour"] + parts["unmet"]}
    best_level = min(val_curve, key=lambda k: val_curve[k]["total"])

    # test table
    def week_outcomes(key):
        rows = []
        for i, block in enumerate(test_blocks):
            actual = s.loc[block].to_numpy()
            cov = cache[("test", i, key)]
            rows.append(evaluate(cov, actual, base))
        return rows

    strategies = {
        "shipped": week_outcomes(best_level),
        "point_forecast_no_margin": week_outcomes("point"),
        "seasonal_naive_plan": week_outcomes("snaive"),
    }
    oracle, greedy, flat, solver_rows = [], [], [], []
    for i, block in enumerate(test_blocks):
        actual = s.loc[block].to_numpy()
        orc = solve_cpsat(required_cabs(actual, base), base)
        oracle.append(evaluate(orc, actual, base))
        sub = tm[tm.block == i + N_VAL]
        point = sub.forecast.to_numpy()
        sized = rq.predict_quantile(block, point, best_level)
        req = required_cabs(sized, base)
        cp = solve_cpsat(req, base)
        gr = solve_greedy(req, base)
        greedy.append(evaluate(gr, actual, base))
        solver_rows.append(
            {
                "cpsat_seconds": cp.solve_seconds,
                "cpsat_status": cp.status,
                "cpsat_cost": cp.cost,
                "greedy_cost": gr.cost,
                "greedy_over_cpsat": gr.cost / cp.cost - 1.0,
                "bound_gap": (cp.cost - cp.lower_bound) / cp.cost,
            }
        )
        prev_actual = s.loc[: block[0]].iloc[:-1].to_numpy()[-len(block) :]
        level_flat = int(required_cabs(prev_actual, base).max())
        flat.append(evaluate(constant_fleet(level_flat, len(block), base), actual, base))
    strategies["oracle_perfect_foresight"] = oracle
    strategies["greedy_on_shipped_demand"] = greedy
    strategies["flat_fleet_last_week_peak"] = flat

    def agg(rows):
        d = pd.DataFrame([asdict(r) for r in rows])
        tot = d[["labour_cost", "unmet", "unmet_cost", "total_cost", "idle_cab_slots", "served"]].sum()
        return {
            "labour_cost": float(tot.labour_cost),
            "unmet_trips": float(tot.unmet),
            "unmet_cost": float(tot.unmet_cost),
            "total_cost": float(tot.total_cost),
            "idle_cab_slots": float(tot.idle_cab_slots),
            "service_level": float(tot.served / (tot.served + tot.unmet)),
            "per_week_total": [float(x) for x in d.total_cost],
            "per_week_service": [float(x) for x in d.service_level],
        }

    strat_tab = {k: agg(v) for k, v in strategies.items()}

    # sensitivity: re-pick the level on validation per assumption, price the test weeks
    sens = []
    for pen in (8.0, 15.0, 25.0, 40.0):
        cm = CostModel(unmet_trip_cost=pen)
        curve = {}
        for lv in levels:
            tot = 0.0
            for i, block in enumerate(val_blocks):
                cov = cache[("validation", i, lv)]
                tot += evaluate(cov, s.loc[block].to_numpy(), cm).total_cost
            curve[lv] = tot
        lv_star = min(curve, key=curve.get)
        ship = sum(
            evaluate(cache[("test", i, lv_star)], s.loc[b].to_numpy(), cm).total_cost
            for i, b in enumerate(test_blocks)
        )
        naive = sum(
            evaluate(cache[("test", i, "snaive")], s.loc[b].to_numpy(), cm).total_cost
            for i, b in enumerate(test_blocks)
        )
        sens.append(
            {
                "unmet_trip_cost": pen,
                "level": lv_star,
                "shipped": ship,
                "seasonal_naive_plan": naive,
                "saving": naive - ship,
            }
        )

    # forward plan for the week after the data ends
    nxt = future_week_index(s)
    fwd_bt = bt[bt.model == chosen]
    rq_all = ResidualQuantiles().fit(
        pd.DatetimeIndex(fwd_bt.ts), fwd_bt.actual.to_numpy(), fwd_bt.forecast.to_numpy()
    )
    fwd = plan_block(s, nxt, forecasters[chosen], rq_all, best_level, base)
    pd.DataFrame(
        {
            "ts": nxt,
            "forecast": fwd.point,
            "sized_demand": fwd.sized_demand,
            "cabs_required": fwd.required,
            "cabs_scheduled": fwd.schedule.coverage,
        }
    ).to_csv(out_dir / "next_week_plan.csv", index=False)

    ship = strat_tab["shipped"]
    regret_by_week = [
        a / b - 1
        for a, b in zip(
            ship["per_week_total"], strat_tab["oracle_perfect_foresight"]["per_week_total"], strict=True
        )
    ]
    snv = strat_tab["seasonal_naive_plan"]
    orc = strat_tab["oracle_perfect_foresight"]
    metrics = {
        "data": describe(s),
        "split": {
            "validation_weeks": N_VAL,
            "test_weeks": N_TEST,
            "validation_span": [str(val_blocks[0][0]), str(val_blocks[-1][-1])],
            "test_span": [str(test_blocks[0][0]), str(test_blocks[-1][-1])],
        },
        "forecast": {
            "validation": val_sum.round(5).to_dict("records"),
            "test": test_sum.round(5).to_dict("records"),
            "chosen": chosen,
            "selection_rule": f"lowest validation WAPE; a simpler model within {TOLERANCE} WAPE wins",
            "bootstrap_test_wape_diff": boots,
        },
        "intervals": cover,
        "event_days": event_days,
        "staffing": {
            "assumptions": asdict(base),
            "validation_curve": {str(k): v for k, v in val_curve.items()},
            "chosen_level": best_level,
            "test_strategies": strat_tab,
            "saving_vs_seasonal_naive": snv["total_cost"] - ship["total_cost"],
            "saving_pct_vs_seasonal_naive": 1 - ship["total_cost"] / snv["total_cost"],
            "regret_by_week": [float(x) for x in regret_by_week],
            "regret_vs_oracle": ship["total_cost"] / orc["total_cost"] - 1,
            "solver": {
                "median_cpsat_seconds": float(np.median([r["cpsat_seconds"] for r in solver_rows])),
                "all_optimal": all(r["cpsat_status"] == "OPTIMAL" for r in solver_rows),
                "greedy_over_cpsat_pct": float(np.mean([r["greedy_over_cpsat"] for r in solver_rows])),
            },
            "sensitivity": sens,
            "next_week_peak_cabs": int(fwd.schedule.coverage.max()),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    bt.to_csv(out_dir / "backtest.csv", index=False)
    tm.to_csv(out_dir / "test_forecasts.csv", index=False)
    from .charts import make_all

    make_all(assets_dir, metrics, s, bt, tm, test_blocks, cache, best_level, rq_all, fwd)
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument(
        "--assets", type=Path, default=None, help="where to write charts (default outputs/charts)"
    )
    ap.add_argument("--quick", action="store_true", help="fewer safety levels, for smoke runs")
    a = ap.parse_args(argv)
    m = run(a.out, a.data, a.quick, a.assets)
    print(
        json.dumps(
            {
                "chosen": m["forecast"]["chosen"],
                "level": m["staffing"]["chosen_level"],
                "saving_pct": round(100 * m["staffing"]["saving_pct_vs_seasonal_naive"], 2),
            }
        )
    )


if __name__ == "__main__":
    main()
