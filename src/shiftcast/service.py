"""The scoring function behind the UI: pick a week, a safety level and costs, get a schedule."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .backtest import run_backtest
from .data import load_series, week_blocks
from .intervals import ResidualQuantiles
from .models import make_window_average
from .planner import future_week_index, plan_block
from .staffing import CostModel

NEXT_WEEK = "next week (after the data ends)"


class Planner:
    """Holds the series and a residual calibrator fitted on the nine validation weeks."""

    def __init__(self, series: pd.Series | None = None, n_val: int = 9, n_test: int = 8):
        self.series = series if series is not None else load_series()
        self.forecaster = make_window_average(4, robust=True)
        self.test_blocks = week_blocks(self.series, n_test, 0)
        val_blocks = week_blocks(self.series, n_val, n_test)
        bt = run_backtest(self.series, val_blocks, {"w": self.forecaster})
        self.rq = ResidualQuantiles().fit(
            pd.DatetimeIndex(bt.ts), bt.actual.to_numpy(), bt.forecast.to_numpy()
        )

    def week_choices(self) -> list[str]:
        return [b[0].strftime("%Y-%m-%d") for b in self.test_blocks] + [NEXT_WEEK]

    def score_week(self, week: str, level: float = 0.5, cost: CostModel | None = None) -> dict:
        cost = cost or CostModel()
        if week == NEXT_WEEK:
            block = future_week_index(self.series)
            history, actual = self.series, None
        else:
            matches = [b for b in self.test_blocks if b[0].strftime("%Y-%m-%d") == week]
            if not matches:
                raise ValueError(f"unknown week {week!r}")
            block = matches[0]
            history = self.series.loc[: block[0]].iloc[:-1]
            actual = self.series.loc[block].to_numpy()
        plan = plan_block(history, block, self.forecaster, self.rq, level, cost, actual)
        frame = pd.DataFrame(
            {
                "forecast": plan.point.round(0),
                "sized_demand": plan.sized_demand.round(0),
                "cabs_required": plan.required,
                "cabs_scheduled": plan.schedule.coverage,
            },
            index=block,
        )
        if actual is not None:
            frame["actual"] = actual
        summary = {
            "week_start": str(block[0].date()),
            "peak_cabs": int(plan.schedule.coverage.max()),
            "labour_cost": round(plan.schedule.cost, 0),
            "solver_status": plan.schedule.status,
            "shift_starts": {int(k): int(v.sum()) for k, v in plan.schedule.starts.items()},
            "assumptions": asdict(cost),
        }
        if plan.outcome is not None:
            summary |= {
                "unmet_trips": round(plan.outcome.unmet, 0),
                "service_level": round(plan.outcome.service_level, 4),
                "total_cost": round(plan.outcome.total_cost, 0),
            }
        return {"summary": summary, "frame": frame}


def _fmt(summary: dict) -> str:
    lines = [f"Week starting {summary['week_start']}", f"Peak cabs on the road: {summary['peak_cabs']:,}"]
    lines.append(f"Shift labour: ${summary['labour_cost']:,.0f}  (solver: {summary['solver_status']})")
    lines.append(
        "Shifts started: " + ", ".join(f"{k} slots x {v:,}" for k, v in summary["shift_starts"].items())
    )
    if "unmet_trips" in summary:
        lines.append(
            f"Trips nobody could take: {summary['unmet_trips']:,.0f} "
            f"(service level {summary['service_level']:.1%})"
        )
        lines.append(f"Total cost incl. unmet trips: ${summary['total_cost']:,.0f}")
    else:
        lines.append("No actuals exist for this week, so only the plan is shown.")
    return "\n".join(lines)


def build_app():
    import gradio as gr
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    planner = Planner()

    def run(week, level, unmet_cost, cab_cost, rate):
        res = planner.score_week(
            week,
            level,
            CostModel(trips_per_cab_slot=rate, cab_slot_cost=cab_cost, unmet_trip_cost=unmet_cost),
        )
        f = res["frame"]
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.plot(f.index, f["cabs_scheduled"] * rate, label="capacity (trips)", color="#2b6cb0")
        ax.plot(f.index, f["sized_demand"], label="sized demand", color="#dd6b20", lw=0.8)
        if "actual" in f:
            ax.plot(f.index, f["actual"], label="actual", color="#1f2933", lw=0.8)
        ax.legend(frameon=False)
        ax.set_ylabel("trips per half hour")
        fig.autofmt_xdate()
        return _fmt(res["summary"]), fig

    with gr.Blocks(title="shiftcast") as demo:
        gr.Markdown("# shiftcast\nPick a week, set the costs, and see the shift plan the optimiser builds.")
        week = gr.Dropdown(planner.week_choices(), value=planner.week_choices()[0], label="Week")
        level = gr.Slider(
            0.5, 0.95, value=0.5, step=0.05, label="Safety level (quantile of past forecast errors)"
        )
        unmet = gr.Slider(5, 60, value=15, step=1, label="Cost of an unmet trip ($)")
        cab = gr.Slider(4, 20, value=9, step=0.5, label="Cost of a cab for 30 minutes ($)")
        rate = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Trips one cab completes per 30 minutes")
        out_text = gr.Textbox(label="Plan", lines=6)
        out_plot = gr.Plot()
        gr.Button("Plan").click(run, [week, level, unmet, cab, rate], [out_text, out_plot])
    return demo


if __name__ == "__main__":
    build_app().launch()
