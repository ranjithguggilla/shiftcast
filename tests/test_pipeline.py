import json

import numpy as np
import pandas as pd

from shiftcast.data import SLOTS_PER_DAY
from shiftcast.pipeline import SIMPLICITY, TOLERANCE, pick_model, run


def test_pick_model_prefers_the_simpler_one_inside_tolerance():
    tab = pd.DataFrame(
        {"model": ["gbm_ratio", "seasonal_naive", "mstl"], "wape": [0.060, 0.060 + TOLERANCE / 2, 0.09]}
    )
    assert pick_model(tab) == "seasonal_naive"


def test_pick_model_keeps_a_clear_winner():
    tab = pd.DataFrame({"model": ["gbm_ratio", "seasonal_naive"], "wape": [0.05, 0.08]})
    assert pick_model(tab) == "gbm_ratio"
    assert SIMPLICITY.index("seasonal_naive") < SIMPLICITY.index("gbm_ratio")


def test_full_run_on_a_synthetic_series(tmp_path):
    rng = np.random.default_rng(5)
    n = 24 * 7 * SLOTS_PER_DAY // 7 * 7  # 24 weeks
    n = 24 * 336
    idx = pd.date_range("2014-07-07", periods=n, freq="30min")
    slot = np.arange(n) % SLOTS_PER_DAY
    dow = (np.arange(n) // SLOTS_PER_DAY) % 7
    vals = 9000 + 6000 * np.sin(2 * np.pi * (slot - 12) / SLOTS_PER_DAY) + 1200 * (dow >= 5)
    vals = np.clip(vals * np.exp(rng.normal(0, 0.05, n)), 0, None).round()
    csv = tmp_path / "toy.csv"
    pd.DataFrame({"timestamp": idx, "value": vals}).to_csv(csv, index=False)

    out = tmp_path / "out"
    m = run(out, csv, quick=True, assets_dir=tmp_path / "charts")

    saved = json.loads((out / "metrics.json").read_text())
    assert saved["forecast"]["chosen"] in SIMPLICITY
    assert saved["staffing"]["solver"]["all_optimal"]
    assert len(saved["staffing"]["test_strategies"]["shipped"]["per_week_total"]) == 8
    assert (out / "next_week_plan.csv").exists()
    assert (tmp_path / "charts" / "cost_by_strategy.png").stat().st_size > 5_000
    # perfect foresight can never cost more than any real plan
    st = m["staffing"]["test_strategies"]
    assert st["oracle_perfect_foresight"]["total_cost"] <= st["shipped"]["total_cost"]
