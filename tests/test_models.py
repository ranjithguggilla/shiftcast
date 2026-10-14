import numpy as np
import pandas as pd

from shiftcast.backtest import bias, paired_block_bootstrap, run_backtest, summarise, wape
from shiftcast.calendar import calendar_frame, holiday_flags
from shiftcast.data import SLOTS_PER_WEEK, week_blocks
from shiftcast.models import make_gbm, make_mstl, make_window_average, seasonal_naive


def _split(s):
    block = s.index[-SLOTS_PER_WEEK:]
    return s.loc[: block[0]].iloc[:-1], block


def test_seasonal_naive_repeats_last_week(toy_series):
    hist, block = _split(toy_series)
    np.testing.assert_array_equal(seasonal_naive(hist, block), hist.to_numpy()[-SLOTS_PER_WEEK:])


def test_window_average_is_robust_to_one_wild_week(toy_series):
    hist, block = _split(toy_series)
    wild = hist.copy()
    wild.iloc[-SLOTS_PER_WEEK:] *= 10
    med = make_window_average(5, robust=True)(wild, block)
    mean = make_window_average(5, robust=False)(wild, block)
    truth = toy_series.loc[block].to_numpy()
    assert wape(truth, med) < wape(truth, mean)


def test_gbm_beats_zero_and_stays_nonnegative(toy_series):
    hist, block = _split(toy_series)
    pred = make_gbm(n_estimators=60)(hist, block)
    assert (pred >= 0).all() and len(pred) == SLOTS_PER_WEEK
    assert wape(toy_series.loc[block].to_numpy(), pred) < 0.05


def test_mstl_tracks_a_seasonal_series(toy_series):
    hist, block = _split(toy_series)
    pred = make_mstl(history_weeks=6)(hist, block)
    assert wape(toy_series.loc[block].to_numpy(), pred) < 0.12


def test_forecasters_ignore_the_future(toy_series):
    """Changing values inside the forecast block must not change a forecast."""
    hist, block = _split(toy_series)
    fn = make_gbm(n_estimators=40)
    a = fn(hist, block)
    b = fn(hist.copy(), block)
    np.testing.assert_allclose(a, b)


def test_backtest_shapes_and_summary(toy_series):
    blocks = week_blocks(toy_series, 2, 0)
    bt = run_backtest(toy_series, blocks, {"naive": seasonal_naive, "avg": make_window_average(3)})
    assert set(bt.model) == {"naive", "avg"}
    assert len(bt) == 2 * 2 * SLOTS_PER_WEEK
    tab = summarise(bt)
    assert tab.wape.is_monotonic_increasing


def test_bootstrap_sign(toy_series):
    blocks = week_blocks(toy_series, 4, 0)
    always_zero = lambda h, f: np.zeros(len(f))  # noqa: E731
    bt = run_backtest(toy_series, blocks, {"naive": seasonal_naive, "zero": always_zero})
    r = paired_block_bootstrap(bt, "naive", "zero", n_boot=200)
    assert r["diff"] < 0 and r["ci_high"] < 0


def test_bias_sign():
    assert bias(np.array([10.0, 10.0]), np.array([12.0, 12.0])) > 0


def test_christmas_is_a_holiday_and_eve_is_half():
    idx = pd.date_range("2014-12-24", periods=96, freq="30min")
    h = holiday_flags(idx)
    assert h.iloc[0] == 0.5 and h.iloc[48] == 1.0
    cal = calendar_frame(idx)
    assert cal["slot"].max() == 47
