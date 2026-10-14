import numpy as np
import pandas as pd

from shiftcast.intervals import ResidualQuantiles, empirical_coverage, part_of_day


def test_part_of_day_edges():
    idx = pd.DatetimeIndex(["2015-01-01 00:00", "2015-01-01 05:30", "2015-01-01 06:00", "2015-01-01 23:30"])
    assert list(part_of_day(idx)) == [0, 0, 1, 3]


def test_coverage_is_close_to_nominal_on_exchangeable_data():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2014-10-01", periods=4000, freq="30min")
    fc = np.full(4000, 1000.0)
    act = fc * np.exp(rng.normal(0, 0.1, 4000))
    rq = ResidualQuantiles().fit(idx[:2000], act[:2000], fc[:2000])
    lo, hi = rq.interval(idx[2000:], fc[2000:], 0.9)
    assert abs(empirical_coverage(act[2000:], lo, hi) - 0.9) < 0.03


def test_higher_level_never_lowers_the_forecast():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2014-10-01", periods=1500, freq="30min")
    fc = np.full(1500, 500.0)
    act = fc * np.exp(rng.normal(0, 0.2, 1500))
    rq = ResidualQuantiles().fit(idx, act, fc)
    a = rq.predict_quantile(idx[:48], fc[:48], 0.6)
    b = rq.predict_quantile(idx[:48], fc[:48], 0.9)
    assert (b >= a).all()


def test_unfitted_raises():
    import pytest

    with pytest.raises(RuntimeError):
        ResidualQuantiles().quantile(pd.DatetimeIndex(["2015-01-01"]), 0.9)
