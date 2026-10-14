import pandas as pd
import pytest

from shiftcast.data import SLOTS_PER_WEEK, describe, load_series, validate_grid, week_blocks


def test_real_file_is_a_clean_grid():
    s = load_series()
    assert len(s) == 10320
    assert s.index[0] == pd.Timestamp("2014-07-01 00:00")
    assert s.index[-1] == pd.Timestamp("2015-01-31 23:30")


def test_gap_is_rejected(toy_series):
    broken = toy_series.drop(toy_series.index[10])
    with pytest.raises(ValueError, match="regular"):
        validate_grid(broken)


def test_negative_counts_rejected(toy_series):
    bad = toy_series.copy()
    bad.iloc[3] = -1
    with pytest.raises(ValueError, match="negative"):
        validate_grid(bad)


def test_blocks_are_end_aligned_and_disjoint(toy_series):
    blocks = week_blocks(toy_series, 3, 0)
    assert all(len(b) == SLOTS_PER_WEEK for b in blocks)
    assert blocks[-1][-1] == toy_series.index[-1]
    assert blocks[0][-1] < blocks[1][0]


def test_blocks_refuse_too_much_history(toy_series):
    with pytest.raises(ValueError):
        week_blocks(toy_series, 11, 0)


def test_describe_reports_the_quietest_day():
    d = describe(load_series())
    assert d["min_daily_date"] == "2015-01-27"
