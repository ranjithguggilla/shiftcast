import pytest

from shiftcast.service import NEXT_WEEK, Planner
from shiftcast.staffing import CostModel


@pytest.fixture(scope="module")
def planner():
    return Planner()


def test_week_choices_end_with_the_future(planner):
    assert planner.week_choices()[-1] == NEXT_WEEK
    assert len(planner.week_choices()) == 9


def test_score_known_week_has_outcome(planner):
    res = planner.score_week(planner.week_choices()[0], 0.5)
    s = res["summary"]
    assert s["solver_status"] == "OPTIMAL"
    assert 0.9 < s["service_level"] <= 1.0
    assert len(res["frame"]) == 336
    assert (res["frame"].cabs_scheduled >= res["frame"].cabs_required).all()


def test_future_week_has_no_actuals(planner):
    res = planner.score_week(NEXT_WEEK, 0.7)
    assert "unmet_trips" not in res["summary"]
    assert "actual" not in res["frame"]


def test_higher_unmet_cost_is_reflected_in_assumptions(planner):
    res = planner.score_week(planner.week_choices()[1], 0.5, CostModel(unmet_trip_cost=40))
    assert res["summary"]["assumptions"]["unmet_trip_cost"] == 40


def test_higher_level_buys_more_capacity(planner):
    wk = planner.week_choices()[2]
    lo = planner.score_week(wk, 0.5)["summary"]
    hi = planner.score_week(wk, 0.95)["summary"]
    assert hi["labour_cost"] > lo["labour_cost"]
    assert hi["service_level"] >= lo["service_level"]


def test_unknown_week_rejected(planner):
    with pytest.raises(ValueError):
        planner.score_week("2001-01-01")


def test_ui_builds():
    from shiftcast.service import build_app

    demo = build_app()
    assert demo is not None
