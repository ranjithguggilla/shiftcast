import numpy as np
import pytest

from shiftcast.data import SLOTS_PER_WEEK
from shiftcast.staffing import (
    CostModel,
    constant_fleet,
    evaluate,
    required_cabs,
    solve_cpsat,
    solve_greedy,
)

COST = CostModel()


def _demand():
    t = np.arange(SLOTS_PER_WEEK)
    return 5000 + 4000 * np.sin(2 * np.pi * (t % 48 - 14) / 48)


def test_required_cabs_rounds_up():
    assert list(required_cabs(np.array([0.0, 0.2, 1.0, 1.01]), CostModel(trips_per_cab_slot=1.0))) == [
        0,
        1,
        1,
        2,
    ]


def test_cpsat_covers_every_slot_and_is_optimal():
    req = required_cabs(_demand(), COST)
    sch = solve_cpsat(req, COST)
    assert sch.status == "OPTIMAL"
    assert (sch.coverage >= req).all()


def test_cpsat_never_costs_more_than_greedy():
    req = required_cabs(_demand(), COST)
    assert solve_cpsat(req, COST).cost <= solve_greedy(req, COST).cost


def test_greedy_is_feasible():
    req = required_cabs(_demand(), COST)
    assert (solve_greedy(req, COST).coverage >= req).all()


def test_flat_requirement_is_covered_by_full_shifts_only():
    req = np.full(SLOTS_PER_WEEK, 100)
    sch = solve_cpsat(req, COST)
    # a flat requirement needs 100 cabs in every slot; any mix of full-price shifts costs the same
    assert sch.cost == pytest.approx(100 * SLOTS_PER_WEEK * COST.cab_slot_cost)


def test_short_shifts_used_for_a_narrow_peak():
    req = np.zeros(SLOTS_PER_WEEK, dtype=int)
    req[100:108] = 50
    sch = solve_cpsat(req, COST)
    assert sch.starts[8].sum() == 50
    assert sch.cost == pytest.approx(50 * 8 * 1.15 * COST.cab_slot_cost)


def test_wrong_length_rejected():
    with pytest.raises(ValueError):
        solve_cpsat(np.ones(10, dtype=int), COST)


def test_evaluate_counts_unmet_trips():
    sch = constant_fleet(100, 4, COST)
    out = evaluate(sch, np.array([50.0, 100.0, 150.0, 200.0]), COST)
    assert out.unmet == 150
    assert out.unmet_cost == 150 * COST.unmet_trip_cost
    assert out.service_level == pytest.approx(350 / 500)
    assert out.total_cost == out.labour_cost + out.unmet_cost
