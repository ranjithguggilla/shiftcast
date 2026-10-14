"""From a demand forecast to a shift schedule, and from a schedule to realised cost."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ortools.sat.python import cp_model

from .data import SLOTS_PER_WEEK


@dataclass(frozen=True)
class CostModel:
    """Every number here is an assumption the user can change. See the README for the reasoning."""

    trips_per_cab_slot: float = 1.0  # trips one working cab completes per half hour on average
    cab_slot_cost: float = 9.0  # dollars to keep one cab on the road for 30 minutes
    unmet_trip_cost: float = 15.0  # dollars lost per trip nobody could take (fare plus goodwill)
    # (length in half hours, cost multiplier per cab-slot). Short shifts cost a premium.
    shifts: tuple[tuple[int, float], ...] = field(default=((16, 1.00), (20, 1.00), (8, 1.15)))


def required_cabs(demand: np.ndarray, cost: CostModel) -> np.ndarray:
    return np.ceil(np.asarray(demand, float) / cost.trips_per_cab_slot - 1e-9).astype(int)


@dataclass
class Schedule:
    starts: dict[int, np.ndarray]  # shift length -> number of cabs starting at each slot
    coverage: np.ndarray  # cabs on the road at each slot
    cost: float  # scheduled labour cost in dollars
    status: str
    solve_seconds: float
    lower_bound: float | None = None


def _coverage_from_starts(starts: dict[int, np.ndarray], n: int) -> np.ndarray:
    cov = np.zeros(n, dtype=int)
    for length, x in starts.items():
        for off in range(length):
            cov += np.roll(x, off)
    return cov


def _labour_cost(starts: dict[int, np.ndarray], cost: CostModel) -> float:
    mult = dict(cost.shifts)
    return float(sum(x.sum() * length * mult[length] * cost.cab_slot_cost for length, x in starts.items()))


def solve_cpsat(
    required: np.ndarray, cost: CostModel, time_limit: float = 20.0, workers: int = 4
) -> Schedule:
    """Minimum-cost cover of the required cabs, over a week that wraps onto itself."""
    n = len(required)
    if n != SLOTS_PER_WEEK:
        raise ValueError(f"expected {SLOTS_PER_WEEK} slots, got {n}")
    m = cp_model.CpModel()
    ub = int(required.max()) + 1
    x = {L: [m.NewIntVar(0, ub, f"x_{L}_{s}") for s in range(n)] for L, _ in cost.shifts}
    for t in range(n):
        covering = [x[L][(t - off) % n] for L, _ in cost.shifts for off in range(L)]
        m.Add(sum(covering) >= int(required[t]))
    # integer objective in cents so the solver stays exact
    obj = []
    for L, mult in cost.shifts:
        unit = int(round(L * mult * cost.cab_slot_cost * 100))
        obj += [unit * v for v in x[L]]
    m.Minimize(sum(obj))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = 11
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"no schedule found: {solver.StatusName(status)}")
    starts = {L: np.array([solver.Value(v) for v in x[L]], dtype=int) for L, _ in cost.shifts}
    cov = _coverage_from_starts(starts, n)
    return Schedule(
        starts,
        cov,
        _labour_cost(starts, cost),
        solver.StatusName(status),
        solver.WallTime(),
        solver.BestObjectiveBound() / 100.0,
    )


def solve_greedy(required: np.ndarray, cost: CostModel) -> Schedule:
    """Walk the week once; whenever a slot is short, start that many full-length shifts there."""
    import time

    t0 = time.perf_counter()
    n = len(required)
    length = max(L for L, mult in cost.shifts if mult == min(m for _, m in cost.shifts) and L <= 16)
    x = np.zeros(n, dtype=int)
    cov = np.zeros(n, dtype=int)
    for t in range(n):
        short = int(required[t]) - int(cov[t])
        if short > 0:
            x[t] += short
            for off in range(length):
                cov[(t + off) % n] += short
    # wrap-around can leave early slots short only if their requirement rose after being checked
    for t in range(n):
        short = int(required[t]) - int(cov[t])
        if short > 0:
            x[(t - length + 1) % n] += short
            for off in range(length):
                cov[((t - length + 1) + off) % n] += short
    starts = {L: np.zeros(n, dtype=int) for L, _ in cost.shifts}
    starts[length] = x
    return Schedule(starts, cov, _labour_cost(starts, cost), "HEURISTIC", time.perf_counter() - t0)


def constant_fleet(level: int, n: int, cost: CostModel) -> Schedule:
    """The same number of cabs around the clock. Priced as full-length shifts back to back."""
    cov = np.full(n, int(level))
    mult = min(m for _, m in cost.shifts)
    labour = float(level * n * mult * cost.cab_slot_cost)
    return Schedule({}, cov, labour, "FLAT", 0.0)


@dataclass
class Outcome:
    labour_cost: float
    served: float
    unmet: float
    unmet_cost: float
    total_cost: float
    idle_cab_slots: float
    service_level: float


def evaluate(schedule: Schedule, actual: np.ndarray, cost: CostModel) -> Outcome:
    capacity = schedule.coverage * cost.trips_per_cab_slot
    served = np.minimum(actual, capacity)
    unmet = actual - served
    unmet_cost = float(unmet.sum() * cost.unmet_trip_cost)
    idle = float(((capacity - served) / cost.trips_per_cab_slot).sum())
    return Outcome(
        labour_cost=schedule.cost,
        served=float(served.sum()),
        unmet=float(unmet.sum()),
        unmet_cost=unmet_cost,
        total_cost=schedule.cost + unmet_cost,
        idle_cab_slots=idle,
        service_level=float(served.sum() / actual.sum()),
    )
