"""vns_tour.py ist eine wortgleiche Kopie des Hill-Climbing-Kerns der hill-climbing-demo (über sa_tour.py aus der
simulated-annealing-demo, die bereits das Bewertungsbudget `max_evaluations` ergänzt hat): Nachbarschaften, Abstieg,
Kreuzungen, 1-Baum-Schranke. Hier gegen unabhängige, vollständige Prüfungen getestet (Brute-Force-2-opt, Brute-Force-
Schranke, CP-SAT), damit sich alle weiteren Stücke dieses Repos (Kick, DLB, ILS-Schleife) auf einen geprüften Kern
verlassen können - dieselbe Prüfung wie in hill-climbing-demo/tests und simulated-annealing-demo/tests."""

import itertools

import numpy as np
import pytest

import vns_tour as T


def _instance(n_nodes, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n_nodes, 2)) * 100
    return xy, T.dist_matrix(xy)


def _brute_two_opt_min(t, D):
    t = list(t)
    n = len(t)
    base = T.tour_length(np.array(t), D)
    best = np.inf
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            u = t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:]
            best = min(best, T.tour_length(np.array(u), D) - base)
    return best


def test_two_opt_matches_brute_force_and_descent_ends_in_a_local_optimum():
    xy, D = _instance(12, 8)
    t = T.random_tour(12, np.random.default_rng(3))
    assert T.neighbor_deltas(t, D, "2opt").min() == pytest.approx(_brute_two_opt_min(t, D), abs=1e-9)
    r = T.descend(D, t, "2opt", "first")
    assert T.is_local_optimum(D, r.tour, "2opt") and _brute_two_opt_min(r.tour, D) >= -1e-9
    lengths = [s.length for s in r.steps]
    assert all(b < a - 1e-12 for a, b in zip(lengths, lengths[1:]))


def test_descent_stops_at_the_evaluation_budget():
    xy, D = _instance(40, 2)
    start = T.random_tour(40, np.random.default_rng(1))
    full = T.descend(D, start, "2opt", "first", keep_steps=False)
    cut = T.descend(D, start, "2opt", "first", keep_steps=False, max_evaluations=full.evaluations // 3)
    assert cut.n_moves < full.n_moves and cut.evaluations >= full.evaluations // 3 and cut.length > full.length
    assert T.descend(D, start, "2opt", "first", keep_steps=False, max_evaluations=10 ** 9).n_moves == full.n_moves


def test_bound_against_brute_force_and_cp_sat():
    for seed in range(3):
        xy, D = _instance(8, seed)
        opt = min(T.tour_length([0, *p], D) for p in itertools.permutations(range(1, 8)))
        b = T.held_karp_bound(D, opt)
        assert b <= opt + 1e-9 and b >= 0.9 * opt
    cp = pytest.importorskip("ortools.sat.python.cp_model")
    xy, D = _instance(20, 1)
    di = np.rint(D * 10000).astype(int)
    m = cp.CpModel()
    lits = {(i, j): m.NewBoolVar("") for i in range(20) for j in range(20) if i != j}
    m.AddCircuit([(i, j, l) for (i, j), l in lits.items()])
    m.Minimize(sum(int(di[i, j]) * l for (i, j), l in lits.items()))
    s = cp.CpSolver()
    s.parameters.max_time_in_seconds = 60
    assert s.StatusName(s.Solve(m)) == "OPTIMAL"
    opt = s.ObjectiveValue() / 10000
    ref = T.descend(D, T.nearest_neighbor_tour(D), "2opt+oropt", "best", keep_steps=False)
    b = T.held_karp_bound(D, ref.length)
    assert b <= opt + 0.01 and (opt - b) / opt < 0.03 and ref.length >= opt - 0.01


def test_random_tour_is_a_valid_permutation_with_depot_first():
    for seed in range(10):
        t = T.random_tour(15, np.random.default_rng(seed))
        assert sorted(t.tolist()) == list(range(15)) and t[0] == 0


def test_nearest_neighbor_tour_is_a_valid_permutation_with_depot_first():
    xy, D = _instance(20, 5)
    t = T.nearest_neighbor_tour(D)
    assert sorted(t.tolist()) == list(range(20)) and t[0] == 0
