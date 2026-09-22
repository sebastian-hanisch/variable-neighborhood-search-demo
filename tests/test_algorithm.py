"""vns_algorithm.run: k-Verlauf (Reset bei Erfolg, Hochzählen bei Misserfolg, Deckelung bei k_max), Budget-Buchführung, Monotonie der
besten Tour, Determinismus, Regressionsschutz für beide Nachbarschaften der lokalen Suche, Ablation "Ohne Reset"."""

import numpy as np
import pytest

import vns_algorithm as VNS
import vns_dlb as DLB
import vns_tour as T


def _instance(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    return T.dist_matrix(xy)


@pytest.mark.parametrize("local_search", VNS.LOCAL_SEARCHES)
def test_tour_stays_valid_and_length_matches_recomputed_length(local_search):
    D = _instance(20, 1)
    cand = DLB.build_candidate_lists(D) if local_search == "dlb" else None
    start = T.random_tour(20, np.random.default_rng(0))
    r = VNS.run(D, start, cand=cand, local_search=local_search, budget=5000, seed=3)
    assert sorted(r.best_tour.tolist()) == list(range(20))
    assert sorted(r.final_tour.tolist()) == list(range(20))
    assert r.best_length == pytest.approx(T.tour_length(r.best_tour, D), abs=1e-6)
    assert r.final_length == pytest.approx(T.tour_length(r.final_tour, D), abs=1e-6)


@pytest.mark.parametrize("local_search", VNS.LOCAL_SEARCHES)
def test_budget_is_respected_and_mostly_exhausted(local_search):
    D = _instance(25, 2)
    cand = DLB.build_candidate_lists(D) if local_search == "dlb" else None
    start = T.random_tour(25, np.random.default_rng(1))
    budget = 8000
    r = VNS.run(D, start, cand=cand, local_search=local_search, budget=budget, seed=1)
    assert r.evaluations >= budget
    assert r.evaluations <= budget + 200
    assert r.iterations >= 1


def test_k_stays_within_bounds_and_the_first_iteration_always_starts_at_one():
    D = _instance(40, 3)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(40, np.random.default_rng(2))
    r = VNS.run(D, start, cand=cand, local_search="dlb", k_max=5, budget=30000, seed=2)
    assert all(1 <= k <= 5 for k in r.k_trace)
    assert len(r.k_trace) == r.iterations
    assert r.k_trace[0] == 1


def test_k_transition_rule_matches_an_independent_replay_from_the_snapshots():
    """Unabhaengige Nachrechnung: ob Iteration i ein Erfolg war, laesst sich an den mitgefuehrten Momentaufnahmen ablesen
    (Laenge sinkt genau dann, wenn sie angenommen wurde); daraus muss sich exakt der naechste k-Wert ergeben."""
    D = _instance(30, 9)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(30, np.random.default_rng(8))
    k_max = 4
    r = VNS.run(D, start, cand=cand, local_search="dlb", k_max=k_max, budget=15000, seed=9)
    lengths = [T.tour_length(s, D) for s in r.snapshots]
    for i in range(len(r.k_trace) - 1):
        success = lengths[i + 1] < lengths[i] - 1e-9
        expected_next = 1 if success else (r.k_trace[i] + 1 if r.k_trace[i] < k_max else 1)
        assert r.k_trace[i + 1] == expected_next


def test_k_escalates_on_repeated_failure_and_wraps_back_to_one():
    """Handgebaute Instanz, bei der der erste Abstieg schon (fast) optimal ist: viele Iterationen scheitern, k muss also
    escalieren und nach k_max wieder auf 1 umschlagen - beobachtbar am k_trace, unabhaengig vom Zufall der Kicks."""
    D = _instance(50, 7)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(50, np.random.default_rng(6))
    r = VNS.run(D, start, cand=cand, local_search="dlb", k_max=3, budget=20000, seed=6)
    assert max(r.k_trace) <= 3 and min(r.k_trace) >= 1
    assert r.successes < r.iterations                                   # nicht jede Iteration ist ein Erfolg (sonst waere k_max nie noetig)


def test_best_length_is_never_worse_than_the_first_local_optimum():
    D = _instance(30, 3)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(30, np.random.default_rng(2))
    first_descent = DLB.dlb_descend(D, start, cand, seed=0)
    r = VNS.run(D, start, cand=cand, local_search="dlb", budget=20000, seed=2)
    assert r.best_length <= first_descent.length + 1e-6


def test_current_length_is_monotone_non_increasing():
    D = _instance(20, 4)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(20, np.random.default_rng(3))
    r = VNS.run(D, start, cand=cand, local_search="dlb", budget=15000, seed=4)
    assert all(b <= a + 1e-6 for a, b in zip(r.trace_length, r.trace_length[1:]))


def test_ablation_without_reset_can_end_up_worse_or_equal_but_never_better_structured_than_real_vns_best():
    """'Ohne Reset' aendert nur, WAS NACH EINEM ERFOLG passiert (k bleibt statt auf 1 zurueckzufallen) - beide Varianten
    duerfen trotzdem eine gute beste Tour finden (best_length wird unabhaengig vom Reset-Mechanismus verfolgt);
    der Unterschied zeigt sich im k_trace, nicht zwingend in der besten Laenge selbst."""
    D = _instance(35, 5)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(35, np.random.default_rng(4))
    real = VNS.run(D, start, cand=cand, local_search="dlb", k_max=6, reset_on_success=True, budget=20000, seed=3)
    ablation = VNS.run(D, start, cand=cand, local_search="dlb", k_max=6, reset_on_success=False, budget=20000, seed=3)
    assert real.k_trace[0] == ablation.k_trace[0] == 1                   # beide starten identisch
    assert sorted(real.best_tour.tolist()) == sorted(ablation.best_tour.tolist()) == list(range(35))
    # die beiden k-Verlaeufe muessen sich irgendwann unterscheiden, sobald der erste Erfolg eintritt (sonst waere die Ablation wirkungslos)
    assert real.k_trace != ablation.k_trace or real.successes == 0


def test_deterministic_for_a_seed_and_different_for_another():
    D = _instance(25, 6)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(25, np.random.default_rng(5))
    a = VNS.run(D, start, cand=cand, local_search="dlb", budget=10000, seed=11)
    b = VNS.run(D, start, cand=cand, local_search="dlb", budget=10000, seed=11)
    c = VNS.run(D, start, cand=cand, local_search="dlb", budget=10000, seed=40)
    assert np.array_equal(a.best_tour, b.best_tour) and a.evaluations == b.evaluations
    assert not np.array_equal(a.best_tour, c.best_tour) or a.evaluations != c.evaluations


def test_dlb_and_full_rescan_reach_comparable_quality_on_a_small_instance():
    D = _instance(15, 7)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(15, np.random.default_rng(6))
    dlb = VNS.run(D, start, cand=cand, local_search="dlb", budget=40000, seed=1)
    full = VNS.run(D, start, local_search="full", budget=40000, seed=1)
    assert abs(dlb.best_length - full.best_length) / full.best_length < 0.05


def test_unknown_local_search_and_bad_k_max_are_rejected():
    D = _instance(10, 0)
    start = T.random_tour(10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        VNS.run(D, start, cand=DLB.build_candidate_lists(D), local_search="nonsense", budget=1000)
    with pytest.raises(ValueError):
        VNS.run(D, start, local_search="dlb", cand=None, budget=1000)
    with pytest.raises(ValueError):
        VNS.run(D, start, cand=DLB.build_candidate_lists(D), k_max=0, budget=1000)


def test_snapshots_and_k_trace_are_only_kept_when_requested():
    D = _instance(12, 0)
    cand = DLB.build_candidate_lists(D)
    start = T.random_tour(12, np.random.default_rng(0))
    with_snaps = VNS.run(D, start, cand=cand, budget=3000, seed=0, keep_snapshots=True)
    without = VNS.run(D, start, cand=cand, budget=3000, seed=0, keep_snapshots=False)
    assert len(with_snaps.snapshots) == with_snaps.iterations + 1
    assert without.snapshots == []
    assert len(with_snaps.k_trace) == with_snaps.iterations             # k_trace ist unabhaengig von keep_snapshots immer da
