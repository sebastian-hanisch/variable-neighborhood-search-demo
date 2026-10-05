"""Unabhängiges Orakel für Variable Neighborhood Search: (1) die Nachbarschaftswechsel-Logik (Störstärke k) wird aus den Momentaufnahmen allein rekonstruiert (Erfolg = strikt kürzer,
Reset auf 1 nur bei Erfolg und `reset_on_success`, sonst k+1 und nach k_max zurück auf 1); (2) die ganze Schleife wird mit festem Zufallsstrom nachsimuliert (Zähler, Touren, Verlaufskurve
in bewerteten Nachbarn); (3) mit k_max = 1 ist VNS dieselbe Kette wie Iterated Local Search mit Störstärke 1 (`ils_fixed_strength`); (4) die Doppelbrücke wird aus Ein-/Ausgabe zurückgerechnet."""

import numpy as np
import pytest

import vns_algorithm as VNS
import vns_dlb as DLB
import vns_evaluation as EV
import vns_kick as K
import vns_tour as T


def test_double_bridge_is_decoded_back_to_exactly_one_cut_triple():
    rng = np.random.default_rng(1)
    for it in range(40):
        n = int(rng.integers(10, 30))
        t = [0] + rng.permutation(np.arange(1, n)).tolist()
        out = K.double_bridge(np.array(t), np.random.default_rng(it))[0].tolist()
        cuts = [(p1, p2, p3) for p1 in range(1, n) for p2 in range(p1 + 1, n) for p3 in range(p2 + 1, n) if t[:p1] + t[p2:p3] + t[p1:p2] + t[p3:] == out]
        assert len(cuts) == 1
        p1, p2, p3 = cuts[0]
        assert min(p1, p2 - p1, p3 - p2, n - p3) >= K.MIN_SEGMENT


def _reference(D, start, cand, local_search, k_max, reset, budget, seed, trace_points=300):
    rng = np.random.default_rng(seed)
    init_seed = int(rng.integers(0, 2**31 - 1))
    if local_search == "dlb":
        r0 = DLB.dlb_descend(D, start, cand, seed=init_seed, max_evaluations=budget)
    else:
        r0 = T.descend(D, start, "2opt", "first", keep_steps=False, max_evaluations=budget)
    cur, cur_len, ev = r0.tour.copy(), r0.length, r0.evaluations
    best, best_len, snaps, ks = cur.copy(), cur_len, [cur.copy()], []
    every = max(1, budget // trace_points)
    next_trace, trace = (ev // every + 1) * every, [(ev, cur_len, cur_len)]
    its = succ = 0
    k = 1
    while ev < budget:
        shaken, touched = K.double_bridge(cur, rng, n_bridges=k)
        ds = int(rng.integers(0, 2**31 - 1))
        if local_search == "dlb":
            r = DLB.dlb_descend(D, shaken, cand, seed=ds, max_evaluations=budget - ev, touched=touched)
        else:
            r = T.descend(D, shaken, "2opt", "first", keep_steps=False, max_evaluations=budget - ev)
        ev += r.evaluations
        its += 1
        ks.append(k)
        if r.length < cur_len - 1e-9:
            cur, cur_len = r.tour.copy(), r.length
            succ += 1
            k = 1 if reset else k
        else:
            k = k + 1 if k < k_max else 1
        if cur_len < best_len - 1e-9:
            best, best_len = cur.copy(), cur_len
        snaps.append(cur.copy())
        if ev >= next_trace or ev >= budget:
            trace.append((ev, cur_len, best_len))
            next_trace = (ev // every + 1) * every
    return best, cur, ev, its, succ, snaps, ks, np.array(trace)


@pytest.mark.parametrize("local_search", ["dlb", "full"])
def test_run_matches_a_resimulation_and_the_k_sequence_follows_the_success_pattern(local_search):
    rng = np.random.default_rng(11)
    for it in range(8):
        n = int(rng.integers(10, 24))
        D = T.dist_matrix(rng.random((n, 2)) * 100)
        cand = DLB.build_candidate_lists(D, 5)
        k_max, reset = int(rng.integers(1, 9)), it % 3 != 0
        budget = 4000 if local_search == "dlb" else 20000
        start = T.random_tour(n, np.random.default_rng(it))
        got = VNS.run(D, start, cand=cand, local_search=local_search, k_max=k_max, reset_on_success=reset, budget=budget, seed=it)
        best, cur, ev, its, succ, snaps, ks, trace = _reference(D, start, cand, local_search, k_max, reset, budget, it)
        assert (got.evaluations, got.iterations, got.successes) == (ev, its, succ) and got.k_trace == ks
        assert np.array_equal(got.best_tour, best) and np.array_equal(got.final_tour, cur)
        assert len(got.snapshots) == len(snaps) == len(got.k_trace) + 1
        assert np.array_equal(got.trace_iter, trace[:, 0]) and np.allclose(got.trace_length, trace[:, 1]) and np.allclose(got.trace_best, trace[:, 2])
        # k-Folge nur aus den Momentaufnahmen abgeleitet
        lens = [T.tour_length(s, D) for s in got.snapshots]
        k = 1
        for q, used in enumerate(got.k_trace):
            assert used == k and 1 <= used <= k_max
            if lens[q + 1] < lens[q] - 1e-9:
                k = 1 if reset else k
            else:
                k = k + 1 if k < k_max else 1
        assert got.best_length == pytest.approx(min(lens), abs=1e-7)


def test_with_k_max_one_vns_is_iterated_local_search_with_strength_one():
    rng = np.random.default_rng(3)
    for it in range(5):
        n = int(rng.integers(10, 24))
        D = T.dist_matrix(rng.random((n, 2)) * 100)
        cand = DLB.build_candidate_lists(D, 5)
        start = T.random_tour(n, np.random.default_rng(it))
        v = VNS.run(D, start, cand=cand, local_search="dlb", k_max=1, budget=3000, seed=it, keep_snapshots=False)
        tour, iterations, evaluations = EV.ils_fixed_strength(D, start, cand, 1, 3000, it)
        assert np.array_equal(v.best_tour, tour) and (v.iterations, v.evaluations) == (iterations, evaluations)
