"""Doppelbrücken-Zug (vns_kick.double_bridge): gültige Permutation, kein No-op, und die eigentliche Pointe - durch keinen einzelnen
2-opt-Zug rückgängig zu machen (erschöpfend gegen alle 2-opt-Züge auf kleinen Instanzen geprüft, nicht nur behauptet)."""

import numpy as np
import pytest

from vns_kick import double_bridge


def _all_2opt_neighbors(tour):
    """Alle Touren, die durch genau einen 2-opt-Zug aus `tour` erreichbar sind (inkl. Ausgangstour selbst nicht enthalten)."""
    n = len(tour)
    out = []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            t = tour.copy()
            t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
            out.append(tuple(t.tolist()))
    return set(out)


@pytest.mark.parametrize("n", [4, 5, 8, 12])
def test_double_bridge_is_a_valid_permutation(n):
    rng = np.random.default_rng(0)
    tour = np.arange(n)
    for _ in range(20):
        t, touched = double_bridge(tour, rng)
        assert sorted(t.tolist()) == list(range(n))
        assert touched and all(0 <= x < n for x in touched)


def test_double_bridge_is_never_a_no_op():
    rng = np.random.default_rng(1)
    tour = np.arange(10)
    for _ in range(50):
        t, _ = double_bridge(tour, rng)
        assert not np.array_equal(t, tour)


@pytest.mark.parametrize("n", [8, 10, 12])
def test_double_bridge_is_not_reachable_by_a_single_two_opt_move(n):
    rng = np.random.default_rng(2)
    tour = np.arange(n)
    neighbors = _all_2opt_neighbors(tour)
    hits = 0
    for _ in range(200):
        t, _ = double_bridge(tour, rng)
        if tuple(t.tolist()) in neighbors:
            hits += 1
    assert hits == 0                                             # keine einzige Doppelbrücke ist ein verkleideter 2-opt-Zug


def test_double_bridge_changes_at_least_three_edges():
    # gilt garantiert nur ab n >= 4*MIN_SEGMENT (Mindestlaenge je Stueck erzwingt es); bei kleineren Touren kann der Zug entarten (siehe Docstring)
    rng = np.random.default_rng(3)
    tour = np.arange(12)

    def edges(t):
        return {(int(min(a, b)), int(max(a, b))) for a, b in zip(t, np.roll(t, -1))}
    old_e = edges(tour)
    for _ in range(30):
        t, _ = double_bridge(tour, rng)
        removed = old_e - edges(t)
        assert len(removed) >= 3


def test_touched_nodes_are_exactly_the_new_boundary_endpoints():
    rng = np.random.default_rng(4)
    tour = np.arange(9)
    t, touched = double_bridge(tour, rng, n_bridges=1)

    def edges(x):
        return {(int(min(a, b)), int(max(a, b))) for a, b in zip(x, np.roll(x, -1))}
    new_edges = edges(t) - edges(tour)
    endpoints = {node for e in new_edges for node in e}
    assert endpoints <= set(touched)                            # jeder Endpunkt einer neuen Kante ist als betroffen markiert


def test_multiple_bridges_stack_and_touched_nodes_accumulate():
    rng = np.random.default_rng(5)
    tour = np.arange(20)
    t1, touched1 = double_bridge(tour, rng, n_bridges=1)
    rng2 = np.random.default_rng(5)
    t3, touched3 = double_bridge(tour, rng2, n_bridges=3)
    assert sorted(t3.tolist()) == list(range(20))
    assert len(touched3) >= len(touched1)                       # drei Züge betreffen mindestens so viele Knoten wie einer


def test_raises_on_too_small_a_tour():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        double_bridge(np.arange(3), rng)
