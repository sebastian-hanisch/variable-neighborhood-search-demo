"""Wortgleiche Kopie aus der simulated-annealing-demo (tests/test_dlb.py), an vns_dlb.py/vns_tour.py angepasst; ergänzt um Tests für den neuen `touched`-Kurzweg (test_dlb_touched_*).

Kandidatenlisten + Don't-Look-Bits (vns_dlb.py): Permutation, Laengenbuchhaltung, Monotonie, Determinismus,
Bewertungsbudget; bei kleinen Instanzen echte 2-opt-Optima (gegen vns_tour.is_local_optimum, unabhaengig von
der Kandidatenliste) und Guete gegen den vollen Rescan-Abstieg vom selben Start; bei 60 Stopps deutlich weniger
Bewertungen fuer vergleichbare Guete (Messreihe 2026-09-22, siehe project_trajectory_metaheuristics_dag_scoping.md)."""

import numpy as np
import pytest

import vns_tour as A
import vns_dlb as D
import vns_kick as K


def _instance(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    return A.dist_matrix(xy)


@pytest.mark.parametrize("n", [6, 8, 12, 20, 40])
def test_tour_stays_valid_length_matches_and_is_monotone(n):
    for seed in range(10):
        Dm = _instance(n, seed)
        cand = D.build_candidate_lists(Dm, min(D.CANDIDATE_K, n - 1))
        t0 = A.random_tour(n, np.random.default_rng(seed + 1000))
        L0 = A.tour_length(t0, Dm)
        r = D.dlb_descend(Dm, t0, cand, seed=seed)
        assert sorted(r.tour.tolist()) == list(range(n))
        assert r.length == pytest.approx(A.tour_length(r.tour, Dm), abs=1e-6)
        assert r.length <= L0 + 1e-9
        assert r.evaluations > 0 and r.converged


def test_deterministic_for_a_seed_and_different_for_another():
    Dm = _instance(30, 3)
    cand = D.build_candidate_lists(Dm)
    t0 = A.random_tour(30, np.random.default_rng(1))
    a = D.dlb_descend(Dm, t0, cand, seed=7)
    b = D.dlb_descend(Dm, t0, cand, seed=7)
    c = D.dlb_descend(Dm, t0, cand, seed=8)
    assert np.array_equal(a.tour, b.tour) and a.evaluations == b.evaluations
    assert not np.array_equal(a.tour, c.tour) or a.evaluations != c.evaluations


def test_max_evaluations_stops_early_and_is_not_converged():
    Dm = _instance(40, 2)
    cand = D.build_candidate_lists(Dm)
    t0 = A.random_tour(40, np.random.default_rng(0))
    full = D.dlb_descend(Dm, t0, cand, seed=0)
    cut = D.dlb_descend(Dm, t0, cand, seed=0, max_evaluations=full.evaluations // 3)
    assert not cut.converged and cut.evaluations <= full.evaluations // 3 + 1
    assert cut.length >= full.length - 1e-9
    unlimited = D.dlb_descend(Dm, t0, cand, seed=0, max_evaluations=10**9)
    assert unlimited.converged and unlimited.evaluations == full.evaluations


# --- Korrektheit gegen eine unabhaengige, vollstaendige Pruefung --------------------------------------------------


@pytest.mark.parametrize("n", [10, 14, 18])
def test_small_instances_reach_a_true_two_opt_local_optimum(n):
    """Bei kleinen Instanzen reicht k=5, um dieselben (echten) 2-opt-Optima wie der volle Abstieg zu finden."""
    hits = 0
    gaps = []
    trials = 15
    for seed in range(trials):
        Dm = _instance(n, seed)
        cand = D.build_candidate_lists(Dm)
        t0 = A.random_tour(n, np.random.default_rng(seed))
        r = D.dlb_descend(Dm, t0, cand, seed=seed)
        if A.is_local_optimum(Dm, r.tour, "2opt"):
            hits += 1
        full = A.descend(Dm, t0, "2opt", "first", keep_steps=False)
        gaps.append(100 * (r.length - full.length) / full.length)
    assert hits == trials                                            # 100 % echte Lokaloptima bei kleinem n
    assert abs(float(np.mean(gaps))) < 3.0                            # im Mittel keine schlechtere Guete als der volle Abstieg


def test_sixty_stops_reaches_a_true_local_optimum_less_often_but_stays_good():
    """Bei 60 Stopps kostet die Kandidatenliste Exaktheit (gemessen: rund die Haelfte), aber nicht die Guete
    (siehe test_evaluations_drop_sharply_at_sixty_stops fuer den Bewertungsvergleich)."""
    hits, gaps, trials = 0, [], 0
    for seed in (100000, 100001, 100002):
        Dm = _instance(60, seed)
        cand = D.build_candidate_lists(Dm)
        for ch in range(5):
            t0 = A.random_tour(60, np.random.default_rng(ch))
            r = D.dlb_descend(Dm, t0, cand, seed=ch)
            trials += 1
            if A.is_local_optimum(Dm, r.tour, "2opt"):
                hits += 1
            full = A.descend(Dm, t0, "2opt", "first", keep_steps=False)
            gaps.append(100 * (r.length - full.length) / full.length)
    assert 0.25 <= hits / trials <= 0.85                              # deutlich unter 100 %, aber nicht selten (gemessen: 51 %)
    assert abs(float(np.mean(gaps))) < 5.0


def test_evaluations_drop_sharply_at_sixty_stops():
    Dm = _instance(60, 100000)
    cand = D.build_candidate_lists(Dm)
    evs_dlb, evs_full, gaps = [], [], []
    for seed in range(8):
        t0 = A.random_tour(60, np.random.default_rng(seed))
        r = D.dlb_descend(Dm, t0, cand, seed=seed)
        full = A.descend(Dm, t0, "2opt", "first", keep_steps=False)
        evs_dlb.append(r.evaluations)
        evs_full.append(full.evaluations)
        gaps.append(100 * (r.length - full.length) / full.length)
    ratio = float(np.mean(evs_full)) / float(np.mean(evs_dlb))
    assert ratio > 50                                                 # gemessen: rund 175x
    assert float(np.mean(evs_dlb)) < 2000                             # gemessen: rund 650
    assert abs(float(np.mean(gaps))) < 5.0                            # vergleichbare Guete trotz weit weniger Bewertungen


def test_candidate_list_shape_and_content():
    Dm = _instance(20, 5)
    cand = D.build_candidate_lists(Dm, 4)
    assert cand.shape == (20, 4)
    for i in range(20):
        assert i not in cand[i]                                       # kein Knoten ist sein eigener Kandidat
        dists = sorted(Dm[i])
        assert Dm[i, cand[i]].tolist() == pytest.approx(dists[1:5])    # die 4 naechsten, sortiert


# --- Neu: der `touched`-Kurzweg (Wiederabstieg nur um die Stoerung, statt vollem Scan) -------------------------


def test_touched_none_is_unchanged_from_the_original_full_scan_behaviour():
    Dm = _instance(30, 4)
    cand = D.build_candidate_lists(Dm)
    t0 = A.random_tour(30, np.random.default_rng(2))
    a = D.dlb_descend(Dm, t0, cand, seed=9, touched=None)
    b = D.dlb_descend(Dm, t0, cand, seed=9)
    assert np.array_equal(a.tour, b.tour) and a.evaluations == b.evaluations


@pytest.mark.parametrize("n", [10, 16, 24])
def test_touched_shortcut_reaches_the_same_local_optimum_quality_as_a_full_scan(n):
    """Startet man den Wiederabstieg nur an den betroffenen Knoten statt an allen, muss trotzdem ein echtes
    2-opt-Optimum erreicht werden (kleine Instanzen: k=5 deckt die ganze Nachbarschaft ab, siehe Test oben) -
    unabhaengig davon, wie wenige Knoten die Warteschlange anfangs enthaelt."""
    for seed in range(8):
        Dm = _instance(n, seed)
        cand = D.build_candidate_lists(Dm)
        t0 = A.random_tour(n, np.random.default_rng(seed))
        full = D.dlb_descend(Dm, t0, cand, seed=seed)                    # bereits ein Lokaloptimum
        rng = np.random.default_rng(seed + 500)
        kicked, touched = K.double_bridge(full.tour, rng, n_bridges=1)
        r = D.dlb_descend(Dm, kicked, cand, seed=seed, touched=touched)
        assert sorted(r.tour.tolist()) == list(range(n))
        assert r.length == pytest.approx(A.tour_length(r.tour, Dm), abs=1e-6)
        assert A.is_local_optimum(Dm, r.tour, "2opt")


def test_touched_shortcut_uses_far_fewer_evaluations_than_a_full_scan_after_a_kick():
    Dm = _instance(60, 100000)
    cand = D.build_candidate_lists(Dm)
    t0 = A.random_tour(60, np.random.default_rng(0))
    full0 = D.dlb_descend(Dm, t0, cand, seed=0)
    rng = np.random.default_rng(1)
    kicked, touched = K.double_bridge(full0.tour, rng, n_bridges=1)
    warm = D.dlb_descend(Dm, kicked, cand, seed=0, touched=touched)
    cold = D.dlb_descend(Dm, kicked, cand, seed=0, touched=None)
    assert warm.evaluations < cold.evaluations / 3                        # der Kurzweg durchsucht nur die Umgebung der Stoerung
    assert warm.length <= cold.length + 1e-6                              # und findet dabei kein schlechteres Ergebnis


def test_touched_shortcut_never_touches_the_edges_of_nodes_outside_the_affected_region():
    """Ein Knoten, der nie in die Warteschlange kommt (nicht betroffen), muss am Ende noch dieselben zwei
    Nachbarn (Kanten) haben wie in der gekickten Ausgangstour - das ist die Garantie, die den Kurzweg ueberhaupt
    korrekt macht (sonst koennte er ein falsches Lokaloptimum liefern). Die reine Array-Position darf sich aendern
    (ein 2-opt-Zug kehrt ein Stueck um: innere Knoten behalten ihre Kanten, aber nicht ihren Index)."""
    Dm = _instance(24, 3)
    cand = D.build_candidate_lists(Dm)
    t0 = A.random_tour(24, np.random.default_rng(0))
    full0 = D.dlb_descend(Dm, t0, cand, seed=0)
    rng = np.random.default_rng(2)
    kicked, touched = K.double_bridge(full0.tour, rng, n_bridges=1)
    r = D.dlb_descend(Dm, kicked, cand, seed=0, touched=touched)

    def node_edges(tour):
        t = tour.tolist()
        n = len(t)
        out = {}
        for i, city in enumerate(t):
            out[city] = frozenset((t[(i - 1) % n], t[(i + 1) % n]))
        return out
    before, after = node_edges(kicked), node_edges(r.tour)
    changed_edges = A.tour_edges(kicked) ^ A.tour_edges(r.tour)
    changed_nodes = {node for e in changed_edges for node in e}
    unaffected = set(range(24)) - changed_nodes - set(touched)
    assert unaffected                                                 # der Test muss etwas pruefen, nicht leer sein
    for node in unaffected:
        assert before[node] == after[node]
