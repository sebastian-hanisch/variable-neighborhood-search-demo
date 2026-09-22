"""Szenario (wortgleich aus der Hill-Climbing-Demo, eingefrorene Werte) und Auswertung (Kennzahlen, Urteil, ILS mit fester Störstärke,
Hill Climbing mit Neustarts (beide Varianten), Sweeps, Streuung)."""

from dataclasses import replace

import numpy as np
import pytest

import vns_constants as C
import vns_evaluation as ev
import vns_scenario as S
import vns_tour as T


# --- Szenario ---------------------------------------------------------------------------------------------------------------------------------


def test_instance_shape_depot_and_area():
    inst = S.generate(60, 0, 3)
    assert inst.xy.shape == (61, 2) and inst.n == 60 and inst.n_nodes == 61
    assert inst.xy[0].tolist() == [50.0, 50.0]
    assert inst.xy.min() >= 0.0 and inst.xy.max() <= C.AREA


def test_instance_is_deterministic_seed_dependent_and_matches_the_frozen_hill_climbing_instance():
    a, b, c = S.generate(40, 25, 5), S.generate(40, 25, 5), S.generate(40, 25, 6)
    assert np.array_equal(a.xy, b.xy) and not np.array_equal(a.xy, c.xy)
    inst, D = ev.instance(60, 0, 100000)
    assert inst.xy[0].tolist() == [50.0, 50.0] and float(inst.xy[1:].sum()) == pytest.approx(float(S.generate(60, 0, 100000).xy[1:].sum()))
    assert ev.reference_bound(60, 0, 100000) == pytest.approx(618.76, abs=0.05)


def test_grouped_stops_lie_closer_together_than_uniform_ones():
    def mean_nn(share):
        vals = []
        for seed in range(10):
            xy = S.generate(80, share, seed).xy[1:]
            d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
            np.fill_diagonal(d, np.inf)
            vals.append(d.min(axis=1).mean())
        return float(np.mean(vals))
    assert mean_nn(100) < 0.7 * mean_nn(0)


# --- Analyse ------------------------------------------------------------------------------------------------------------------------------------


def test_analysis_fields_are_consistent():
    a = ev.analyse(ev.Settings(budget=20000))
    run = a.run
    assert a.bound < run.best_length <= run.final_length + 1e-9 <= a.hc.length + 20 * 1e3
    assert a.gap == pytest.approx(100 * (run.best_length - a.bound) / a.bound) and a.final_gap >= a.gap - 1e-9
    assert a.hc_gap > a.gap and a.start_gap > a.hc_gap
    assert run.evaluations >= a.settings.budget
    assert 0.0 <= a.success_rate <= 1.0


def test_analysis_is_deterministic_and_chain_seed_matters():
    # n=30 mit reichlich Budget konvergiert von jedem Seed auf dasselbe (echte) Optimum - hier bewusst ein groesseres n
    # mit knapperem Budget, damit die Ketten tatsaechlich unterschiedliche Ergebnisse liefern koennen.
    s = ev.Settings(n=60, budget=8000)
    a, b, c = ev.analyse(s), ev.analyse(s), ev.analyse(replace(s, chain_seed=1))
    assert np.array_equal(a.run.best_tour, b.run.best_tour) and a.gap == b.gap
    assert a.gap != c.gap or not np.array_equal(a.run.final_tour, c.run.final_tour)


def test_nearest_neighbor_start_is_deterministic_and_random_start_follows_the_chain_seed():
    s = ev.Settings(n=25, budget=5000, start="nearest")
    assert np.array_equal(ev.analyse(s).start_tour, ev.analyse(replace(s, chain_seed=3)).start_tour)
    r = ev.Settings(n=25, budget=5000)
    assert not np.array_equal(ev.analyse(r).start_tour, ev.analyse(replace(r, chain_seed=1)).start_tour)


def test_without_hill_climbing_the_comparison_fields_are_empty_and_fast():
    a = ev.analyse(ev.Settings(n=20, budget=2000), with_hc=False)
    assert a.hc is None and a.hcr_tour is None and a.hcr_starts == 0 and a.dlbr_starts == 0 and a.ils1_tour is None and a.ils3_tour is None


def test_hill_climbing_with_restarts_uses_at_least_one_full_descent_and_stays_near_the_budget():
    inst, D = ev.instance(40, 0, 100000)
    single = T.descend(D, T.random_tour(len(D), np.random.default_rng(0)), "2opt", "first", keep_steps=False)
    tour, starts, used = ev.hill_climbing_restarts(D, 1000, 0)
    assert starts == 1 and used > 1000 and T.is_local_optimum(D, tour, "2opt")
    tour, starts, used = ev.hill_climbing_restarts(D, 5 * single.evaluations, 0)
    assert starts >= 3 and used <= 5 * single.evaluations + 2 * len(D) ** 2
    best_single = ev.hill_climbing_restarts(D, 1, 0)[0]
    assert T.tour_length(tour, D) <= T.tour_length(best_single, D) + 1e-9


def test_dlb_restarts_uses_at_least_one_descent_and_is_far_cheaper_than_a_full_rescan_restart():
    inst, D = ev.instance(60, 0, 100000)
    cand = ev.DLB.build_candidate_lists(D)
    best_tour, starts, used = ev.dlb_restarts(D, cand, 1000, 0)
    assert starts >= 1 and used >= 1000 and sorted(best_tour.tolist()) == list(range(len(D)))
    hcr_tour, starts_hcr, used_hcr = ev.hill_climbing_restarts(D, 200000, 0)
    best_dlb, starts_dlb, used_dlb = ev.dlb_restarts(D, cand, 200000, 0)
    assert starts_dlb > 20 * starts_hcr
    assert T.tour_length(best_dlb, D) <= T.tour_length(hcr_tour, D)


def test_ils_fixed_strength_uses_at_least_the_budget_and_reaches_a_valid_tour():
    inst, D = ev.instance(30, 0, 100000)
    cand = ev.DLB.build_candidate_lists(D)
    start = T.random_tour(len(D), np.random.default_rng(0))
    tour, iterations, used = ev.ils_fixed_strength(D, start, cand, 1, 10000, 0)
    assert sorted(tour.tolist()) == list(range(len(D))) and used >= 10000 and iterations >= 1


# --- Urteil -------------------------------------------------------------------------------------------------------------------------------------


def _fake(gap, ils3_gap):
    class F:
        pass
    f = F()
    f.gap, f.ils3_gap = gap, ils3_gap
    return f


def test_verdict_codes():
    assert ev.verdict(_fake(1.0, 1.0 + ev.WIN_MARGIN + 0.1)) == "beats_ils3"
    assert ev.verdict(_fake(1.0 + ev.LOSE_MARGIN + 0.1, 1.0)) == "ils3_wins"
    assert ev.verdict(_fake(1.0, 1.05)) == "comparable"


def test_verdict_of_a_real_run_at_the_default_settings():
    assert ev.verdict(ev.analyse(ev.Settings())) in ("beats_ils3", "comparable")


# --- Sweeps und Tabellen ------------------------------------------------------------------------------------------------------------------


def test_run_config_counts_runs_and_aggregates():
    r = ev.run_config(ev.Settings(n=20, budget=5000))
    assert r["n_runs"] == len(C.SWEEP_SEEDS) * C.SWEEP_CHAINS
    assert r["gap_min"] <= r["gap"] <= r["gap_max"] and r["gap_sd"] >= 0 and r["seconds"] > 0


def test_run_config_ignores_the_seeds_of_the_base_settings():
    a = ev.run_config(ev.Settings(n=15, budget=3000, seed=1, chain_seed=5))
    b = ev.run_config(ev.Settings(n=15, budget=3000, seed=999, chain_seed=0))
    assert all(a[k] == b[k] for k in a if not k.endswith("seconds"))


def test_sweep_values_labels_and_ordering():
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    rows = ev.sweep("local_search", ev.Settings(n=15, budget=20000))
    assert [r["value"] for r in rows] == list(ev.VNS.LOCAL_SEARCHES) and rows[0]["gap"] < rows[1]["gap"]      # DLB besser als voller Rescan
    rows = ev.sweep("budget", ev.Settings(n=20), (2000, 20000))
    assert rows[1]["gap"] <= rows[0]["gap"] + 1.0


def test_reset_ablation_table_helper():
    base = ev.Settings(n=20, budget=15000)
    with_reset = ev.run_config_with_reset(base, True)
    without_reset = ev.run_config_with_reset(base, False)
    assert with_reset["n_runs"] == without_reset["n_runs"] == len(C.SWEEP_SEEDS) * C.SWEEP_CHAINS
    assert with_reset["gap"] <= without_reset["gap"] + 1.0             # Reset hilft (oder schadet zumindest nicht deutlich)


def test_scaling_table_structure(monkeypatch):
    monkeypatch.setattr(ev, "SCALING_N", (10, 20))
    tab = ev.scaling_table(ev.Settings(budget=3000))
    assert len(tab) == 2 and all([r["value"] for r in blk["rows"]] == [10, 20] for blk in tab) and tab[0]["label"] != tab[1]["label"]


def test_chain_spread_returns_one_value_per_chain_and_is_deterministic():
    a = ev.chain_spread(ev.Settings(n=15, budget=3000), 5)
    b = ev.chain_spread(ev.Settings(n=15, budget=3000), 5)
    assert len(a["vns"]) == len(a["hc"]) == 5 and np.array_equal(a["vns"], b["vns"]) and np.array_equal(a["hc"], b["hc"])
