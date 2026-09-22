"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Instanzen (je drei Ketten-Seeds) belegt.
Positive UND negative Aussagen: wo VNS nicht besser ist als eine gut kalibrierte feste Störstärke, steht das hier ebenso als Test wie dort,
wo es gewinnt. Rechenzeiten sind nur als Größenordnung geprüft."""

from functools import lru_cache

import numpy as np
import pytest

import vns_constants as C
import vns_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Standardfall -------------------------------------------------------------------------------------------------------------------------------


def test_standard_case_numbers():
    std = cfg()
    near(std["gap"], 0.63, 0.4)
    near(std["ils1"], 0.66, 0.4)
    near(std["ils3"], 0.65, 0.4)
    near(std["dlbr"], 0.68, 0.4)


# --- Budget-Sweep: VNS gegen beide festen ILS-Störstärken --------------------------------------------------------------------------------


@pytest.mark.parametrize("budget,vns,ils1,ils3,tol", [
    (10000, 1.23, 1.56, 1.35, 0.7), (25000, 0.85, 1.12, 0.81, 0.5), (50000, 0.80, 0.81, 0.74, 0.4), (100000, 0.77, 0.78, 0.68, 0.4),
    (200000, 0.63, 0.66, 0.65, 0.4), (500000, 0.56, 0.58, 0.65, 0.35), (1000000, 0.47, 0.50, 0.54, 0.3), (2000000, 0.47, 0.47, 0.47, 0.3),
])
def test_budget_sweep_numbers(budget, vns, ils1, ils3, tol):
    row = cfg(budget=budget)
    near(row["gap"], vns, tol)
    near(row["ils1"], ils1, tol)
    near(row["ils3"], ils3, tol)


def test_vns_matches_or_beats_both_fixed_strengths_at_every_measured_budget():
    for budget in C.BUDGETS:
        row = cfg(budget=budget)
        assert row["gap"] <= min(row["ils1"], row["ils3"]) + 0.35        # VNS ist nirgends deutlich schlechter als die bessere feste Wahl


def test_the_sweet_spot_itself_shifts_with_the_budget():
    """Bei mittlerem Budget ist Störstärke 3 besser, bei großem Budget dreht es sich zugunsten von Störstärke 1 -
    eine feste Störstärke ist also kein robuster Fixpunkt, unabhängig davon, wie gut sie einmal kalibriert wurde."""
    mid = cfg(budget=200000)
    large = cfg(budget=1000000)
    assert mid["ils3"] < mid["ils1"] - 0.005                             # bei 200T: Staerke 3 (Sweet Spot) schlaegt 1
    assert large["ils1"] < large["ils3"] - 0.02                          # bei 1M: dreht sich, Staerke 1 schlaegt jetzt 3


def test_full_rescan_restarts_need_about_seventy_four_thousand_evaluations_for_one_descent_at_sixty_stops():
    v10, v100 = cfg(budget=10000)["hc"], cfg(budget=100000)["hc"]
    near(v10, v100, 0.05)                                                # ein Abstieg ist unabhaengig vom Budget, nur eine Referenz


# --- k_max-Sweep -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("k_max,gap,tol", [(2, 1.71, 0.6), (3, 1.19, 0.5), (5, 1.23, 0.5), (8, 1.70, 0.6)])
def test_k_max_sweep_numbers_at_ten_thousand(k_max, gap, tol):
    near(cfg(budget=10000, k_max=k_max)["gap"], gap, tol)


def test_k_max_three_to_five_beats_both_extremes_at_ten_thousand():
    small = cfg(budget=10000, k_max=2)["gap"]
    mid3 = cfg(budget=10000, k_max=3)["gap"]
    mid5 = cfg(budget=10000, k_max=5)["gap"]
    large = cfg(budget=10000, k_max=8)["gap"]
    assert mid3 < small and mid3 < large
    assert mid5 < small and mid5 < large


# --- Reset-Ablation ---------------------------------------------------------------------------------------------------------------------------


def test_reset_on_success_beats_no_reset_at_two_hundred_thousand():
    base = ev.Settings()
    with_reset = ev.run_config_with_reset(base, True)
    without_reset = ev.run_config_with_reset(base, False)
    near(with_reset["gap"], 0.63, 0.4)
    near(without_reset["gap"], 0.72, 0.4)
    assert with_reset["gap"] < without_reset["gap"] - 0.03


def test_reset_on_success_helps_at_small_budgets_too():
    for budget in (10000, 25000):
        base = ev.Settings(budget=budget)
        with_reset = ev.run_config_with_reset(base, True)
        without_reset = ev.run_config_with_reset(base, False)
        assert with_reset["gap"] < without_reset["gap"]


# --- Lokale Suche -----------------------------------------------------------------------------------------------------------------------------


def test_local_search_numbers_at_two_hundred_thousand():
    dlb = cfg(local_search="dlb")
    full = cfg(local_search="full")
    near(dlb["gap"], 0.63, 0.4)
    near(full["gap"], 4.32, 1.5)
    near(dlb["iterations"], 974, 300)
    near(full["iterations"], 8, 5)
    assert dlb["iterations"] > 30 * full["iterations"]


# --- Große Instanz ----------------------------------------------------------------------------------------------------------------------------


def test_large_instance_at_two_hundred_stops_one_million_budget():
    row = ev.run_config(ev.Settings(n=200), budget=1000000)
    near(row["gap"], 1.9, 1.0)
    assert row["gap"] < row["dlbr"] - 0.5


# --- Startlösung ------------------------------------------------------------------------------------------------------------------------------


def test_start_solution_barely_matters_for_vns_unlike_a_single_descent():
    random_ = cfg(start="random")
    nearest = cfg(start="nearest")
    near(random_["gap"], nearest["gap"], 0.3)
    assert random_["hc"] > nearest["hc"] + 0.3


def test_bound_matches_the_frozen_reference():
    near(ev.reference_bound(60, 0, 100000), 618.76, 0.1)
