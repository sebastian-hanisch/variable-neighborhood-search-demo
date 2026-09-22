"""Presets: Vollständigkeit, gültige Werte, Urteile über mehrere Instanzen und Ketten (Bänder), Permalink-Konstanten."""

import pytest

import vns_constants as C
import vns_evaluation as ev
import vns_presets as P


def _settings(p, seed=None, chain_seed=None):
    return ev.Settings(n=p["n"], cluster_share=p["ballung"], seed=p["seed"] if seed is None else seed, local_search=p["local_search"], k_max=p["k_max"],
                       budget=p["budget"], start=p["start"], chain_seed=p["chain_seed"] if chain_seed is None else chain_seed)


def test_every_preset_has_help_bands_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS) == set(C.PRESET_EXPECTED_BANDS) and len(C.PRESETS) == 8
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0
        assert C.BALLUNG_MIN <= p["ballung"] <= C.BALLUNG_MAX and p["ballung"] % C.BALLUNG_STEP == 0
        assert p["local_search"] in C.LOCAL_SEARCH_LABELS and p["start"] in C.START_LABELS
        assert C.K_MAX_MIN <= p["k_max"] <= C.K_MAX_MAX and p["budget"] in C.BUDGETS
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(p[key])


def test_default_preset_equals_the_default_settings():
    assert _settings(C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_verdicts_stay_in_their_bands_over_instances_and_chains(name):
    p = C.PRESETS[name]
    seeds = range(2) if p["n"] >= 150 else range(5)
    seen = {ev.verdict(ev.analyse(_settings(p, seed=seed, chain_seed=ch), keep_snapshots=False)) for seed in seeds for ch in (0, 1)}
    assert seen <= C.PRESET_EXPECTED_BANDS[name], seen
    assert ev.verdict(ev.analyse(_settings(p), keep_snapshots=False)) in C.PRESET_EXPECTED_BANDS[name]


def test_preset_contrasts_at_the_default_instance():
    g = {name: ev.analyse(_settings(p), keep_snapshots=False) for name, p in C.PRESETS.items() if p["n"] == C.DEFAULT_N}
    std = g["Standardfall (Voreinstellung)"]
    assert g["Voller Rescan (langsam)"].gap > std.gap + 1.0
    assert g["Kleines Budget (10 Tausend)"].run.evaluations < std.run.evaluations


def test_bounds_and_snapping_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert P.STEPS == {"n_slider": C.N_STEP, "ballung_slider": C.BALLUNG_STEP}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
    assert C.DEFAULT_BUDGET in C.BUDGETS
