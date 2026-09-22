"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt, Randwerte, Würfel-Knöpfe, Permalink-Grenzen, Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import vns_constants as C
import vns_evaluation as ev

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(vns_step=1, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if vns_step != 1:
        at.select_slider(key="vns_step").set_value(vns_step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_measured_default():
    at = _run()
    _ok(at)
    assert _metric(at, "Beste Tour") == "0.1 %"
    assert any("Gleichauf" in s.value for s in at.info) or any("Besser als" in s.value for s in at.success)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["local_search_select"] == p["local_search"] and at.session_state["budget_select"] == p["budget"] and at.session_state["n_slider"] == p["n"]
    assert at.metric


@pytest.mark.parametrize("step", [1, 2, 3, 4])
@pytest.mark.parametrize("n", [10, 60])
def test_every_step_runs(step, n):
    at = _run(n_slider=n, budget_select=25000, vns_step=step)
    _ok(at)
    assert at.get("plotly_chart") and at.session_state["vns_step"] == step


def test_step_three_iteration_slider_and_play_shakes():
    at = _run(vns_step=3, budget_select=25000)
    _ok(at)
    lv = next(s for s in at.slider if s.key == "vns_iter")
    assert lv.value == lv.max
    lv.set_value(0).run()
    _ok(at)
    at2 = _run(vns_step=3, budget_select=25000)
    next(b for b in at2.button if b.label == "▶️ Shakes abspielen").click().run()
    _ok(at2)


def test_play_runs_through_all_steps_without_duplicate_keys():
    at = _run(n_slider=20, budget_select=10000)
    next(b for b in at.button if b.label == "▶️ Abspielen").click().run()
    _ok(at)


def test_local_search_toggle_changes_the_result():
    at = _run(local_search_select="full", budget_select=25000)
    _ok(at)
    full_gap = _metric(at, "Beste Tour")
    at2 = _run(local_search_select="dlb", budget_select=25000)
    dlb_gap = _metric(at2, "Beste Tour")
    assert full_gap != dlb_gap


def test_k_max_extremes_run():
    for kmax in (C.K_MAX_MIN, C.K_MAX_MAX):
        _ok(_run(k_max_slider=kmax, budget_select=10000))


def test_dice_buttons_change_the_seeds():
    at = _run(budget_select=10000)
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old
    old_c = at.session_state["chain_seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Kette würfeln").click().run()
    _ok(at)
    assert at.session_state["chain_seed_input"] != old_c


@pytest.mark.parametrize("kw", [dict(n_slider=200, budget_select=50000), dict(n_slider=10, ballung_slider=100, budget_select=10000),
                                 dict(local_search_select="full", budget_select=25000), dict(k_max_slider=1, budget_select=10000),
                                 dict(k_max_slider=8, budget_select=10000), dict(start_radio="nearest", budget_select=25000)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["ballung"] = "40"
    at.query_params["ls"] = "nonsense"
    at.query_params["kmax"] = "99"
    at.query_params["budget"] = "12345"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["ballung_slider"] == 50 and at.session_state["local_search_select"] == C.DEFAULT_LOCAL_SEARCH
    assert at.session_state["k_max_slider"] == C.K_MAX_MAX and at.session_state["budget_select"] == C.DEFAULT_BUDGET


def test_sweeps_run_on_demand():
    at = _run(n_slider=10, budget_select=10000)
    at.selectbox(key="sweep_select").set_value("local_search").run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_experiments_run_on_demand(monkeypatch):
    monkeypatch.setitem(ev.SWEEP_VALUES, "budget", (2000, 5000))
    monkeypatch.setitem(ev.SWEEP_VALUES, "k_max", (1, 2))
    monkeypatch.setattr(C, "SCALING_N", (10, 20))
    monkeypatch.setattr(ev, "SCALING_POLICIES", (("Budget 4 Tausend", lambda n: 4000), ("Budget 300 · Stopps", lambda n: 300 * n)))
    at = _run(n_slider=10, budget_select=10000)
    for key, flag in (("budget_start", "budget_on"), ("kmax_start", "kmax_on"), ("reset_start", "reset_on"),
                      ("spread_start", "spread_on"), ("scaling_start", "scaling_on")):
        next(b for b in at.button if b.key == key).click().run()
        _ok(at)
        assert at.session_state[flag]


def test_footer_and_grenzen_are_present():
    at = _run(budget_select=10000)
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Das Budget reicht für mehrere Eskalationszyklen" in m.value for m in at.markdown)
