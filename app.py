"""Variable Neighborhood Search - eine Lieferrunde, deren Störstärke sich selbst anpasst - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Trajektorien-Metaheuristiken-Linie der "Konzepte"-Reihe: baut direkt auf der iterated-local-search-demo auf. ILS
brauchte eine FESTE Störstärke (dort von Hand kalibriert: 2-3 Doppelbrücken, nicht der literaturübliche Standardwert 1). VNS ersetzt
den festen Regler durch ein systematisches Verfahren: die Störstärke k startet bei 1, wächst nach einem erfolglosen Kick, und wird
nach einem ERFOLG wieder auf 1 zurückgesetzt. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time
from dataclasses import replace

import numpy as np
import streamlit as st

import vns_constants as C
import vns_dlb as DLB
import vns_tour as T
from vns_evaluation import SWEEP_LABELS, Settings, analyse, chain_spread, run_config_with_reset, scaling_table, sweep, verdict
from vns_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_chain_seed,
    randomize_seed,
    sync_query_params,
)
from vns_visualization import build_budget, build_instance, build_k_trace, build_scaling, build_spread, build_sweep, build_tour, build_trace

st.set_page_config(page_title="Variable Neighborhood Search – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _spread(base):
    return chain_spread(base)


@st.cache_data(show_spinner=False)
def _scaling(base):
    return scaling_table(base)


@st.cache_data(show_spinner=False)
def _reset_ablation(base):
    return {"mit Reset (echtes VNS)": run_config_with_reset(base, True), "ohne Reset": run_config_with_reset(base, False)}


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


st.title("🪜 Variable Neighborhood Search – eine Lieferrunde, deren Störstärke sich selbst anpasst")
st.markdown(
    """
**Iterated Local Search** (die vorige Demo) stört eine gute Tour mit einer **festen** Störstärke – von Hand kalibriert lag der Sweet Spot bei 2-3 Doppelbrücken, nicht beim literaturüblichen Standardwert 1. **Variable Neighborhood Search** (VNS) braucht diese Kalibrierung nicht: die Störstärke $k$
startet bei 1 und **wächst**, solange ein Kick keinen Erfolg bringt; sobald einer erfolgreich ist, fällt $k$ **zurück auf 1**. Kein Regler für "die richtige Störstärke" – das Verfahren passt sich selbst an.
Schlägt das systematische Eskalieren + Zurücksetzen eine gut kalibrierte, aber feste Störstärke – oder kommt es zumindest nah heran, ohne den Sweet Spot vorher zu kennen? Das misst diese Demo, gegen dieselbe untere Schranke wie die Geschwister-Demos.
"""
)
st.caption(
    "Viertes Stück der Trajektorien-Metaheuristiken-Linie der \"Konzepte\"-Reihe: dieselbe Rundtour wie in der "
    "[hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/), der "
    "[simulated-annealing-demo](https://sebastianhanisch-simulated-annealing-demo.streamlit.app/) und der "
    "[iterated-local-search-demo](https://github.com/sebastian-hanisch/iterated-local-search-demo) - ein Depot in der Mitte, n Kundenstopps in einem 100 × 100-km-Gebiet, euklidische Entfernungen. "
    "ALNS baut als Nächstes direkt auf VNS auf (lernt, welcher Umbau sich lohnt, statt blind zu eskalieren); Tabu Search und GRASP sind andere Antworten auf dieselbe Schwäche der Wurzel."
)

with st.expander("So funktioniert Variable Neighborhood Search", expanded=True):
    st.markdown(
        """
1. **Erster Abstieg.** Aus der Startlösung wird ganz normal lokal optimiert (2-opt), bis kein Zug mehr verbessert.
2. **Shake mit Störstärke $k$.** $k$ Doppelbrücken-Züge nacheinander (derselbe Zug wie bei Iterated Local Search) – zu Beginn $k=1$, die schwächste Störung.
3. **Wiederabstieg.** Nur die Umgebung der Störung wird neu durchsucht (Kandidatenliste + Don't-Look-Bits, oder zum Vergleich der volle Rescan).
4. **Erfolg oder nicht?** Ist die neue Tour kürzer als die aktuelle: **übernehmen, $k$ zurück auf 1** – die nächste Störung ist wieder die schwächste. Sonst: **verwerfen, $k$ um 1 erhöhen** (stärker stören); ist $k_{\\max}$ erreicht, ohne dass es geklappt hat: zurück auf $k=1$.
5. **Bewertung.** Der Abstand zur **1-Baum-Schranke**, wie in den Geschwister-Demos. Verglichen wird mit **Iterated Local Search bei fester Störstärke** (1 = naiv, 3 = der von Hand gefundene Sweet Spot) und mit **Hill Climbing mit Neustarts** (beide Varianten), gleiches Bewertungsbudget.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:4], preset_names[4:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider(
        "Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
        help="Anzahl der Kundenstopps (das Depot kommt dazu).",
    )
    cluster_share = st.slider(
        "Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP,
        help="Wie viele Stopps in fünf Gruppen (Städten) liegen statt gleichverteilt im Gebiet.",
    )
    local_search = st.selectbox(
        "Lokale Suche", list(C.LOCAL_SEARCH_LABELS), key="local_search_select", format_func=lambda k: C.LOCAL_SEARCH_LABELS[k],
        help="Kandidatenliste + Don't-Look-Bits durchsucht nach einem Shake nur die Umgebung der Störung (schnell: 0.63 % über der Schranke bei 200 Tausend Vorschlägen, rund 974 Iterationen); "
             "der volle Rescan bewertet nach jedem Zug wieder alle Nachbarn (langsam: 4.32 % bei nur rund 8 Iterationen).",
    )
    k_max = st.slider(
        "Maximale Störstärke k_max", C.K_MAX_MIN, C.K_MAX_MAX, key="k_max_slider",
        help="Wie weit die Störstärke eskalieren darf, bevor sie ohne Erfolg auf 1 zurückgesetzt wird. Bei 10 Tausend Vorschlägen: k_max=2 → 1.71 %, k_max=3 → 1.19 %, k_max=5 → 1.23 % (Voreinstellung), k_max=8 → 1.70 % über der Schranke - "
             "ein zu hoher Wert verschenkt bei knappem Budget: das Eskalieren bis 8 kostet selbst Bewertungen, bevor ein Erfolg zurücksetzt.",
    )
    budget = st.select_slider(
        "Budget (bewertete Nachbarschaften)", options=list(C.BUDGETS), key="budget_select", format_func=_fmt_int,
        help="Bei 10 Tausend Vorschlägen liegt VNS bei 1.23 % über der Schranke, ILS mit dem von Hand kalibrierten Sweet Spot (Störstärke 3) bei 1.35 %, ILS naiv (Störstärke 1) bei 1.56 % - VNS schlägt hier BEIDE. "
             "Bei sehr großem Budget (1-2 Millionen) dreht sich das Bild leicht: die naive Störstärke 1 wird dort besser als Störstärke 3 (0.50 gegen 0.54 % bei 1 Million) - der 'Sweet Spot' ist selbst budgetabhängig, VNS trifft ihn trotzdem ohne ihn zu kennen (0.47 %).",
    )
    start = st.radio(
        "Startlösung", list(C.START_LABELS), key="start_radio", format_func=lambda k: C.START_LABELS[k], horizontal=True,
        help="Zufällige Reihenfolge oder Nächster Nachbar.",
    )
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für die Lage der Stopps.")
    chain_seed = st.number_input(
        "Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
        help="Steuert die zufällige Startlösung, die Doppelbrücken-Schnittpunkte und die Reihenfolge der Wiederabstiege.",
    )
    st.button("🎲 Neue Kette würfeln", width="stretch", on_click=randomize_chain_seed, help="Würfelt einen neuen Seed für dieselbe Instanz.")

sync_query_params({
    "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "seed_input": int(seed), "local_search_select": local_search,
    "k_max_slider": int(k_max), "budget_select": int(budget), "start_radio": start, "chain_seed_input": int(chain_seed),
})

settings = Settings(int(n_stops), int(cluster_share), int(seed), local_search, int(k_max), int(budget), start, int(chain_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
run = a.run
xy = a.inst.xy
code = verdict(a)
data_key = settings

# --- Variable Neighborhood Search in Aktion ---------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Variable Neighborhood Search in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Erster Abstieg", 3: "3 · Shakes", 4: "4 · Ergebnis"}
if "vns_step" not in st.session_state or st.session_state.get("vns_step_owner") != data_key:
    st.session_state["vns_step"] = 1
    st.session_state["vns_step_owner"] = data_key
    st.session_state.pop("vns_iter", None)
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="vns_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

n_snaps = len(run.snapshots)
iteration = n_snaps - 1
play_shakes = False
if step == 3 and n_snaps > 1:
    it_col, itplay_col = st.columns([5, 2])
    with it_col:
        iteration = st.slider("Iteration", 0, n_snaps - 1, value=n_snaps - 1, key="vns_iter", help="Die Tour nach dieser Iteration (0 = nach dem ersten Abstieg, vor dem ersten Shake).")
    with itplay_col:
        play_shakes = st.button("▶️ Shakes abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_snaps <= 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_snaps - 1, min(n_snaps, 40))})


def _render(current_step, it):
    with view_slot.container():
        if current_step == 1:
            st.markdown(f"**{a.inst.n} Kundenstopps und das Depot (Stern)** – {a.inst.cluster_share} % der Stopps in Gruppen")
            st.plotly_chart(build_instance(xy), width="stretch", key="s1_map")
        elif current_step == 2:
            c1, c2 = st.columns([3, 2])
            c1.markdown(f"**Tour nach dem ersten Abstieg** – {a.start_gap:.1f} % über der Schranke vor, {100 * (run.trace_length[0] - a.bound) / a.bound:.1f} % nach dem Abstieg")
            c1.plotly_chart(build_tour(xy, run.snapshots[0]), width="stretch", key="s2_map")
            c2.markdown("**Vor den Shakes**")
            c2.metric("Startlösung", f"{a.start_gap:.1f} %", help="Abstand zur Schranke der Startlösung, vor jeder Optimierung.")
            c2.metric("Nach dem ersten Abstieg", f"{100 * (run.trace_length[0] - a.bound) / a.bound:.1f} %", help="Abstand zur Schranke, bevor der erste Shake angewendet wird.")
        elif current_step == 3:
            st.markdown("**Störstärke k über die Iterationen** – fällt bei jedem Erfolg auf 1 zurück")
            st.plotly_chart(build_k_trace(run.k_trace[:max(it, 1)]), width="stretch", key=f"s3_k_{it}")
            st.markdown("**Länge der aktuellen und der besten Tour über die bewerteten Nachbarschaften**")
            st.plotly_chart(build_trace(run.trace_iter, run.trace_length, run.trace_best, a.bound, a.hc.length, T.tour_length(a.hcr_tour, a.D), T.tour_length(a.dlbr_tour, a.D)), width="stretch", key=f"s3_trace_{it}")
            st.markdown(f"**Tour nach Iteration {it} von {n_snaps - 1}**")
            st.plotly_chart(build_tour(xy, run.snapshots[it], ghost=run.best_tour if it < n_snaps - 1 else None), width="stretch", key=f"s3_map_{it}")
        else:
            c1, c2 = st.columns(2)
            c1.markdown(f"**VNS: beste Tour** – {a.gap:.1f} % über der Schranke")
            c1.plotly_chart(build_tour(xy, run.best_tour), width="stretch", key="s4_vns")
            c2.markdown(f"**ILS, Störstärke 3 (von Hand kalibrierter Sweet Spot)** (gleiches Budget) – {a.ils3_gap:.1f} % über der Schranke")
            c2.plotly_chart(build_tour(xy, a.ils3_tour), width="stretch", key="s4_ils3")


if auto_play:
    for s in STEP_LABELS:
        if s == 3:
            for f in _frames():
                _render(3, f)
                time.sleep(0.1)
            time.sleep(0.6)
        else:
            _render(s, iteration)
            time.sleep(1.2)
    step = 4
elif play_shakes:
    for f in _frames():
        _render(3, f)
        time.sleep(0.1)
else:
    _render(step, iteration)

if step == 1:
    st.caption(f"{a.inst.n} Stopps; die untere Schranke der kürzesten Rundtour liegt bei {a.bound:,.0f} km (1-Baum-Schranke, Held-Karp).".replace(",", "."))
elif step == 2:
    st.caption("Der erste Abstieg optimiert lokal (2-opt), bis kein Zug mehr verbessert. Ab hier beginnen die Shakes, mit k=1.")
elif step == 3:
    st.caption(f"{_fmt_int(run.evaluations)} bewertete Nachbarschaften in {run.iterations} Iterationen; erfolgreich waren {a.success_rate:.0%} davon (k fällt dann auf 1 zurück).")
else:
    st.caption(f"Links die beste Tour aus {run.iterations} Iterationen, rechts die beste Tour von Iterated Local Search mit derselben lokalen Suche und Störstärke 3 bei gleichem Bewertungsbudget ({_fmt_int(settings.budget)}).")

st.markdown("---")

# --- Ergebnis -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Kette gefunden hat")
st.caption(
    "**Abstand zur Schranke:** Länge der Tour gegenüber einer unteren Schranke der kürzesten Rundtour (1-Baum, Held-Karp) in Prozent. "
    "Ein Lauf ist eine Ziehung: die Ketten streuen (siehe Streuung unten), Vergleiche gelten für diesen Lauf."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Beste Tour", f"{a.gap:.1f} %", delta=f"letzte Tour {a.final_gap:.1f} %", delta_color="off", help="Abstand zur Schranke der kürzesten je besuchten Tour; im Delta der der letzten Tour der Kette.")
m2.metric("ILS, Störstärke 1 (naiv)", f"{a.ils1_gap:.1f} %", delta_color="off", help="Iterated Local Search mit fester Störstärke 1, gleiches Budget.")
m3.metric("ILS, Störstärke 3 (Sweet Spot)", f"{a.ils3_gap:.1f} %", delta_color="off", help="Iterated Local Search mit dem in der Schwester-Demo von Hand gefundenen Sweet Spot, gleiches Budget - die faire Latte für VNS.")
m4.metric("Erfolgreiche Iterationen", f"{a.success_rate:.0%}", delta=f"{run.successes} von {run.iterations}", delta_color="off", help="Anteil der Shakes, deren Wiederabstieg eine kürzere Tour fand (k fällt danach auf 1 zurück).")

if code == "beats_ils3":
    st.success(f"✅ Besser als Iterated Local Search mit dem von Hand kalibrierten Sweet Spot: {a.gap:.1f} % über der Schranke gegen {a.ils3_gap:.1f} % (Störstärke 3, gleiches Budget). VNS muss den Sweet Spot nicht kennen. Andere Ketten streuen um dieses Ergebnis.")
elif code == "comparable":
    st.info(f"ℹ️ Gleichauf: VNS {a.gap:.1f} %, Iterated Local Search mit Störstärke 3 {a.ils3_gap:.1f} % über der Schranke - VNS kommt nah an den von Hand kalibrierten Sweet Spot heran, ohne ihn zu kennen. Eine andere Kette kann das Bild drehen.")
else:
    st.warning(f"⚠️ Der von Hand kalibrierte Sweet Spot ist besser: {a.ils3_gap:.1f} % gegen {a.gap:.1f} % über der Schranke bei gleichem Budget. Bei sehr knappem Budget kann das Eskalieren selbst zu viele Bewertungen kosten (siehe README 'Was nicht funktioniert hat').")

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    unit_time = lambda sec: f"{sec * 1000:.0f} ms"  # noqa: E731
    st.table({"": ["Länge (km)", "Abstand zur Schranke", "Bewertete Nachbarschaften", "Rechenzeit"],
              "VNS (beste Tour)": [f"{run.best_length:.1f}", f"{a.gap:.2f} %", _fmt_int(run.evaluations), unit_time(a.seconds)],
              "VNS (letzte Tour)": [f"{run.final_length:.1f}", f"{a.final_gap:.2f} %", "–", "–"],
              "ILS, Störstärke 1": [f"{T.tour_length(a.ils1_tour, a.D):.1f}", f"{a.ils1_gap:.2f} %", _fmt_int(settings.budget), unit_time(a.ils1_seconds)],
              "ILS, Störstärke 3": [f"{T.tour_length(a.ils3_tour, a.D):.1f}", f"{a.ils3_gap:.2f} %", _fmt_int(settings.budget), unit_time(a.ils3_seconds)],
              "HC-Neustarts (Kandidatenliste + DLB)": [f"{T.tour_length(a.dlbr_tour, a.D):.1f}", f"{a.dlbr_gap:.2f} %", _fmt_int(settings.budget), unit_time(a.dlbr_seconds)]})
with d2:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["Lokale Suche", "Maximale Störstärke k_max", "Budget", "Iterationen", "Startlösung", "Kreuzungen der besten Tour"],
              "Einstellung": [C.LOCAL_SEARCH_LABELS[settings.local_search], f"{settings.k_max}", _fmt_int(settings.budget), f"{run.iterations}", C.START_LABELS[settings.start], f"{a.crossings_end}"]})
    st.caption("Ein Vorschlag ist eine bewertete Nachbarschaft, dieselbe Einheit wie in den Geschwister-Demos. Rechenzeiten hängen vom Rechner ab, nur die Größenordnung zählt.")

st.markdown("---")

# --- Sweeps -----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Budget, k_max, Suche und Instanz ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
base_sweep = replace(settings, seed=0, chain_seed=0)
if st.button("Sweep über 5 feste Instanzen berechnen (dauert etwa 10 bis 60 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Instanzen × 3 Ketten..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    categorical = sweep_param in ("local_search", "start")
    labels = {"local_search": C.LOCAL_SEARCH_LABELS, "start": C.START_LABELS}.get(sweep_param)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], categorical=categorical, key_labels=labels), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band bzw. Balken) über 5 feste Instanzen (Seeds 100000–100004, getrennt vom Seed oben) mit je drei Ketten; alle anderen Regler wie in der Seitenleiste. "
               "Gestrichelt: ein Hill-Climbing-Abstieg; gepunktet: Iterated Local Search mit Störstärke 3 (der Sweet Spot, den VNS nicht kennen muss).")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Budget: erreicht VNS den Sweet Spot, ohne ihn zu kennen?")
if st.button("Budget von 10 Tausend bis 2 Millionen durchfahren (dauert etwa 90 Sekunden)", key="budget_start"):
    st.session_state["budget_on"] = True
if st.session_state.get("budget_on"):
    with st.spinner("Rechne 8 Budgets × 5 Instanzen × 3 Ketten..."):
        rows_b = _sweep("budget", base_sweep)
    st.plotly_chart(build_budget(rows_b), width="stretch", key="budget_chart")
    st.table({"Budget": [_fmt_int(r["value"]) for r in rows_b], "VNS (%)": [f"{r['gap']:.2f}" for r in rows_b],
              "ILS Störstärke 3 (%)": [f"{r['ils3']:.2f}" for r in rows_b], "ILS Störstärke 1 (%)": [f"{r['ils1']:.2f}" for r in rows_b],
              "HC-Neustarts, Kandidatenliste+DLB (%)": [f"{r['dlbr']:.2f}" for r in rows_b]})
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (60 Stopps, Kandidatenliste + DLB, k_max=5). Mit dem kalibrierten k_max **schlägt VNS bei jedem gemessenen Budget sowohl die naive feste Störstärke 1 als auch den von Hand gefundenen Sweet Spot (Störstärke 3)** oder liegt gleichauf - "
               "10 Tausend: VNS 1.23 % gegen 1.56 % (Störstärke 1) und 1.35 % (Störstärke 3); 200 Tausend: 0.63 % gegen 0.66 % und 0.65 %. Bei sehr großem Budget (1-2 Millionen) dreht sich, WELCHE feste Störstärke die bessere ist (1 wird besser als 3: 0.50 gegen 0.54 % bei 1 Million) - "
               "**der 'Sweet Spot' ist selbst budgetabhängig, nicht universell**, und genau das macht eine feste Störstärke fragil. VNS trifft die jeweils beste Wahl trotzdem, ohne sie zu kennen (0.47 % bei 1-2 Millionen, gleichauf mit der jeweils besseren festen Störstärke). "
               "Mit einem zu großen k_max (8 statt 5) kippt das Bild bei knappem Budget allerdings (siehe k_max-Experiment) - die Robustheit hängt an einer vernünftig gewählten Obergrenze, nicht am Prinzip allein.")

st.markdown("---")

st.subheader("🔬 k_max: wie weit soll die Eskalation reichen dürfen?")
if st.button("k_max 1 bis 8 bei knappem Budget durchfahren (dauert etwa 20 Sekunden)", key="kmax_start"):
    st.session_state["kmax_on"] = True
if st.session_state.get("kmax_on"):
    with st.spinner("Rechne 5 k_max-Werte × 5 Instanzen × 3 Ketten..."):
        rows_k = _sweep("k_max", replace(base_sweep, budget=10000))
    st.plotly_chart(build_sweep(rows_k, SWEEP_LABELS["k_max"]), width="stretch", key="kmax_chart")
    st.table({"k_max": [f"{r['value']}" for r in rows_k], "VNS (%)": [f"{r['gap']:.2f}" for r in rows_k]})
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten, 10 Tausend Vorschläge (60 Stopps, Kandidatenliste + DLB): k_max=2 **1.71 %**, k_max=3 **1.19 %** (bestes gemessen), k_max=5 1.23 %, k_max=8 **1.70 %** über der Schranke - "
               "ein zu großes k_max ist bei knappem Budget fast so schlecht wie ein zu kleines: die Eskalation bis 8 kostet Bewertungen, ohne dass sie oft gebraucht wird.")

st.markdown("---")

st.subheader("🔬 Zurücksetzen: ist das der eigentliche Hebel?")
if st.button("Mit und ohne Reset vergleichen (dauert etwa 15 Sekunden)", key="reset_start"):
    st.session_state["reset_on"] = True
if st.session_state.get("reset_on"):
    with st.spinner("Rechne beide Varianten × 5 Instanzen × 3 Ketten..."):
        rows_r = _reset_ablation(base_sweep)
    st.table({"Regel": list(rows_r.keys()), "Beste Tour (%)": [f"{v['gap']:.2f}" for v in rows_r.values()], "Letzte Tour (%)": [f"{v['final']:.2f}" for v in rows_r.values()]})
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (200 Tausend Vorschläge). Mit Reset (echtes VNS): **0.63 %**. Ohne Reset (k eskaliert bei Erfolg NICHT zurück auf 1, sondern bleibt): **0.72 %** - "
               "das Zurücksetzen selbst ist der Hebel, nicht nur das Eskalieren an sich. Bei knapperem Budget ist der Unterschied relativ noch größer (15-22 % schlechter ohne Reset).")

st.markdown("---")

st.subheader("🔬 Streuung: wie verlässlich ist eine Kette?")
if st.button("20 Ketten auf dieser Instanz berechnen (dauert etwa 10 Sekunden)", key="spread_start"):
    st.session_state["spread_on"] = True
if st.session_state.get("spread_on"):
    with st.spinner("Rechne 20 Ketten und 20 Abstiege..."):
        sp = _spread(replace(settings, chain_seed=0))
    st.plotly_chart(build_spread(sp["vns"], sp["hc"]), width="stretch", key="spread_chart")
    s1, s2 = st.columns(2)
    s1.metric("VNS: Mittel ± Streuung", f"{sp['vns'].mean():.2f} ± {sp['vns'].std():.2f} %", help="Mittel und Standardabweichung des Abstands der besten Tour über 20 Ketten.")
    s2.metric("Ein Hill-Climbing-Abstieg: Mittel ± Streuung", f"{sp['hc'].mean():.2f} ± {sp['hc'].std():.2f} %", help="Ein Abstieg je Kette aus derselben zufälligen Startlösung.")
    st.caption("Dieselbe Instanz, 20 verschiedene Ketten-Seeds (die Startlösung wechselt mit).")

st.markdown("---")

st.subheader("🔬 Skalierung: wie viel Budget braucht ein größeres Problem?")
if st.button("Stopps von 20 bis 200 durchfahren (dauert etwa 60 Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Rechne 6 Größen × 2 Budgetregeln × 5 Instanzen × 3 Ketten..."):
        sc = _scaling(replace(base_sweep, n=C.DEFAULT_N))
    st.plotly_chart(build_scaling(sc), width="stretch", key="scaling_chart")
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (Einstellungen wie in der Seitenleiste außer Stopps und Budget); zum Vergleich Hill Climbing mit Neustarts (voller Rescan, wie in der Wurzel-Demo).")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Budget reicht für mehrere Eskalationszyklen** | Bei sehr knappem Budget (5-10 Tausend) verliert VNS sogar gegen die naive feste Störstärke 1 - das Eskalieren selbst kostet Bewertungen, bevor ein Erfolg zurücksetzt. Erst ab etwa 25 Tausend erreicht VNS den von Hand kalibrierten Sweet Spot. | **ALNS** (lernt, welcher Umbau sich lohnt, statt blind zu eskalieren) |
| **k_max ist selbst kalibriert** | k_max=8 ist bei knappem Budget (10 Tausend) fast so schlecht wie k_max=2 (**1.70 %** gegen **1.71 %**); k_max=3-5 liegt klar davor (**1.19-1.23 %**) - VNS nimmt einem also nicht JEDE Kalibrierung ab, nur die der Störstärke selbst. | (Kalibrierungsfrage, kein Verfahrensnachfolger) |
| **Das Zurücksetzen passiert** | Ohne Reset (k bleibt nach einem Erfolg stehen statt auf 1 zu fallen): **0.72 %** gegen **0.63 %** bei 200 Tausend Vorschlägen - das Zurücksetzen ist der eigentliche Hebel, nicht nur das Eskalieren. | Kein direkter Nachfolger; die Lehre gilt sinngemäß für jedes eskalierende Verfahren |
| **Die lokale Suche ist billig** | Ohne Kandidatenliste + DLB (voller Rescan): **4.32 %** bei nur rund 8 Iterationen statt 0.63 % bei rund 974 (200 Tausend Vorschläge) - dieselbe Lehre wie bei Iterated Local Search. | Kein Nachfolger nötig - gilt für jede iterative Metaheuristik mit vielen Wiederabstiegen |
| **Die Störung ist strukturell, nicht zufällig** | Wie bei ILS: der Doppelbrücken-Zug lässt sich durch keinen einzelnen 2-opt-Zug rückgängig machen (in der ILS-Demo geprüft, hier wortgleich übernommen). | **Lin-Kernighan** (chained LK), Nachbarschafts-Zweig |
"""
)
st.caption(
    "Die Nachbarn der Trajektorien-Metaheuristiken-Linie: ALNS baut als Nächstes direkt auf VNS auf; Tabu Search, GRASP und der Nachbarschafts-Zweig "
    "(Lin-Kernighan, VLSN, VRP-Nachbarschaften) sind andere Antworten auf dieselbe Schwäche der Wurzel."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Kürzeste Rundtour über $N = n+1$ Knoten mit euklidischen Entfernungen $d_{ij}$; $L(\pi)$ ist die Länge einer Tour $\pi$.

**Nachbarschaftsstruktur $N_k$.** $N_k(\pi)$ = die Touren, die durch $k$ nacheinander ausgeführte Doppelbrücken-Züge aus $\pi$ erreichbar sind (derselbe Zug wie bei Iterated Local Search, ils_kick.double_bridge).

**Basic VNS** (Mladenović & Hansen 1997). $k \leftarrow 1$. Wiederhole: ziehe $\pi' \in N_k(\pi)$ zufällig (Shake), setze $\pi'' \leftarrow \text{LocalSearch}(\pi')$ (2-opt bis zum lokalen Optimum). Ist $L(\pi'') < L(\pi)$: $\pi \leftarrow \pi''$, $k \leftarrow 1$. Sonst: $k \leftarrow k+1$; ist $k > k_{\max}$: $k \leftarrow 1$. Gemerkt wird $\arg\min L$ über alle je besuchten $\pi''$.

**Kandidatenliste + Don't-Look-Bits.** Wie in Iterated Local Search: nach einem Shake startet die Warteschlange der lokalen Suche nur mit den Endpunkten der neuen Kanten, nicht mit allen $N$ Knoten.

**Kennzahl.** Abstand zur Schranke $= 100 \cdot (L - w)/w$ mit der 1-Baum-Schranke $w$. Vergleichsgrößen bei gleichem Budget: Iterated Local Search mit fester Störstärke 1 (naiv) und 3 (von Hand kalibrierter Sweet Spot der Schwester-Demo), Hill Climbing mit Neustarts (voller Rescan und Kandidatenliste + DLB).

**Grenzen.** (1) Das Eskalieren selbst kostet Bewertungen - bei sehr knappem Budget lohnt es sich nicht. (2) $k_{\max}$ ist selbst ein (kleinerer) Kalibrierungsparameter. (3) Das Zurücksetzen bei Erfolg ist der eigentliche Hebel, nicht nur das Eskalieren. (4) Die lokale Suche muss billig sein.

Implementiert in `vns_kick.py` (Doppelbrücke, wortgleich aus der ILS-Demo), `vns_dlb.py` (Kandidatenliste + Don't-Look-Bits mit `touched`-Kurzweg), `vns_algorithm.py` (die VNS-Schleife: Eskalieren + Zurücksetzen), `vns_tour.py` (Nachbarschaften, Abstieg, Schranke), `vns_scenario.py` (Instanzen), `vns_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil, ILS-Vergleichsgrößen mit fester Störstärke).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
