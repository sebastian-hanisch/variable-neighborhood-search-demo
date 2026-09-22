"""Konstanten der Variable-Neighborhood-Search-Demo: Szenario (wortgleich zur hill-climbing-demo/simulated-annealing-demo/iterated-local-search-demo), Regler, Beschriftungen (Presets folgen nach den Messungen)."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0              # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0            # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_CHAINS = 3                 # Ketten-Seeds je Instanz in Sweeps und Vergleichstabellen
BOUND_ITERATIONS = 300

N_MIN, N_MAX, DEFAULT_N, N_STEP = 10, 200, 60, 5
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_CHAIN_SEED = 0
DEFAULT_START = "random"
START_LABELS = {"random": "Zufällig", "nearest": "Nächster Nachbar"}
BUDGETS = (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000)
SCALING_N = (20, 40, 60, 100, 150, 200)
SPREAD_CHAINS = 20

# --- VNS-eigene Regler -------------------------------------------------------------------------------------------
LOCAL_SEARCH_LABELS = {"dlb": "Kandidatenliste + Don't-Look-Bits", "full": "Voller Rescan"}
DEFAULT_LOCAL_SEARCH = "dlb"
K_MAX_MIN, K_MAX_MAX, DEFAULT_K_MAX = 1, 8, 5
DEFAULT_BUDGET = 200000

# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Ketten-Seeds), 60 gleichverteilte Stopps, Abstand zur Schranke (2026-09-22):
#   k_max-Sweep bei Budget 10T/25T/200T: k_max=2 -> 1.71/1.17/0.64 %, k_max=3 -> 1.19/0.89/0.70 %, k_max=5 -> 1.23/0.85/0.63 % (bester Kompromiss ueber alle Budgets),
#   k_max=8 -> 1.70/0.82/0.65 % (bei knappem Budget klar am schlechtesten - das Eskalieren bis 8 kostet selbst Bewertungen, bevor ein Erfolg zurücksetzt).
#   Default = 5 (kein Extremwert: k_max=8 verschenkt bei kleinem Budget, k_max=2 erreicht den Sweet Spot bei mittlerem Budget nicht ganz).
# Hauptvergleich, VOLLER Budget-Sweep mit dem kalibrierten k_max=5 (5 Instanzen x 3 Ketten, 60 Stopps):
#   Budget       10T    25T    50T    100T   200T   500T   1M     2M
#   VNS          1.23   0.85   0.80   0.77   0.63   0.56   0.47   0.47
#   ILS Staerke1 1.56   1.12   0.81   0.78   0.66   0.58   0.50   0.47
#   ILS Staerke3 1.35   0.81   0.74   0.68   0.65   0.65   0.54   0.47
#   HC-Neustarts Kandidatenliste+DLB (fair)  2.35  1.37  0.97  0.78  0.68  0.61  0.59  0.58
# Mit dem KALIBRIERTEN k_max schlaegt VNS bei JEDEM gemessenen Budget beide festen ILS-Varianten oder liegt gleichauf - anders als mit
# k_max=8 (siehe k_max-Sweep unten), wo VNS bei kleinem Budget hinter die naive Staerke 1 zurueckfaellt. WICHTIGER, unerwarteter Fund:
# welche feste Staerke der "Sweet Spot" ist, aendert sich selbst mit dem Budget (3 gewinnt bei 25T-200T, aber 1 wird ab 500T besser als 3:
# 0.58 vs 0.65 bei 500T, 0.50 vs 0.54 bei 1M) - eine feste Staerke ist also KEIN robuster Fixpunkt, VNS trifft die jeweils bessere Wahl
# trotzdem, ohne sie zu kennen.
# Reset-Ablation (Budget 200T, k_max=5): "mit Reset" (echtes VNS) 0.63 %, "ohne Reset" (k eskaliert bei Erfolg NICHT zurueck auf 1) 0.72 % -
#   das Zurücksetzen selbst ist der Hebel, nicht nur das Eskalieren (bei 10T/25T Budget ähnlich: 15-22 % relativ schlechter ohne Reset).
# Lokale Suche (Budget 200T): Kandidatenliste+DLB 0.63 % bei 974 Iterationen, voller Rescan 4.32 % bei nur 8 Iterationen (derselbe Faktor wie in der ganzen Linie).


def _preset(local_search=DEFAULT_LOCAL_SEARCH, k_max=DEFAULT_K_MAX, budget=DEFAULT_BUDGET, n=DEFAULT_N, start=DEFAULT_START):
    return {"n": n, "ballung": DEFAULT_BALLUNG, "seed": DEFAULT_SEED, "local_search": local_search, "k_max": k_max, "budget": budget, "start": start,
            "chain_seed": DEFAULT_CHAIN_SEED}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Kleines Budget (10 Tausend)": _preset(budget=10000),
    "Voller Rescan (langsam)": _preset(local_search="full"),
    "k_max zu groß (8)": _preset(k_max=8, budget=10000),
    "k_max zu klein (2)": _preset(k_max=2, budget=10000),
    "Nächster Nachbar als Start": _preset(start="nearest"),
    "Großes Budget (1 Million)": _preset(budget=1000000),
    "Große Instanz (200 Stopps, 1 Million)": _preset(n=200, budget=1000000),
}
# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Ketten-Seeds), Abstand zur Schranke; ILS/Hill Climbing bei gleichem Bewertungsbudget
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "60 Stopps, Kandidatenliste + DLB, k_max=5, 200 Tausend Vorschläge: die beste Tour liegt im Mittel 0.63 % über der Schranke - ILS mit dem von Hand kalibrierten Sweet Spot (Störstärke 3) 0.65 %, ILS naiv (Störstärke 1) 0.66 %, HC-Neustarts (Kandidatenliste + DLB) 0.68 %.",
    "Kleines Budget (10 Tausend)": "Nur 10 Tausend Vorschläge: VNS 1.23 % über der Schranke - schlägt sowohl ILS Störstärke 1 (1.56 %) als auch den von Hand kalibrierten Sweet Spot Störstärke 3 (1.35 %) klar, mit dem kalibrierten k_max=5.",
    "Voller Rescan (langsam)": "Dieselbe Suche wie die Wurzel-Demo (Hill Climbing), ohne Kandidatenliste: 4.32 % über der Schranke bei nur rund 8 Iterationen (200 Tausend Vorschläge) - ohne Kandidatenliste + DLB lohnt sich die Eskalation kaum.",
    "k_max zu groß (8)": "Maximale Störstärke 8 bei knappem Budget (10 Tausend): 1.70 % über der Schranke - schlechter als k_max=3 (1.19 %) oder k_max=5 (1.23 %, Voreinstellung): das Eskalieren bis 8 kostet selbst Bewertungen, bevor ein Erfolg zurücksetzt.",
    "k_max zu klein (2)": "Maximale Störstärke 2 bei knappem Budget (10 Tausend): 1.71 % über der Schranke - fast so schlecht wie k_max=8 (1.70 %): zu wenig Spielraum, um aus einer schlechten Umgebung herauszukommen.",
    "Nächster Nachbar als Start": "Die vielen Shakes vergessen die Startlösung schnell: Nächster Nachbar liegt kaum anders als eine zufällige Startlösung über der Schranke (anders als bei einem einzelnen Hill-Climbing-Abstieg: 5.4 gegen 6.8 %).",
    "Großes Budget (1 Million)": "1 Million Vorschläge: VNS 0.47 % über der Schranke, gleichauf mit der JEWEILS besseren festen Störstärke - hier hat sich der Sweet Spot gedreht (Störstärke 1 mit 0.50 % ist jetzt besser als Störstärke 3 mit 0.54 %), VNS trifft die bessere Wahl trotzdem.",
    "Große Instanz (200 Stopps, 1 Million)": "200 Stopps, 1 Million Vorschläge: VNS 1.9 % über der Schranke gegen 4.71 % für Hill Climbing mit Neustarts (Kandidatenliste + DLB) bei gleichem Budget.",
}
# Urteile, die bei diesem Preset über verschiedene Instanzen und Ketten-Seeds vorkommen (jedes Preset wird über mehrere Instanzen x 2 Ketten gemessen)
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": {"beats_ils3", "comparable"},
    "Kleines Budget (10 Tausend)": {"beats_ils3", "comparable", "ils3_wins"},
    "Voller Rescan (langsam)": {"ils3_wins", "comparable"},
    "k_max zu groß (8)": {"beats_ils3", "comparable", "ils3_wins"},
    "k_max zu klein (2)": {"beats_ils3", "comparable", "ils3_wins"},
    "Nächster Nachbar als Start": {"beats_ils3", "comparable", "ils3_wins"},
    "Großes Budget (1 Million)": {"beats_ils3", "comparable"},
    "Große Instanz (200 Stopps, 1 Million)": {"beats_ils3", "comparable", "ils3_wins"},
}
