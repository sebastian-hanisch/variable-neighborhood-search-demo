"""Auswertung der Variable-Neighborhood-Search-Demo: ein Lauf gegen Iterated Local Search mit fester Störstärke (Störstärke 1 = naiv,
Störstärke 3 = der in der Schwester-Demo von Hand gefundene Sweet Spot) und gegen Hill Climbing mit Neustarts (voller Rescan UND
Kandidatenliste + Don't-Look-Bits), gleiches Bewertungsbudget. Sweeps, Vergleichstabellen, Kettenstreuung.

Der Abstand zur Schranke ist der Abstand zu einer *unteren* Schranke der kürzesten Tour (1-Baum, Held-Karp); er überschätzt die wahre
Lücke um die Schrankenlücke (im Mittel unter 1 %). Ein Vorschlag ist eine bewertete Nachbarschaft - bei Kandidatenliste + Don't-Look-Bits
ein geprüftes Kandidatenpaar, beim vollen Rescan ein bewerteter Nachbar im Abstieg."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import vns_algorithm as VNS
import vns_constants as C
import vns_dlb as DLB
import vns_kick as K
import vns_scenario as S
import vns_tour as T


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    cluster_share: int = C.DEFAULT_BALLUNG
    seed: int = C.DEFAULT_SEED
    local_search: str = C.DEFAULT_LOCAL_SEARCH
    k_max: int = C.DEFAULT_K_MAX
    budget: int = C.DEFAULT_BUDGET
    start: str = C.DEFAULT_START
    chain_seed: int = C.DEFAULT_CHAIN_SEED


@lru_cache(maxsize=256)
def instance(n, cluster_share, seed):
    inst = S.generate(n, cluster_share, seed)
    return inst, T.dist_matrix(inst.xy)


@lru_cache(maxsize=256)
def reference_bound(n, cluster_share, seed):
    """1-Baum-Schranke; Ziel des Subgradientenverfahrens ist die Länge eines guten lokalen Optimums (Nächster Nachbar + 2-opt/Or-opt, steilster Abstieg)."""
    inst, D = instance(n, cluster_share, seed)
    ref = T.descend(D, T.nearest_neighbor_tour(D), "2opt+oropt", "best", keep_steps=False)
    return T.held_karp_bound(D, ref.length, C.BOUND_ITERATIONS)


@lru_cache(maxsize=256)
def candidate_lists(n, cluster_share, seed):
    inst, D = instance(n, cluster_share, seed)
    return DLB.build_candidate_lists(D)


def make_start(settings, D):
    if settings.start == "nearest":
        return T.nearest_neighbor_tour(D)
    return T.random_tour(len(D), np.random.default_rng(settings.chain_seed))


def hill_climbing_restarts(D, budget, seed):
    """Hill Climbing mit Neustarts, voller Rescan (wie in der Wurzel-Demo)."""
    rng = np.random.default_rng(seed)
    used, starts, best = 0, 0, None
    while used < budget or best is None:
        cap = None if best is None else budget - used
        r = T.descend(D, T.random_tour(len(D), rng), "2opt", "first", keep_steps=False, max_evaluations=cap)
        used += r.evaluations
        starts += 1
        if best is None or r.length < best.length:
            best = r
    return best.tour, starts, used


def dlb_restarts(D, cand, budget, seed):
    """Wie hill_climbing_restarts, mit Kandidatenliste + Don't-Look-Bits statt vollem Rescan - die FAIRE Vergleichsgröße."""
    rng = np.random.default_rng(seed)
    used, starts, best_tour, best_length = 0, 0, None, None
    while used < budget or best_tour is None:
        cap = None if best_tour is None else budget - used
        r = DLB.dlb_descend(D, T.random_tour(len(D), rng), cand, seed=starts, max_evaluations=cap)
        used += r.evaluations
        starts += 1
        if best_length is None or r.length < best_length:
            best_length, best_tour = r.length, r.tour
    return best_tour, starts, used


def ils_fixed_strength(D, start, cand, n_bridges, budget, seed):
    """Iterated Local Search mit FESTER Störstärke (Kopie der ILS-Schleife aus iterated-local-search-demo, hier ohne Cross-Import
    neu geschrieben): dieselbe Kandidatenliste + DLB-Wiederabstiege wie VNS, aber ohne Eskalieren/Zurücksetzen. Gibt (beste Tour,
    Iterationen, verbrauchte Bewertungen) zurück - die Vergleichsgröße "wüsste man die richtige Störstärke vorher"."""
    rng = np.random.default_rng(seed)
    init_seed = int(rng.integers(0, 2**31 - 1))
    r0 = DLB.dlb_descend(D, start, cand, seed=init_seed, max_evaluations=budget)
    current, current_length = r0.tour, r0.length
    evaluations = r0.evaluations
    best_tour, best_length = current.copy(), current_length
    iterations = 0
    while evaluations < budget:
        remaining = budget - evaluations
        shaken, touched = K.double_bridge(current, rng, n_bridges=n_bridges)
        descend_seed = int(rng.integers(0, 2**31 - 1))
        r = DLB.dlb_descend(D, shaken, cand, seed=descend_seed, max_evaluations=remaining, touched=touched)
        evaluations += r.evaluations
        iterations += 1
        if r.length < current_length - 1e-9:
            current, current_length = r.tour, r.length
        if current_length < best_length - 1e-9:
            best_length, best_tour = current_length, current.copy()
    return best_tour, iterations, evaluations


@dataclass
class Analysis:
    settings: Settings
    inst: object
    D: np.ndarray
    bound: float
    start_tour: np.ndarray
    run: object
    seconds: float
    hc: object                      # ein Hill-Climbing-Abstieg (voller Rescan) aus derselben Startlösung
    hc_seconds: float
    hcr_tour: np.ndarray             # Hill Climbing mit Neustarts, voller Rescan, gleiches Budget
    hcr_starts: int
    hcr_seconds: float
    dlbr_tour: np.ndarray             # Hill Climbing mit Neustarts, Kandidatenliste + DLB, gleiches Budget
    dlbr_starts: int
    dlbr_seconds: float
    ils1_tour: np.ndarray             # ILS, Störstärke 1 (naiv), gleiches Budget
    ils1_seconds: float
    ils3_tour: np.ndarray             # ILS, Störstärke 3 (von Hand kalibrierter Sweet Spot), gleiches Budget
    ils3_seconds: float
    crossings_end: int

    def gap_of(self, length):
        return 100.0 * (length - self.bound) / self.bound

    @property
    def gap(self):
        return self.gap_of(self.run.best_length)

    @property
    def final_gap(self):
        return self.gap_of(self.run.final_length)

    @property
    def hc_gap(self):
        return self.gap_of(self.hc.length)

    @property
    def hcr_gap(self):
        return self.gap_of(T.tour_length(self.hcr_tour, self.D))

    @property
    def dlbr_gap(self):
        return self.gap_of(T.tour_length(self.dlbr_tour, self.D))

    @property
    def ils1_gap(self):
        return self.gap_of(T.tour_length(self.ils1_tour, self.D))

    @property
    def ils3_gap(self):
        return self.gap_of(T.tour_length(self.ils3_tour, self.D))

    @property
    def start_gap(self):
        return self.gap_of(T.tour_length(self.start_tour, self.D))

    @property
    def success_rate(self):
        return self.run.successes / max(self.run.iterations, 1)


def analyse(settings, keep_snapshots=True, with_hc=True):
    inst, D = instance(settings.n, settings.cluster_share, settings.seed)
    bound = reference_bound(settings.n, settings.cluster_share, settings.seed)
    start = make_start(settings, D)
    cand = candidate_lists(settings.n, settings.cluster_share, settings.seed) if settings.local_search == "dlb" else None
    t0 = time.perf_counter()
    run = VNS.run(D, start, cand=cand, local_search=settings.local_search, k_max=settings.k_max, budget=settings.budget,
                  seed=settings.chain_seed, keep_snapshots=keep_snapshots)
    seconds = time.perf_counter() - t0
    hc = hc_tour = dlbr_tour = ils1_tour = ils3_tour = None
    hc_seconds = hcr_seconds = dlbr_seconds = ils1_seconds = ils3_seconds = 0.0
    hcr_starts = dlbr_starts = 0
    if with_hc:
        t0 = time.perf_counter()
        hc = T.descend(D, start, "2opt", "first", keep_steps=False)
        hc_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        hc_tour, hcr_starts, _ = hill_climbing_restarts(D, settings.budget, settings.chain_seed)
        hcr_seconds = time.perf_counter() - t0
        cand_hc = cand if cand is not None else DLB.build_candidate_lists(D)
        t0 = time.perf_counter()
        dlbr_tour, dlbr_starts, _ = dlb_restarts(D, cand_hc, settings.budget, settings.chain_seed)
        dlbr_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        ils1_tour, _, _ = ils_fixed_strength(D, start, cand_hc, 1, settings.budget, settings.chain_seed)
        ils1_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        ils3_tour, _, _ = ils_fixed_strength(D, start, cand_hc, 3, settings.budget, settings.chain_seed)
        ils3_seconds = time.perf_counter() - t0
    return Analysis(settings, inst, D, bound, start, run, seconds, hc, hc_seconds, hc_tour, hcr_starts, hcr_seconds,
                     dlbr_tour, dlbr_starts, dlbr_seconds, ils1_tour, ils1_seconds, ils3_tour, ils3_seconds,
                     T.count_crossings(inst.xy, run.best_tour))


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------------

WIN_MARGIN = 0.3                  # so viel besser als ILS mit der von Hand kalibrierten Störstärke 3 gilt als Sieg (Prozentpunkte)
LOSE_MARGIN = 0.3                 # so viel schlechter gilt als Niederlage


def verdict(a):
    """Code: beats_ils3 (deutlich besser als Iterated Local Search mit dem von Hand gefundenen Sweet Spot bei gleichem Budget - VNS
    braucht diesen Sweet Spot nicht zu kennen), ils3_wins (der kalibrierte Sweet Spot ist besser), comparable. Gilt für diesen einen Lauf."""
    if a.gap <= a.ils3_gap - WIN_MARGIN:
        return "beats_ils3"
    if a.ils3_gap <= a.gap - LOSE_MARGIN:
        return "ils3_wins"
    return "comparable"


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    """Mittel über die festen Instanzen und je `chains` Ketten-Seeds für die Einstellungen `base` mit `changes`."""
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        for ch in range(chains):
            a = analyse(replace(s0, seed=seed, chain_seed=ch), keep_snapshots=False)
            rows.append({"gap": a.gap, "final": a.final_gap, "hc": a.hc_gap, "hcr": a.hcr_gap, "dlbr": a.dlbr_gap, "ils1": a.ils1_gap, "ils3": a.ils3_gap,
                         "seconds": a.seconds, "success_rate": a.success_rate, "iterations": a.run.iterations, "crossings": a.crossings_end})
    out = {k: _mean(rows, k) for k in rows[0]}
    out.update({"gap_sd": float(np.std([r["gap"] for r in rows])), "gap_min": float(np.min([r["gap"] for r in rows])),
                "gap_max": float(np.max([r["gap"] for r in rows])), "n_runs": len(rows)})
    return out


def run_config_with_reset(base, reset_on_success, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    """Wie run_config, aber für die Reset-Ablation (`reset_on_success` ist kein Settings-Feld, deshalb ein eigener Pfad)."""
    s0 = replace(base, **changes)
    gaps, finals = [], []
    for seed in seeds:
        for ch in range(chains):
            s = replace(s0, seed=seed, chain_seed=ch)
            inst, D = instance(s.n, s.cluster_share, s.seed)
            bound = reference_bound(s.n, s.cluster_share, s.seed)
            start = make_start(s, D)
            cand = candidate_lists(s.n, s.cluster_share, s.seed)
            r = VNS.run(D, start, cand=cand, local_search=s.local_search, k_max=s.k_max, reset_on_success=reset_on_success, budget=s.budget, seed=s.chain_seed, keep_snapshots=False)
            gaps.append(100 * (r.best_length - bound) / bound)
            finals.append(100 * (r.final_length - bound) / bound)
    return {"gap": float(np.mean(gaps)), "final": float(np.mean(finals)), "gap_sd": float(np.std(gaps)), "n_runs": len(gaps)}


SWEEP_VALUES = {"budget": (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000), "k_max": (1, 2, 3, 5, 8),
                "local_search": VNS.LOCAL_SEARCHES, "n": (10, 20, 40, 60, 100, 150, 200), "cluster_share": (0, 25, 50, 75, 100),
                "start": ("random", "nearest")}
SWEEP_LABELS = {"budget": "Budget (bewertete Nachbarschaften)", "k_max": "Maximale Störstärke k_max", "local_search": "Lokale Suche",
                "n": "Stopps", "cluster_share": "Anteil in Gruppen (%)", "start": "Startlösung"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


SCALING_N = C.SCALING_N
SCALING_POLICIES = (("Budget 200 Tausend", lambda n: 200000), ("Budget 5 000 · Stopps", lambda n: 5000 * n))


def scaling_table(base=Settings()):
    return [{"label": label, "rows": [{"value": n, **run_config(base, n=n, budget=fn(n))} for n in SCALING_N]} for label, fn in SCALING_POLICIES]


def chain_spread(settings, k=C.SPREAD_CHAINS):
    """k Ketten-Seeds auf derselben Instanz: VNS (beste Tour) und ein Hill-Climbing-Abstieg (voller Rescan) aus derselben zufälligen Startlösung."""
    vns_gaps, hc_gaps = [], []
    for ch in range(k):
        a = analyse(replace(settings, chain_seed=ch, start="random"), keep_snapshots=False)
        vns_gaps.append(a.gap)
        hc_gaps.append(a.hc_gap)
    return {"vns": np.array(vns_gaps), "hc": np.array(hc_gaps)}
