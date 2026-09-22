"""Variable Neighborhood Search (Basic VNS, Mladenović & Hansen 1997) für eine Rundtour (TSP): wie Iterated Local Search wird ein bereits
gutes lokales Optimum mit einer Doppelbrücke gestört und neu abgestiegen - aber die Störstärke k ist nicht fest, sondern wandert selbst:
nach einem ERFOLG (die neue Tour ist besser) wird sie auf 1 zurückgesetzt (zurück zur schwächsten Störung); nach einem MISSERFOLG wird
sie erhöht (stärker stören, in der Hoffnung, aus der Umgebung des aktuellen Optimums herauszukommen), bis k_max erreicht ist, dann
zurück auf 1. Kein Regler für die "richtige" Störstärke nötig - das ist der ganze Witz gegenüber Iterated Local Search mit fester
Störstärke. Budget = bewertete Nachbarschaften (wie in der ganzen Trajektorien-Metaheuristiken-Linie); der Shake selbst zählt nicht."""

from dataclasses import dataclass, field

import numpy as np

import vns_dlb as DLB
import vns_kick as K
import vns_tour as T

LOCAL_SEARCHES = ("dlb", "full")


@dataclass
class Run:
    best_tour: np.ndarray
    best_length: float
    final_tour: np.ndarray
    final_length: float
    evaluations: int = 0
    iterations: int = 0
    successes: int = 0
    snapshots: list = field(default_factory=list)          # aktuelle Tour nach jeder Iteration (nur bei keep_snapshots=True)
    k_trace: list = field(default_factory=list)             # Störstärke k VOR jeder Iteration (dieselbe Länge wie snapshots[1:])
    trace_iter: np.ndarray = None
    trace_length: np.ndarray = None                        # aktuelle (angenommene) Länge über die Iterationen
    trace_best: np.ndarray = None
    local_search: str = "dlb"
    k_max: int = 8
    reset_on_success: bool = True


def run(D, start, cand=None, local_search="dlb", k_max=8, reset_on_success=True, budget=100000, seed=0, keep_snapshots=True, trace_points=300):
    """Ein Lauf. `cand` (Kandidatenlisten aus vns_dlb.build_candidate_lists) ist nur für `local_search="dlb"` nötig.
    `reset_on_success=False` ist die Ablation für das Experiment "Ohne Reset" (k eskaliert, wird bei Erfolg aber NICHT zurückgesetzt) -
    kein Sidebar-Regler, nur zum Messen, ob das Zurücksetzen selbst der Hebel ist."""
    if local_search not in LOCAL_SEARCHES:
        raise ValueError(local_search)
    if local_search == "dlb" and cand is None:
        raise ValueError("local_search='dlb' braucht Kandidatenlisten (cand)")
    if k_max < 1:
        raise ValueError(k_max)
    rng = np.random.default_rng(seed)
    evaluations = iterations = successes = 0
    trace_every = max(1, budget // trace_points)

    # Erster Abstieg aus der Startlösung (touched=None: die ganze Tour muss optimiert werden) - das eigentliche VNS
    # beginnt erst danach, mit Shakes AUF einem bereits lokal optimalen Tour.
    init_seed = int(rng.integers(0, 2**31 - 1))
    if local_search == "dlb":
        r0 = DLB.dlb_descend(D, start, cand, seed=init_seed, max_evaluations=budget)
        current, current_length = r0.tour, r0.length
    else:
        r0 = T.descend(D, start, "2opt", "first", keep_steps=False, max_evaluations=budget)
        current, current_length = r0.tour, r0.length
    evaluations += r0.evaluations
    best_tour, best_length = current.copy(), current_length
    snapshots = [current.copy()] if keep_snapshots else []
    k_trace = []
    tr_it, tr_len, tr_best = [evaluations], [current_length], [current_length]

    k = 1
    while evaluations < budget:
        remaining = budget - evaluations
        shaken, touched = K.double_bridge(current, rng, n_bridges=k)
        descend_seed = int(rng.integers(0, 2**31 - 1))
        if local_search == "dlb":
            r = DLB.dlb_descend(D, shaken, cand, seed=descend_seed, max_evaluations=remaining, touched=touched)
        else:
            r = T.descend(D, shaken, "2opt", "first", keep_steps=False, max_evaluations=remaining)
        candidate_tour, candidate_length, spent = r.tour, r.length, r.evaluations
        evaluations += spent
        iterations += 1
        k_trace.append(k)
        success = candidate_length < current_length - 1e-9
        if success:
            current, current_length = candidate_tour, candidate_length
            successes += 1
            if reset_on_success:
                k = 1
        else:
            k = k + 1
            if k > k_max:
                k = 1
        if current_length < best_length - 1e-9:
            best_length, best_tour = current_length, current.copy()
        if keep_snapshots:
            snapshots.append(current.copy())
        if iterations % trace_every == 0 or evaluations >= budget:
            tr_it.append(evaluations)
            tr_len.append(current_length)
            tr_best.append(best_length)

    final_length = T.tour_length(current, D)                          # Rundungsfehler der Delta-Summen beseitigen
    best_length = T.tour_length(best_tour, D)
    return Run(best_tour, best_length, current, final_length, evaluations, iterations, successes, snapshots, k_trace,
               np.array(tr_it), np.array(tr_len), np.array(tr_best), local_search, k_max, reset_on_success)
