"""Kandidatenlisten + Don't-Look-Bits fuer 2-opt: derselbe Zug wie in vns_tour, aber statt nach jedem Zug
alle Nachbarn neu zu bewerten, wird nur eine feste Kandidatenliste (die k naechsten Knoten) geprueft, und ein
Knoten kommt erst wieder in die Warteschlange, wenn eine seiner Kanten sich geaendert hat (Don't-Look-Bit).
Pruning: die Kandidaten sind nach Abstand sortiert, die Suche bricht ab, sobald d(c1,c2) >= die Laenge der zu
entfernenden Kante ist - eine Standard-Heuristik (Bentley 1992; Johnson und McGeoch), keine exakte Schranke
(siehe tests/test_dlb.py: bei kleinen Instanzen 100 % echte 2-opt-Optima, bei 60 Stopps nur noch rund die Haelfte).

Delta-Formel und Gueltigkeitspruefung sind wortgleich aus hc_algorithm._delta_2opt/_valid_pairs (hill-climbing-demo)
uebernommen (a = min(pos[c1], pos[c2]), b = max(...): delta = D[a,b] + D[a+1,b+1] - D[a,a+1] - D[b,b+1], gueltig
wenn b >= a+2 und nicht (a==0 und b==n-1)) - das vermeidet einen Vorzeichenfehler aus eigener Herleitung.
Ergebnis der Messreihe (2026-09-22, project_trajectory_metaheuristics_dag_scoping.md): bei 60 Stopps dieselbe
Guete wie der volle Abstieg (~7 % ueber der Schranke) mit rund 650 statt 74 000 bewerteten Nachbarn.

Erweiterung gegenueber hc_dlb.py/sa_dlb.py: der optionale Parameter `touched` seedet die Warteschlange nur mit
den uebergebenen Knoten (alle anderen starten mit gesetztem Don't-Look-Bit) statt mit allen n Knoten - der
Mechanismus, der den Wiederabstieg nach einer Doppelbruecken-Stoerung in Iterated Local Search billig macht:
nur die Umgebung der Stoerung wird neu durchsucht, nicht die ganze Tour. `touched=None` verhaelt sich exakt wie
das Original (voller Scan ab allen Knoten)."""

from collections import deque
from dataclasses import dataclass

import numpy as np

import vns_tour as A

CANDIDATE_K = 5


def build_candidate_lists(D, k=CANDIDATE_K):
    """Fuer jeden Knoten die k naechsten anderen Knoten, nach Abstand sortiert."""
    order = np.argsort(D, axis=1)
    return order[:, 1:k + 1]                                        # Spalte 0 ist die Diagonale (Abstand 0 zu sich selbst)


def _valid(a, b, n):
    return b >= a + 2 and not (a == 0 and b == n - 1)


@dataclass
class DlbResult:
    tour: np.ndarray
    length: float
    evaluations: int
    converged: bool                                                 # False, wenn max_evaluations vor der Konvergenz erreicht wurde


def dlb_descend(D, start, cand, seed=0, max_evaluations=None, touched=None):
    """Ein Kandidatenlisten- + Don't-Look-Bit-2-opt-Abstieg (erste Verbesserung). Eine Bewertung = ein geprueftes
    Kandidatenpaar (c1, c2) - dieselbe Einheit wie 'bewertete Nachbarn' im vollen Abstieg (vns_tour.descend).
    `touched` (optional): nur diese Knoten starten in der Warteschlange (Kurzweg fuer den Wiederabstieg nach
    einer lokalen Stoerung); `None` = alle Knoten (voller Scan, wie hc_dlb.py/sa_dlb.py)."""
    n = len(start)
    t = [int(x) for x in start]
    pos = [0] * n
    for idx, city in enumerate(t):
        pos[city] = idx
    rng = np.random.default_rng(seed)
    if touched is None:
        dontlook = [False] * n
        queue = deque(rng.permutation(n).tolist())
        in_queue = [True] * n
    else:
        dontlook = [True] * n
        start_nodes = rng.permutation(np.asarray(touched, dtype=np.int64)).tolist()
        queue = deque(start_nodes)
        in_queue = [False] * n
        for city in start_nodes:
            dontlook[city] = False
            in_queue[city] = True
    evaluations = 0
    length = A.tour_length(np.array(t), D)

    while queue:
        c1 = queue.popleft()
        in_queue[c1] = False
        if dontlook[c1]:
            continue
        improved = False
        i = pos[c1]
        for sign in (1, -1):
            anchor_pos = i if sign == 1 else (i - 1) % n             # Position des "linken" Endpunkts der Ankerkante
            nxt_pos = (anchor_pos + 1) % n
            d_anchor = D[t[anchor_pos], t[nxt_pos]]
            for c2 in cand[c1]:
                if max_evaluations is not None and evaluations >= max_evaluations:
                    return DlbResult(np.array(t), length, evaluations, False)
                evaluations += 1
                if D[c1, c2] >= d_anchor:
                    break                                            # sortierte Kandidaten: kein Gewinn mehr moeglich (Heuristik)
                j = pos[c2]
                p2 = j if sign == 1 else (j - 1) % n
                # zwei Kanten (anchor_pos,+1) und (p2,+1) teilen sich einen Knoten genau dann, wenn |anchor_pos-p2| == 1
                # (auch im Wrap-Fall 0/n-1) - das prueft _valid vollstaendig; p2 == anchor_pos kann nicht auftreten
                # (c2 != c1, pos ist eine Bijektion)
                a, b = (anchor_pos, p2) if anchor_pos < p2 else (p2, anchor_pos)
                if not _valid(a, b, n):
                    continue
                bp1 = (b + 1) % n                                    # b kann n-1 sein (bei a > 0): die Kante wickelt auf Position 0
                delta = D[t[a], t[b]] + D[t[a + 1], t[bp1]] - D[t[a], t[a + 1]] - D[t[b], t[bp1]]
                if delta < -1e-9:
                    t[a + 1:b + 1] = t[a + 1:b + 1][::-1]
                    for q in range(a + 1, b + 1):
                        pos[t[q]] = q
                    length += delta
                    improved = True
                    for city in (t[a], t[a + 1], t[b], t[(b + 1) % n]):
                        dontlook[city] = False
                        if not in_queue[city]:
                            queue.append(city)
                            in_queue[city] = True
                    break
            if improved:
                break
        if not improved:
            dontlook[c1] = True

    return DlbResult(np.array(t), length, evaluations, True)
