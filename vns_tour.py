"""Wortgleiche Kopie des Hill-Climbing-Kerns der hill-climbing-demo (Nachbarschaften, Abstieg, Kreuzungen, 1-Baum-Schranke), dazu das Bewertungsbudget `max_evaluations` in `descend`.

Hill Climbing (lokale Suche) für eine einzelne Rundtour (TSP), numpy von Grund auf.

Eine Tour ist eine Permutation der Knoten 0..N-1 (Knoten 0 = Depot), als geschlossener Zyklus gelesen. Vier Nachbarschaften: Tausch zweier nicht benachbarter Stopps, 2-opt (ein Stück umkehren),
Or-opt (Stück aus 1-3 Stopps an anderer Stelle einfügen, auch umgekehrt) und die Vereinigung 2-opt + Or-opt. Zwei Auswahlregeln: erste Verbesserung (erster verbessernder Nachbar in fester Reihenfolge)
oder beste Verbesserung (steilster Abstieg). Alle Züge werden je Suchdurchgang als N x N-Matrizen der Längenänderung (Delta) bewertet; die Zahl der *logisch* bewerteten Nachbarn wird mitgezählt.
Daneben: Startlösungen, Kreuzungszählung, 1-Baum-Schranke (Held-Karp, Subgradientenverfahren) und Mehrfachstart."""

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-9
NEIGHBORHOODS = ("swap", "2opt", "oropt", "2opt+oropt")
RULES = ("first", "best")
STARTS = ("random", "nearest", "input")
MAX_SEGMENT = 3


# --- Grundlagen ---------------------------------------------------------------------------------------------------------------------------------


def dist_matrix(xy):
    xy = np.asarray(xy, dtype=float)
    d = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((d * d).sum(axis=2))


def tour_length(tour, D):
    t = np.asarray(tour)
    return float(D[t, np.roll(t, -1)].sum())


def canonical(tour):
    """Zyklus so drehen, dass das Depot (Knoten 0) vorn steht, und so richten, dass der kleinere Nachbar des Depots folgt (gleiche Tour = gleiche Liste)."""
    t = np.asarray(tour)
    t = np.roll(t, -int(np.where(t == 0)[0][0]))
    if len(t) > 2 and t[1] > t[-1]:
        t = np.concatenate([t[:1], t[:0:-1]])
    return t


def tour_edges(tour):
    t = np.asarray(tour)
    a, b = t, np.roll(t, -1)
    return {(int(min(x, y)), int(max(x, y))) for x, y in zip(a, b)}


def edge_share(tour, reference):
    """Anteil der Kanten von `tour`, die auch in `reference` vorkommen."""
    return len(tour_edges(tour) & tour_edges(reference)) / len(tour)


# --- Startlösungen -------------------------------------------------------------------------------------------------------------------------------


def random_tour(n_nodes, rng):
    rest = rng.permutation(np.arange(1, n_nodes))
    return np.concatenate([[0], rest]).astype(np.int64)


def nearest_neighbor_tour(D):
    n = len(D)
    tour = [0]
    left = np.ones(n, dtype=bool)
    left[0] = False
    cur = 0
    for _ in range(n - 1):
        cand = np.where(left)[0]
        nxt = int(cand[np.argmin(D[cur, cand])])
        tour.append(nxt)
        left[nxt] = False
        cur = nxt
    return np.array(tour, dtype=np.int64)


def start_tour(kind, D, rng=None):
    if kind == "random":
        return random_tour(len(D), rng if rng is not None else np.random.default_rng(0))
    if kind == "nearest":
        return nearest_neighbor_tour(D)
    if kind == "input":
        return np.arange(len(D), dtype=np.int64)
    raise ValueError(kind)


# --- Züge: Bewertung ---------------------------------------------------------------------------------------------------------------------------
# Ein Zug ist ein Tupel: ("2opt", i, j) | ("swap", i, j) | ("oropt", i, L, m, reversed). Positionen sind Indizes in der Tour (zyklisch).


def _valid_pairs(n):
    """Paare i < j mit j >= i+2 ohne (0, N-1) (die beiden Kanten würden einen Knoten teilen)."""
    i, j = np.indices((n, n))
    ok = j >= i + 2
    ok[0, n - 1] = False
    return ok


def _delta_2opt(t, D):
    n = len(t)
    nxt = np.roll(t, -1)
    e = D[t, nxt]
    delta = D[np.ix_(t, t)] + D[np.ix_(nxt, nxt)] - e[:, None] - e[None, :]
    return delta, _valid_pairs(n)


def _delta_swap(t, D):
    n = len(t)
    prev, nxt = np.roll(t, 1), np.roll(t, -1)
    r = D[prev, t] + D[t, nxt]
    # added[i, j] = D[P_i, t_j] + D[t_j, N_i] + D[P_j, t_i] + D[t_i, N_j]
    a1 = D[np.ix_(prev, t)]                    # [i, j] = D[P_i, t_j]
    a2 = D[np.ix_(nxt, t)]                     # [i, j] = D[N_i, t_j]
    a3 = D[np.ix_(t, prev)]                    # [i, j] = D[t_i, P_j]
    a4 = D[np.ix_(t, nxt)]                     # [i, j] = D[t_i, N_j]
    added = a1 + a2 + a3 + a4
    return added - r[:, None] - r[None, :], _valid_pairs(n)


def _delta_oropt(t, D, length):
    """delta[i, m]: das Stück der Länge `length` ab Position i wird zwischen t[m] und t[m+1] eingefügt (bessere der beiden Richtungen); zweite Matrix: True = umgekehrt eingefügt."""
    n = len(t)
    idx = np.arange(n)
    s0, s1 = t, np.roll(t, -(length - 1))
    p, nx = np.roll(t, 1), np.roll(t, -length)
    u, v = t, np.roll(t, -1)
    gain = D[p, s0] + D[s1, nx] - D[p, nx]                                  # gespart beim Herausnehmen
    e = D[u, v]
    fwd = D[np.ix_(s0, u)] + D[np.ix_(s1, v)]                               # u, s0..s1, v
    rev = D[np.ix_(s1, u)] + D[np.ix_(s0, v)]                               # u, s1..s0, v
    reversed_ = rev < fwd - EPS
    add = np.where(reversed_, rev, fwd) - e[None, :]
    delta = add - gain[:, None]
    ok = ((idx[None, :] - (idx[:, None] - 1)) % n) >= length + 1            # Kante m nicht am Stück beteiligt
    return delta, ok, reversed_


def _first_or_best(delta, ok, rule):
    """Index (flach) des gewählten Zuges und Zahl der dabei bewerteten Nachbarn; (None, Zahl) falls keiner verbessert."""
    d = np.where(ok, delta, np.inf)
    flat = d.ravel()
    if rule == "best":
        k = int(np.argmin(flat))
        return (k if flat[k] < -EPS else None), int(ok.sum())
    improving = flat < -EPS
    if not improving.any():
        return None, int(ok.sum())
    k = int(np.argmax(improving))
    return k, int(ok.ravel()[:k + 1].sum())


def _candidates(t, D, neighborhood):
    """Liste (Art, Delta, gültig, Zusatz) in der festen Reihenfolge der Nachbarschaft."""
    out = []
    if neighborhood in ("swap",):
        d, ok = _delta_swap(t, D)
        out.append(("swap", d, ok, None))
    if neighborhood in ("2opt", "2opt+oropt"):
        d, ok = _delta_2opt(t, D)
        out.append(("2opt", d, ok, None))
    if neighborhood in ("oropt", "2opt+oropt"):
        for length in range(1, MAX_SEGMENT + 1):
            if len(t) < length + 3:
                continue
            d, ok, rev = _delta_oropt(t, D, length)
            out.append((f"oropt{length}", d, ok, rev))
    return out


def neighbor_deltas(t, D, neighborhood):
    """Längenänderung aller Nachbarn der Tour `t` in der Nachbarschaft (negativ = verbessernd)."""
    return np.concatenate([delta[ok] for _, delta, ok, _ in _candidates(np.asarray(t), D, neighborhood)])


def find_move(t, D, neighborhood, rule):
    """Nächster Zug (Zug, Delta) oder (None, 0.0), dazu die Zahl bewerteter Nachbarn."""
    t = np.asarray(t)
    n = len(t)
    evaluations = 0
    cands = _candidates(t, D, neighborhood)
    if rule == "first":
        for kind, delta, ok, extra in cands:
            k, count = _first_or_best(delta, ok, "first")
            evaluations += count
            if k is not None:
                return _make_move(kind, k, n, delta, extra), evaluations
        return None, evaluations
    best = None
    for kind, delta, ok, extra in cands:
        k, count = _first_or_best(delta, ok, "best")
        evaluations += count
        if k is not None and (best is None or delta.ravel()[k] < best[0] - EPS):
            best = (float(delta.ravel()[k]), kind, k, delta, extra)
    if best is None:
        return None, evaluations
    return _make_move(best[1], best[2], n, best[3], best[4]), evaluations


def _make_move(kind, k, n, delta, extra):
    i, j = divmod(k, n)
    d = float(delta[i, j])
    if kind == "2opt" or kind == "swap":
        return (kind, i, j, d)
    length = int(kind[-1])
    return ("oropt", i, length, j, bool(extra[i, j]), d)


# --- Züge: Ausführen -----------------------------------------------------------------------------------------------------------------------------


def apply_move(t, move):
    t = np.asarray(t)
    kind = move[0]
    if kind == "2opt":
        i, j = move[1], move[2]
        out = t.copy()
        out[i + 1:j + 1] = t[i + 1:j + 1][::-1]
        return out
    if kind == "swap":
        i, j = move[1], move[2]
        out = t.copy()
        out[i], out[j] = t[j], t[i]
        return out
    if kind == "oropt":
        i, length, m, reverse = move[1], move[2], move[3], move[4]
        n = len(t)
        seg = np.roll(t, -i)[:length]
        rest = np.roll(t, -(i + length))[:n - length]                     # beginnt nach dem Stück, endet davor
        q = (m - (i + length)) % n                                        # Position von t[m] in `rest`
        if reverse:
            seg = seg[::-1]
        return np.concatenate([rest[:q + 1], seg, rest[q + 1:]])
    raise ValueError(kind)


def move_edges(t, move):
    """(entfernte Kanten, neue Kanten) eines Zuges als Knotenpaare - für die Darstellung."""
    t = np.asarray(t)
    n = len(t)
    new = apply_move(t, move)
    old_e, new_e = tour_edges(t), tour_edges(new)
    return sorted(old_e - new_e), sorted(new_e - old_e)


# --- Abstieg -----------------------------------------------------------------------------------------------------------------------------------


@dataclass
class Step:
    move: object            # None beim Start
    tour: np.ndarray
    length: float


@dataclass
class Descent:
    tour: np.ndarray
    length: float
    steps: list = field(default_factory=list)      # Start + ein Eintrag je Zug (bei keep_steps=False nur der Start)
    evaluations: int = 0
    n_moves: int = 0
    kinds: dict = field(default_factory=dict)
    neighborhood: str = ""
    rule: str = ""

    def moves_by_kind(self):
        return dict(self.kinds)


def descend(D, tour, neighborhood="2opt", rule="first", max_moves=100000, keep_steps=True, max_evaluations=None):
    """Verbessernde Züge, bis keiner mehr existiert (lokales Optimum), `max_moves` oder das Bewertungsbudget `max_evaluations` erreicht sind."""
    if neighborhood not in NEIGHBORHOODS:
        raise ValueError(neighborhood)
    if rule not in RULES:
        raise ValueError(rule)
    t = np.asarray(tour, dtype=np.int64).copy()
    length = tour_length(t, D)
    steps = [Step(None, t.copy(), length)]
    evaluations, n_moves, kinds = 0, 0, {}
    for _ in range(max_moves):
        if max_evaluations is not None and evaluations >= max_evaluations:        # Bewertungsbudget aufgebraucht (die Suche endet nach der Suche, die es überschreitet)
            break
        move, count = find_move(t, D, neighborhood, rule)
        evaluations += count
        if move is None:
            break
        t = apply_move(t, move)
        length += move[-1]
        n_moves += 1
        kinds[move[0]] = kinds.get(move[0], 0) + 1
        if keep_steps:
            steps.append(Step(move, t.copy(), length))
    length = tour_length(t, D)                                            # Rundungsfehler der Delta-Summe beseitigen
    return Descent(t, length, steps, evaluations, n_moves, kinds, neighborhood, rule)


def is_local_optimum(D, tour, neighborhood):
    move, _ = find_move(np.asarray(tour), D, neighborhood, "best")
    return move is None


# --- Kreuzungen --------------------------------------------------------------------------------------------------------------------------------


def count_crossings(xy, tour):
    """Zahl der Paare von Tourkanten, die sich echt schneiden (Kanten mit gemeinsamem Knoten zählen nicht)."""
    xy = np.asarray(xy, dtype=float)
    t = np.asarray(tour)
    n = len(t)
    p = xy[t]
    q = xy[np.roll(t, -1)]

    def orient(a, b, c):
        return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    a, b = p[:, None, :], q[:, None, :]
    c, d = p[None, :, :], q[None, :, :]
    o1, o2 = np.sign(orient(a, b, c)), np.sign(orient(a, b, d))
    o3, o4 = np.sign(orient(c, d, a)), np.sign(orient(c, d, b))
    cross = (o1 * o2 < 0) & (o3 * o4 < 0)
    i, j = np.indices((n, n))
    cross &= (j > i)
    return int(cross.sum())


# --- 1-Baum-Schranke (Held-Karp) ---------------------------------------------------------------------------------------------------------


def _one_tree(C):
    """Minimaler 1-Baum: Spannbaum über Knoten 1..N-1 (Prim) + die zwei billigsten Kanten von Knoten 0. Rückgabe: Kosten, Grad je Knoten."""
    n = len(C)
    deg = np.zeros(n)
    m = n - 1
    in_tree = np.zeros(m, dtype=bool)
    key = C[1:, 1:][0].copy()
    parent = np.zeros(m, dtype=np.int64)
    in_tree[0] = True
    key[0] = np.inf
    total = 0.0
    sub = C[1:, 1:]
    for _ in range(m - 1):
        k = np.where(in_tree, np.inf, key)
        v = int(np.argmin(k))
        total += key[v]
        deg[v + 1] += 1
        deg[parent[v] + 1] += 1
        in_tree[v] = True
        upd = (~in_tree) & (sub[v] < key)
        key[upd] = sub[v][upd]
        parent[upd] = v
    two = np.argsort(C[0, 1:])[:2] + 1
    total += C[0, two].sum()
    deg[0] += 2
    deg[two] += 1
    return float(total), deg


def held_karp_bound(D, upper, iterations=300):
    """Untere Schranke für die kürzeste Rundtour: max über Knotengewichte pi von (1-Baum-Kosten - 2 * sum(pi)); Subgradientenverfahren mit Polyak-Schrittweite (Ziel `upper`)."""
    n = len(D)
    if n < 4:
        return tour_length(np.arange(n), D)
    pi = np.zeros(n)
    best = -np.inf
    lam = 2.0
    stall = 0
    for _ in range(iterations):
        C = D + pi[:, None] + pi[None, :]
        cost, deg = _one_tree(C)
        value = cost - 2.0 * pi.sum()
        if value > best + 1e-9:
            best = value
            stall = 0
        else:
            stall += 1
            if stall >= 15:
                lam *= 0.5
                stall = 0
        g = deg - 2.0
        norm = float((g * g).sum())
        if norm == 0.0 or lam < 1e-6:
            break
        pi = pi + lam * max(upper - value, 1e-9) / norm * g
    return float(best)


# --- Mehrfachstart -----------------------------------------------------------------------------------------------------------------------------


def multi_start(D, k, seed, neighborhood="2opt", rule="first", start="random"):
    """k Abstiege aus zufälligen Startlösungen (nur die Endtouren und Kennzahlen werden behalten). Gibt (Längen, Touren, Züge, Bewertungen) zurück."""
    rng = np.random.default_rng(seed)
    lengths, tours, moves, evals = [], [], [], []
    for _ in range(k):
        r = descend(D, random_tour(len(D), rng) if start == "random" else start_tour(start, D, rng), neighborhood, rule, keep_steps=False)
        lengths.append(r.length)
        tours.append(canonical(r.tour))
        moves.append(r.n_moves)
        evals.append(r.evaluations)
    return np.array(lengths), tours, np.array(moves), np.array(evals)
