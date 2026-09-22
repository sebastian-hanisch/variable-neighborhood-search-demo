"""Der Doppelbrücken-Zug (double bridge): die Störung von Iterated Local Search. Anders als ein zufälliger 2-opt-Zug, den der anschließende
Abstieg sofort wieder rückgängig machen kann, lässt sich eine Doppelbrücke durch keinen einzelnen 2-opt-Zug rückgängig machen (siehe
tests/test_kick.py: erschöpfende Prüfung gegen alle 2-opt-Züge auf kleinen Instanzen) - das ist der eigentliche Grund, warum Iterated Local
Search eine eigene Störung braucht und keine zufällige Nachbarschaftswahl genügt.

Drei Schnittpunkte 1 <= p1 < p2 < p3 <= n-1 teilen die Tour in vier Stücke S1..S4 (S1 vorn, S4 hinten, zyklisch). Die Störung setzt sie in der
Reihenfolge S1, S3, S2, S4 neu zusammen (Stücke S2 und S3 tauschen den Platz, ihre innere Reihenfolge bleibt erhalten). Sind BEIDE mittleren
Stücke genau ein Knoten lang, entartet der Zug zum Tausch zweier benachbarter Knoten - das IST ein einzelner 2-opt-Zug (Segment der Länge 2
umkehren) und widerspräche dem Zweck der Doppelbrücke. MIN_SEGMENT erzwingt deshalb eine Mindestlänge je Stück (2, wenn die Tour groß genug
dafür ist, sonst 1 als Rückfall für sehr kleine Instanzen)."""

import numpy as np

MIN_SEGMENT = 2


def double_bridge(tour, rng, n_bridges=1):
    """`n_bridges` Doppelbrücken-Züge nacheinander auf `tour` (Permutation, numpy-Array). Gibt (neue Tour, betroffene Knoten) zurück -
    die betroffenen Knoten sind die Endpunkte der neuen Kanten aller angewandten Züge (Eingabe für den `touched`-Kurzweg von ils_dlb.dlb_descend)."""
    t = np.asarray(tour).copy()
    n = len(t)
    if n < 4:
        raise ValueError("Doppelbrücke braucht mindestens 4 Knoten (drei Schnittpunkte in vier nichtleere Stücke)")
    L = MIN_SEGMENT if n >= 4 * MIN_SEGMENT else 1
    touched = set()
    for _ in range(n_bridges):
        for _attempt in range(200):
            p1, p2, p3 = sorted(rng.choice(np.arange(1, n), size=3, replace=False).tolist())
            if p1 >= L and p2 - p1 >= L and p3 - p2 >= L and n - p3 >= L:
                break
        s1, s2, s3, s4 = t[:p1], t[p1:p2], t[p2:p3], t[p3:]
        touched.update(int(x) for x in (s1[-1], s3[0], s3[-1], s2[0], s2[-1], s4[0]))
        t = np.concatenate([s1, s3, s2, s4])
    return t, sorted(touched)
