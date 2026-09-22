# Variable Neighborhood Search – eine Lieferrunde, deren Störstärke sich selbst anpasst – Streamlit-Demo

**[→ Demo live ausprobieren](#)** (Deploy offen)

Viertes Stück der **Trajektorien-Metaheuristiken-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
dieselbe Rundtour wie in der [hill-climbing-demo](../hill-climbing-demo), der [simulated-annealing-demo](../simulated-annealing-demo) und der [iterated-local-search-demo](../iterated-local-search-demo) (ein Depot, n Kundenstopps in einem 100 × 100-km-Gebiet), dieselbe untere Schranke.

**Einordnung in die Reihe:** **Variable Neighborhood Search** (VNS, Mladenović & Hansen 1997) baut direkt auf Iterated Local Search auf und löst dessen offene Frage: ILS brauchte eine **feste** Störstärke (in der Schwester-Demo von Hand kalibriert: 2-3 Doppelbrücken, nicht der literaturübliche Standardwert 1). VNS ersetzt den festen Regler durch ein systematisches Verfahren: die Störstärke $k$ startet bei 1, **wächst** nach einem erfolglosen Kick und wird nach einem **Erfolg auf 1 zurückgesetzt**.
```
hill-climbing-demo (Wurzel: nur bergab, bleibt im ersten Optimum stecken)        [gebaut]
  ├─ simulated-annealing-demo (nimmt Verschlechterungen an, Abkühlplan)          [gebaut]
  ├─ iterated-local-search-demo (stört ein gutes Optimum mit fester Störstärke)  [gebaut]
  │     └─ variable-neighborhood-search-demo (Störstärke eskaliert + Reset)     [dieses Stück]
  │           └─ ALNS (lernt, welcher Umbau sich lohnt, statt blind zu eskalieren)   [nicht gebaut]
  ├─ Tabu Search              (Gedächtnis gegen Rückwege)                        [nicht gebaut]
  └─ GRASP                    (randomisierte Konstruktion, viele Starts)         [nicht gebaut]
```

Ergebnis in Kürze: **mit einer vernünftig gewählten Obergrenze $k_{\max}$ schlägt VNS bei JEDEM gemessenen Budget sowohl die naive feste Störstärke 1 als auch den von Hand kalibrierten Sweet Spot (Störstärke 3) – oder liegt gleichauf.** 60 Stopps, 10 Tausend Vorschläge: VNS **1.23 %** über der Schranke gegen 1.56 % (Störstärke 1) und 1.35 % (Störstärke 3); bei 200 Tausend: 0.63 % gegen 0.66 % und 0.65 %.
Der eigentliche Überraschungsfund: **der "Sweet Spot" selbst ist nicht robust** – bei mittlerem Budget gewinnt Störstärke 3, ab etwa 500 Tausend Vorschlägen dreht es sich zugunsten der naiven Störstärke 1 (0.50 % gegen 0.54 % bei 1 Million). VNS trifft die jeweils bessere Wahl trotzdem, ohne sie vorher zu kennen. Zwei Einschränkungen: die Obergrenze $k_{\max}$ selbst braucht eine (kleinere) Kalibrierung (zu groß verschenkt bei knappem Budget), und das **Zurücksetzen** bei Erfolg ist der eigentliche Hebel – ohne es verliert VNS klar (0.72 % gegen 0.63 % bei 200 Tausend).

| Frage | Ergebnis (60 gleichverteilte Stopps, Kandidatenliste + DLB, k_max=5, zufällige Startlösung; Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Ketten-Seeds; Abstand = Prozent über der 1-Baum-Schranke) |
|---|---|
| Standardfall | ✅ VNS **0.63 %** über der Schranke gegen ILS Störstärke 1 (naiv) 0.66 %, ILS Störstärke 3 (Sweet Spot) 0.65 %, HC-Neustarts (Kandidatenliste + DLB) 0.68 % |
| **Budget** | ✅ VNS bei 10 / 25 / 50 / 100 / 200 Tausend / 0.5 / 1 / 2 Millionen: **1.23 / 0.85 / 0.80 / 0.77 / 0.63 / 0.56 / 0.47 / 0.47 %** – schlägt oder erreicht bei JEDEM Budget die jeweils bessere feste Störstärke |
| **Der Sweet Spot selbst verschiebt sich** | ❗ ILS Störstärke 3 schlägt Störstärke 1 bei 25-200 Tausend (z. B. 0.65 gegen 0.66 % bei 200T); ab 500 Tausend dreht es sich (0.58 gegen 0.65 % bei 500T, 0.50 gegen 0.54 % bei 1M) – eine feste Störstärke ist kein robuster Fixpunkt |
| **k_max** | ⚠️ Bei 10 Tausend Vorschlägen: k_max=2 → 1.71 %, k_max=3 → **1.19 %** (bestes gemessen), k_max=5 → 1.23 % (Voreinstellung), k_max=8 → 1.70 % – ein Sweet Spot, kein "größer ist besser" |
| **Zurücksetzen** | ❗ Mit Reset (echtes VNS) 0.63 %, ohne Reset (k eskaliert bei Erfolg NICHT zurück) 0.72 % (200 Tausend) – das Zurücksetzen ist der eigentliche Hebel, nicht nur das Eskalieren |
| **Lokale Suche** | ❌❗ Voller Rescan statt Kandidatenliste + DLB: **4.32 %** bei nur rund **8** Iterationen statt 0.63 % bei rund **974** – dieselbe Lehre wie in der ganzen Linie |
| **Startlösung** | ➖ Zufällig 0.63 %, Nächster Nachbar 0.63 % über der Schranke – kein messbarer Unterschied |
| **Größe** | ✅ 200 Stopps, 1 Million Vorschläge: **1.9 %** gegen **4.71 %** für Hill Climbing mit Neustarts (Kandidatenliste + DLB) |

## Was die Demo zeigt

1. **VNS in Aktion** (Schritt-Slider + Abspielen): **Instanz** → **Erster Abstieg** → **Shakes** (Störstärke-$k$-Verlauf als Treppenkurve + Iterations-Regler + ▶️ Shakes abspielen: Länge der aktuellen/besten Tour, dazu die Tour nach der gewählten Iteration) → **Ergebnis** (beste Tour neben der besten aus Iterated Local Search mit Störstärke 3).
2. **Was die Kette gefunden hat:** beste und letzte Tour, ILS Störstärke 1 und 3, erfolgreiche Iterationen; Urteil (`beats_ils3` → `comparable` → `ils3_wins`), Detailtabellen.
3. **📐 Sweeps** über Budget, k_max, lokale Suche, Stopps, Gruppen und Startlösung (feste Instanzen ab 100000, drei Ketten je Instanz).
4. **🔬 Experimente auf Abruf:** Budget von 10 Tausend bis 2 Millionen (VNS gegen beide feste ILS-Störstärken); **k_max** (1 bis 8 bei knappem Budget – der Sweet-Spot-Fund); **Zurücksetzen** (mit gegen ohne Reset); **Streuung** über 20 Ketten; **Skalierung** von 20 bis 200 Stopps.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (das Budget reicht für mehrere Eskalationszyklen, k_max ist selbst kalibriert, das Zurücksetzen passiert, die lokale Suche ist billig, die Störung ist strukturell).

Regler: Stopps (10–200), Anteil der Stopps in Gruppen, **Lokale Suche** (Kandidatenliste + DLB / voller Rescan), **Maximale Störstärke k_max** (1–8),
**Budget** (10 Tausend bis 2 Millionen bewertete Nachbarschaften), **Startlösung**, Seed der Instanz (+ 🎲), Seed der Kette (+ 🎲).

## Messwerte der Presets (Instanz-Seed 35, Ketten-Seed 0; sie prüfen sich mit Urteil-Bändern selbst)

Die einzelne Standardinstanz landet auch hier durch Zufall oft sehr nah am echten Optimum; die Mittelwerte oben in der Tabelle sind die belastbaren Zahlen (siehe App-Hilfetexte für die Presets im Detail).

| Preset | Urteil (Band über Instanzen × Ketten) |
|---|---|
| Standardfall (Voreinstellung) | beats_ils3 / comparable |
| Kleines Budget (10 Tausend) | beats_ils3 / comparable / ils3_wins |
| Voller Rescan (langsam) | ils3_wins / comparable |
| k_max zu groß (8) | beats_ils3 / comparable / ils3_wins |
| k_max zu klein (2) | beats_ils3 / comparable / ils3_wins |
| Nächster Nachbar als Start | beats_ils3 / comparable / ils3_wins |
| Großes Budget (1 Million) | beats_ils3 / comparable |
| Große Instanz (200 Stopps, 1 Million) | beats_ils3 / comparable / ils3_wins |

## Modell und Verfahren

- **Instanz, Nachbarschaften, Abstieg, Schranke, Doppelbrücke, Kandidatenliste + Don't-Look-Bits** (`vns_scenario.py`, `vns_tour.py`, `vns_kick.py`, `vns_dlb.py`): wortgleiche Kopien aus der [iterated-local-search-demo](../iterated-local-search-demo) (die ihrerseits aus der [hill-climbing-demo](../hill-climbing-demo)/[simulated-annealing-demo](../simulated-annealing-demo) stammen) – gegen eingefrorene Werte testen, wie bei jedem bisherigen Stück.
- **Basic VNS** (`vns_algorithm.py`): erster Abstieg aus der Startlösung, dann: Shake mit $k$ Doppelbrücken → Wiederabstieg → bei Erfolg übernehmen und $k \leftarrow 1$, sonst verwerfen und $k \leftarrow k+1$ (bei $k > k_{\max}$: zurück auf 1). VND (mehrere Nachbarschaften als lokale Suche) ist bewusst NICHT gebaut – laut DAG-Planung kein eigener Knoten, höchstens ein Umschalter, den dieses Stück nicht braucht.
- **Vergleichsgrößen ohne Cross-Import** (`vns_evaluation.py`): `ils_fixed_strength` (Kopie der ILS-Schleife mit fester Störstärke 1 bzw. 3), `hill_climbing_restarts`/`dlb_restarts` (wortgleich aus der ILS-Demo) – jedes Repo der Linie ist eigenständig lauffähig.
- **Auswertung**: Kennzahlen, Urteil, Sweeps über feste Instanzen × Ketten, Reset-Ablation, Streuung, Skalierung.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutung: "VNS erreicht den von Hand gefundenen ILS-Sweet-Spot, ohne ihn zu kennen"** – **mehr als bestätigt**: mit dem kalibrierten $k_{\max}=5$ schlägt VNS bei JEDEM gemessenen Budget beide festen Störstärken oder liegt gleichauf. Der eigentliche Überraschungsfund ging noch weiter: **der Sweet Spot selbst ist nicht robust über das Budget** – Störstärke 3 gewinnt bei 25-200 Tausend, ab 500 Tausend gewinnt die naive Störstärke 1 (0.50 gegen 0.54 % bei 1 Million). Ein fester Regler wäre also selbst bei sorgfältiger Kalibrierung fragil gegenüber Budgetänderungen; VNS umgeht dieses Problem, ohne es zu kennen.
- **$k_{\max}$ ist selbst ein Kalibrierungsparameter, nur ein kleinerer.** Bei knappem Budget (10 Tausend) ist $k_{\max}=8$ fast so schlecht wie $k_{\max}=2$ (1.70 gegen 1.71 %), $k_{\max}=3$-$5$ liegt klar davor (1.19-1.23 %) – ein erster Entwurf mit $k_{\max}=8$ (der maximale Bereich des Reglers) hätte VNS bei knappem Budget schlechter als die naive feste Störstärke aussehen lassen; erst die eigene Kalibrierungsmessung deckte das auf und führte zum Default $k_{\max}=5$.
- **Das Zurücksetzen ist der eigentliche Hebel, nicht nur das Eskalieren.** Die Ablation "ohne Reset" (k eskaliert bei Erfolg nicht zurück auf 1) verliert bei jedem gemessenen Budget gegen die echte Regel (200 Tausend: 0.72 gegen 0.63 %) – wer nur "wachsende Störstärke" ohne Rückfall implementiert, verschenkt einen wesentlichen Teil des Verfahrens.
- **Startlösung ist bei VNS fast egal** (0.63 % gegen 0.63 %), wie schon bei Iterated Local Search – die vielen Shakes vergessen sie schneller, als ein einzelner Abstieg es könnte.
- **Synthetische Instanzen:** euklidisch, gleichverteilt oder in fünf Gruppen, ein Fahrzeug, keine Kapazitäten oder Zeitfenster. Zeiten hängen vom Rechner und der Python-Version ab (die Tests prüfen nur Größenordnungen).

## Verifikation

- **Übernommener Kern** (Doppelbrücke, Kandidatenliste + Don't-Look-Bits, 2-opt/Schranke): wortgleiche Kopien der bereits geprüften Korrektheitstests aus der Iterated-Local-Search-Demo.
- **VNS-Schleife:** unabhängige Nachrechnung des $k$-Verlaufs aus den mitgeführten Momentaufnahmen (ob eine Iteration ein Erfolg war, lässt sich an der Längenänderung ablesen; daraus muss sich exakt der nächste $k$-Wert ergeben – Reset auf 1 bei Erfolg, +1 sonst, Deckelung bei $k_{\max}$); Budget-Buchführung, Monotonie der aktuellen Länge, Determinismus je Seed, Regressionsschutz für beide lokalen Suchen, die Reset-Ablation ändert nachweislich nur den $k$-Verlauf nach einem Erfolg.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Budget-, k_max-, Reset- und Größen-Aussagen; jeweils Mittel über die festen Sweep-Instanzen × Ketten; positive **und** negative Aussagen; Rechenzeiten nur als Größenordnung); alle 8 Presets über mehrere Instanzen und Ketten in Urteil-Bändern; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt bei 10 und 60 Stopps, Iterations-Regler, ▶️ Abspielen und ▶️ Shakes abspielen ohne doppelte Schlüssel, Würfel-Knöpfe, Permalink-Grenzen, Extremwerte, Experimente auf Abruf, Footer).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Budget, k_max, Zurücksetzen, Streuung, Skalierung), 🚧 Grenzen, Mathe |
| `vns_kick.py` | Doppelbrücken-Zug (wortgleich aus der ILS-Demo) |
| `vns_dlb.py` | Kandidatenliste + Don't-Look-Bits für 2-opt, mit `touched`-Kurzweg |
| `vns_algorithm.py` | Die VNS-Schleife: erster Abstieg, Shake mit Störstärke k, Wiederabstieg, Erfolg → Reset auf k=1, Misserfolg → k+1 |
| `vns_tour.py` | Nachbarschaften, Abstieg (mit Bewertungsbudget), Kreuzungen, 1-Baum-Schranke |
| `vns_scenario.py`, `vns_constants.py` | Instanzen; Konstanten, Presets |
| `vns_evaluation.py` | Analyse, Urteil, ILS mit fester Störstärke (Vergleichsgröße), Hill Climbing mit Neustarts (beide Varianten), Sweeps, Reset-Ablation, Streuung, Skalierung |
| `vns_presets.py`, `vns_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Doppelbrücken-Zug, Kandidatenliste + Don't-Look-Bits, VNS-Schleife, übernommener Kern, Szenario und Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
