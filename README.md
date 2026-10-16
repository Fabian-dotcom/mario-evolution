# Mario Neuroevolution Lab

Viele Mario-Instanzen spielen **gleichzeitig** das echte
_Super Mario Bros._ (emuliert mit [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)),
steuert von selbst entwickelten, vollständig verbundenen
**Feedforward-Neuronalen Netzen**. Die Gewichte und Bias-Werte dieser
Netze werden durch einen **evolutionären Algorithmus** (Selektion +
Mutation, _kein_ Backpropagation) von Generation zu Generation besser.
Ein **Tkinter-Dashboard** zeigt alles live auf einem Bildschirm.

Die Theorie folgt _The Nature of Code_ (Kapitel 10: Neuronale Netze,
Kapitel 11: Neuroevolution/Genetische Algorithmen) und ist im Code an
den relevanten Stellen zitiert und auf Mario übertragen.

---

## 1. Projektziel

- Verstehen, wie ein neuronales Netz mathematisch funktioniert
  (gewichtete Summe, Bias, Aktivierung).
- Verstehen, wie Neuroevolution funktioniert (Fitness, Selektion,
  Mutation, Elitismus) und wie sie sich vom klassischen überwachten
  Training unterscheidet.
- All das sichtbar machen: echte Spielszenen aller Instanzen, das
  Netz des ausgewählten Individuums mit echten Gewichten und
  Aktivierungen, Fitnessstatistiken über Generationen.

## 2. Architekturübersicht

```
                ┌────────────────────────────────────────────────┐
                │                    GUI (Tkinter)               │
                │   Raster aller Instanzen │ Netz │ Statistiken │
                └────────────┬───────────────────────┬───────────┘
                             │ after()-Timer (50 ms) │ nur zeichnen
                ┌────────────▼───────────────────────▼───────────┐
                │              Experiment (experiment.py)        │
                │  Generationen, Statistiken, Rekord, Save/Load  │
                └───────┬────────────────────────────┬───────────┘
                        │ start_round()              │ advance_generation()
             ┌──────────▼──────────┐        ┌─────────▼─────────┐
             │ MultiMarioSimulation│        │ Evolver (evolution)│
             │  (simulation.py)    │        │ Selektion/Mutation │
             │  Lockstep-Schleife  │        └─────────┬─────────┘
             └──┬───────────┬──────┘                  │ Genome
                │           │                         │
     ┌──────────▼──┐   ┌────▼─────────┐       ┌───────▼────────┐
     │  Episode 1  │   │  Episode N   │       │   Population   │
     │ (eigenes    │   │ (eigenes     │       │ (individual.py)│
     │  nes-py Env)│   │  nes-py Env) │       └───────┬────────┘
     └──┬──────┬───┘   └──────────────┘               │
        │      │                              ┌───────▼────────┐
  Beobachtung │                              │ NeuralNetwork  │
  (observation│                              │ (network.py)   │
  .py)        │                              │ reines NumPy   │
        │      │                              └────────────────┘
  ┌─────▼──────▼──┐
  │  actions.py   │  Netzausgabe → Joypad-Byte → env.step()
  └───────────────┘
```

**Datenfluss pro Simulationsschritt (je Instanz):**

```
NES-Frame + info → encode() → Beobachtungsvektor (13 Werte)
     → network.forward() → 7 Aktionswerte
     → choose_action() (argmax) → Aktionsindex
     → gym_action_index() → Joypad-Byte → env.step()
     → neuer Frame + info → Fitness-Zähler (x_pos, blocked, reward)
```

## 3. Mathematische Grundlagen (The Nature of Code → Code)

### Neuron (Kapitel 10)

Ein Neuron berechnet die **gewichtete Summe** seiner Eingaben und legt
darüber eine **Aktivierungsfunktion** an:

```
z = Σᵢ (wᵢ · xᵢ) + b        (gewichtete Summe, "weighted sum")
a = f(z)                    (Aktivierungsfunktion)
```

- `xᵢ`: Eingabe (Sensorwert, z. B. "Hindernis rechts?")
- `wᵢ`: **Gewicht** – bestimmt, wie viel diese Eingabe zählt.
  Positives Gewicht = verstärkt, negatives = hemmt, ~0 = egal.
- `b`: **Bias** – verschiebt die Entscheidungsschwelle. Ohne Bias
  müsste jedes Neuron bei z=0 aktivieren; mit Bias ist die Schwelle
  frei lernbar (im Material als "immer-1-Eingabe" beschrieben).
- `f`: **Aktivierungsfunktion** – macht das Netz nichtlinear. Ohne sie
  wäre ein mehrschichtiges Netz mathematisch nur eine Schicht.

Implementierung (`network.py`):

- Versteckte Schichten: `f = ReLU` mit `f(z) = max(0, z)`.
- Ausgabeschicht: `f = Identität` (rohe Aktionswerte, `argmax`
  entscheidet – keine Normalisierung nötig).

### Mehrschichtiges Netz

Die Ausgaben einer Schicht sind die Eingaben der nächsten. In
Matrixform (ein NumPy-Schritt pro Schicht):

```
a_next = f(W · a + b)
```

`W` hat die Form (Neuronen der Zielschicht × Neuronen der Quellschicht).
Das Projekt nutzt standardmäßig `13 → 12 → 7` (13 Sensorwerte, 12
versteckte Neuronen, 7 Aktionen).

### Neuroevolution statt Backpropagation

|                    | Überwachtes Training (Backpropagation)        | Neuroevolution (dieses Projekt)                  |
| ------------------ | --------------------------------------------- | ------------------------------------------------ |
| Lernsignal         | Fehler zum Zielwert (Gradient)                | **Fitness** (ein Skalar pro Individuum)          |
| Parameteränderung  | Gradientenabstieg, alle Gewichte gleichzeitig | Selektion guter Genomes + **zufällige Mutation** |
| Braucht Zielwerte? | Ja (Label)                                    | Nein – nur eine Bewertung der Episode            |
| Reproduktion       | ein Netz                                      | Population, Generationen, Elitismus              |

In `evolution.py`:

1. **Ranking** nach Fitness.
2. **Elitismus**: die besten `elite_count` Genomes werden _bitidentisch_
   übernommen (dafür gibt es einen Test!).
3. **Roulette-Wheel-Selektion**: Eltern werden mit Wahrscheinlichkeit
   proportional zur Fitness gewählt.
4. **Mutation**: jedes Gen (Gewicht oder Bias) wird mit Wahrscheinlichkeit
   `mutation_rate` um eine Gauß-Störung `N(0, mutation_strength²)`
   verschoben.

## 4. Beobachtung und Aktion (die Schnittstelle)

### Eingaben (13 Werte, alle in [0,1]) – `observation.py`

| Index | Name             | Bedeutung                                            |
| ----- | ---------------- | ---------------------------------------------------- |
| 0     | reststrecke_x    | `(4800 - x_pos) / 4800` – wie weit bis zum Levelende |
| 1     | x_position       | `x_pos / 4800` – absolute Position                   |
| 2     | höhe_über_boden  | `(193 - y_pixel) / 180` – Sprunghöhe                 |
| 3     | fällt_gerade     | 1, wenn `y_pixel` gegenüber dem Vor-Schritt wächst   |
| 4     | restzeit         | `time / 400`                                         |
| 5     | ist_tot          | 1, wenn Mario tot ist                                |
| 6     | flagge_erreicht  | 1, wenn das Level geschafft ist                      |
| 7     | hindernis_links  | solides Hindernis/Gegner links neben Mario           |
| 8     | hindernis_rechts | solides Hindernis/Gegner rechts neben Mario          |
| 9–12  | letzte Aktion    | One-Hot: rechts / rechts_sprung / springen / links   |

**Pixel-Sensorik ohne Bildverarbeitungsbibliothek:** Mario wird im
Bild über seine rote Kleidung `(248,56,0)` gefunden (der Viewport
scrollt mit!). In einem 32-Pixel-Streifen neben ihm wird geprüft:

- **Hindernis (Rohr/Block)**: der nicht-Himmel/nicht-Boden-Anteil
  überschreitet 0.20 (Busch ~0.07, Rohr ~0.30 – kalibriert an echten
  Frames).
- **Gegner**: das Goomba-Fell `(240,208,176)` kommt im Streifen vor
  (diese Farbe gibt es sonst nirgends in der Szene).

### Ausgaben (7 Aktionen) – `actions.py`

Der Index im Netzausgabevektor **ist** die Aktion (Kopplung an
`SIMPLE_MOVEMENT` des Pakets):

| Index | Aktion               | NES-Tasten    |
| ----- | -------------------- | ------------- |
| 0     | halten               | NOOP          |
| 1     | rechts               | right         |
| 2     | rechts_sprung        | right + A     |
| 3     | rennen_rechts        | right + B     |
| 4     | rennen_sprung_rechts | right + A + B |
| 5     | springen             | A             |
| 6     | links                | left          |

Die Auswahlregel: **argmax** – der größte Ausgabewert gewinnt
(deterministisch, Tie-Break kleinster Index).

## 5. Fitness – `fitness.py`

```
fitness = 1.0 · final_x_pos        (Fortschritt in Pixeln, Ziel ~4800)
        + 5000 · (1 wenn Flagge)   (Zielbonus, dominiert alles)
        − 1.0 · blocked_steps      (Strafe für Anlaufen)
        + 0.5 · reward_sum         (sekundäres Spiel-Signal, z.B. Münzen)
```

Damit gilt: Zielerreichung schlägt jeden Fortschritt; Fortschritt
schlägt jede Strafe; effizientere Erfolge schlagen ineffiziente.

## 6. Installation

```powershell
# Empfohlen: virtuelle Umgebung
python -m venv .venv
.venv\Scripts\Activate.ps1

# Abhängigkeiten (NumPy + Emulator)
pip install numpy gym-super-mario-bros nes-py pytest
```

`nes-py` bringt den NES-Emulator als kompiliertes Binary mit; für
Windows/Python 3.12+ existieren fertige Wheels. Auf sehr neuen
Python-Versionen (z. B. 3.14) ggf. das passende nes-py-Wheel
installieren.

## 7. Start

```powershell
# Dashboard
python -m mario_evolution
```

Ablauf im Dashboard:

1. **Start** – startet die Simulation aller Instanzen (Standard: 8
   parallele Emulator-Instanzen im Level 1-1).
2. **Pause / 1 Schritt** – pausieren oder einzeln weiterschalten.
3. Wenn alle Episoden fertig sind (Flagge, Tod oder Schrittbudget),
   **Neue Generation** drücken – Selektion + Mutation erzeugen die
   nächste Population.
4. **Speichern/Laden** – Experiment (alle Genome, Statistiken, Rekord)
   als JSON sichern und später fortsetzen.

Hinweis: 8 Emulator-Instanzen sind eine praktische Obergrenze für
flüssige Darstellung; mehr Instanzen = mehr CPU-Last (jede Instanz
ist ein voller NES-Emulator).

## 8. Die GUI

- **Links**: Raster aller Mario-Instanzen mit _echten Spielszenen_
  (NES-Frames). Rahmenfarbe = Status: gelb = aktiv, grün = Flagge
  erreicht, rot = beendet. Label zeigt Individuum + x-Position.
- **Rechts oben (Bester Lauf)**: Detailansicht des aktuell besten
  Laufs (dieser Generation) plus der Rekord über alle Generationen –
  beides klar getrennt angezeigt.
- **Rechts Mitte (Neuronales Netz)**: die _tatsächliche_ Architektur
  des ausgewählten Individuums. Verbindungen blau = positives Gewicht,
  rot = negatives; Linienbreite ~ Gewichtsstärke. Eingabeneuronen
  leuchten mit der aktuellen Beobachtung, Ausgabeneuronen mit den
  aktuellen Aktionswerten. Bei > 400 Verbindungen erscheint eine
  Reduktions-Meldung (bewusste, gekennzeichnete Kürzung).
- **Rechts unten (Statistiken)**: bester/durchschnittlicher/schlechtester
  Fitnesswert, Erfolgsquote, durchschnittliche Schrittzahl, blockierte
  Schritte, bestes x; darunter der Verlauf von bester/Ø Fitness über
  die Generationen.
- **Auswahl**: Combobox "Individuum" – wählt, dessen Netz gezeigt wird.

## 9. Konfiguration

Direkt in der Kopfzeile:

- **Populationsgröße** (2–40) und **versteckte Neuronen** (1–64).
  Änderungen, die eine neue Population erfordern, müssen bestätigt
  werden – vorhandene Genome werden nie stillschweigend überschrieben.

In `visualizer.py` / `experiment.py` (Konstanten):

- `max_steps` (Schrittbudget pro Episode, Standard 1000),
- `EvolutionConfig(elite_count, mutation_rate, mutation_strength)`,
- Seed (Standard `42`) – gleiches Experiment mit gleichem Seed liefert
  dieselben Genome (reproduzierbar).

## 10. Tests

```powershell
python -m pytest tests/ -q
```

Abgedeckt (67 Tests):

- handnachprüfbare Neuronenberechnung (`z = w·x + b`, ReLU-Klemmung),
- ReLU bei negativen/0/positiven Werten,
- Schicht-/Matrixdimensionen, Genomform, `genome_size`,
- Vorwärtsberechnung eines kleinen Netzes mit fest vorgegebenen
  Gewichten,
- reproduzierbare Initialisierung (Seed) und Genom-Roundtrip,
- Mutation (rate=0 → bitidentisch, Input bleibt unverändert),
- unveränderte Eliten (bitidentisch),
- Roulette-Selektion (bestes Individuum überproportional oft),
- Aktionsauswahl (argmax, Tie-Break) und Joypad-Bitmasken,
- Beobachtung: Größe/Wertebereich, Fortschritt normalisiert,
  Hindernis-/Gegner-Erkennung, One-Hot, Validierung fehlerhafter
  Eingaben,
- Fitness: Fortschritt, Zielbonus-Dominanz, effizienter vs.
  ineffizienter Erfolg, blockierte Schritte, ungültige Gewichte,
- Statistiken über eine Generation (alle Werte von Hand prüfbar),
- Rekord (nur strikt besser ersetzt), Save/Load-Roundtrip,
- **Integrations-Test mit echtem Emulator**: unabhängige Instanzen,
  beendete Episoden bleiben eingefroren, blockierte Schritte gezählt,
  Fitness exakt aus eigenem Zustand, reproduzierbares Verhalten bei
  festem Genom.

## 11. Bekannte Grenzen

- **Nes-py ist ein Emulator**: ~8 gleichzeitige Instanzen sind mit
  Tkinter noch flüssig; mehr Instanzen kosten proportional CPU.
- **Kein Crossover**: Selektion + Mutation reichen für die erste
  Version; Crossover ist bewusst _nicht_ implementiert (das Genom ist
  dafür strukturiert, aber nicht geprüft).
- **Sensorik ist Heuristik**: Die Farbsensorik funktioniert für
  Level 1-1; Rohre/Hintergrundbüsche teilen sich eine Palette und
  werden über den Füllanteil unterschieden. Andere Level (Nacht,
  Unterwasser) brauchen angepasste Farbwerte.
- **Fitness favorisiert Nach-Rechts**: `x_pos` ist das natürliche
  Mario-Maß; vertikale Geheimwege (über die Oberkante) werden nicht
  belohnt.
- **Keine Rekurrenten Verbindungen**: Das Netz ist rein feedforward;
  "Gedächtnis" kommt nur über die One-Hot-Eingabe der letzten Aktion.
- **GUI-Performance**: Bei sehr großen Populationen (>20) das Raster
  kleiner skalieren; die Netzansicht reduziert Verbindungen ab 400.
- **Reproduzierbarkeit**: Genome/Mutationen sind per Seed reproduzier-
  bar; die Emulator-Umgebung selbst ist deterministisch pro Episode,
  aber zwischen Maschinen können Emulator-Detailverhalten abweichen.
