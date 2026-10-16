"""Feedforward-Neuronales Netz - Implementierung nach "The Nature of Code".

Dieses Modul implementiert ein vollständig verbundenes (fully connected)
Feedforward-Netzwerk REIN mit NumPy - ohne PyTorch, TensorFlow o.ä. Die
Mathematik folgt direkt dem Material:

Perzeptron / Neuron (Kapitel 10, "The Nature of Code"):

    z = sum_i (w_i * x_i) + b       (gewichtete Summe, "weighted sum")
    a = f(z)                        (Aktivierungsfunktion, z.B. ReLU)

Dabei ist:
    x_i : Eingabe i (z.B. Sensorwert aus observation.py)
    w_i : Gewicht i - bestimmt, WIE WICHTIG die Eingabe für das Neuron ist.
          Großes positives Gewicht = Eingabe verstärkt das Neuron,
          negatives Gewicht = Eingabe hemmt es, Gewicht ~ 0 = egal.
    b   : Bias - ein zusätzlicher, von Eingaben unabhängiger Wert.
          Er verschiebt die Entscheidungsschwelle: Ohne Bias müsste das
          Neuron bei z=0 aktivieren; mit Bias kann die Schwelle frei
          gewählt werden. (Kapitel 10: "eine zusätzliche Eingabe, die
          immer 1 ist" - hier als expliziter b-Wert implementiert.)
    f   : Aktivierungsfunktion - sie macht das Netz nichtlinear. Ohne
          sie wäre eine Kette von Schichten mathematisch nur EINE
          Schicht, weil sich lineare Funktionen ineinander aufösen.

Mehrere Schichten (Kapitel: "Multi-layered feedforward networks"):
    Die Ausgaben einer Schicht sind die Eingaben der nächsten. In
    Matrixform pro Schicht:
        a_next = f(W @ a + b)
    wobei W die Form (Anzahl Neuronen nächster Schicht, Anzahl Eingaben)
    hat.

Was hier NICHT passiert: Backpropagation / Gradientenabstieg. Die
Gewichte werden NICHT durch Ableiten gelernt, sondern von der Evolution
gesetzt und verändert (siehe evolution.py). Genau das ist Neuroevolution.
"""

from __future__ import annotations

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU-Aktivierungsfunktion: f(z) = max(0, z).

    Mathematisch: Negative Werte werden auf 0 gesetzt, positive bleiben
    unverändert. Warum ReLU für die versteckten Schichten?
    - nichtlinear (notwendig, damit mehrschichtige Netze mehr können
      als einziges Perzeptron),
    - numerisch billig und robust,
    - im Material als übliche, gut verständliche Aktivierung genannt.

    Fehler:
        ValueError bei NaN/Inf-Eingaben - keine stillen Korrekturen.
    """
    if not np.all(np.isfinite(z)):
        raise ValueError(f"relu: Eingabe muss endlich sein (kein NaN/Inf), erhielt {z}")
    return np.maximum(0.0, z)


def identity(z: np.ndarray) -> np.ndarray:
    """Identitätsaktivierung f(z) = z für die Ausgabeschicht.

    Die Ausgabe soll rohe "Aktionswerte" liefern; die Auswahl der
    Aktion (argmax in actions.py) funktioniert mit beliebigen reellen
    Werten. Daher braucht die Ausgabeschicht keine nichtlineare
    Aktivierung und keine Normalisierung.
    """
    if not np.all(np.isfinite(z)):
        raise ValueError(f"identity: Eingabe muss endlich sein (kein NaN/Inf), erhielt {z}")
    return z


class NeuralNetwork:
    """Vollständig verbundenes Feedforward-Netz mit beliebiger Schichtzahl.

    Attribute:
        layer_sizes: Liste der Neuronenzahlen, inkl. Eingabe- und
            Ausgabeschicht. Beispiel [16, 8, 4] = 16 Eingaben, 8 Neuronen
            in einer versteckten Schicht, 4 Aktionswerte als Ausgabe.
        weights: Liste von Matrizen, weights[k] verbindet Schicht k mit
            Schicht k+1; Form (layer_sizes[k+1], layer_sizes[k]).
        biases: Liste von Vektoren, biases[k] gehört zu Schicht k+1;
            Form (layer_sizes[k+1],).
        hidden_activation: Aktivierungsfunktion der versteckten Schichten.
        output_activation: Aktivierungsfunktion der Ausgabeschicht.

    Die Eingabeschicht selbst hat KEINE Gewichte - sie ist nur der
    Vektor der Eingabewerte (wie im Material beschrieben: Eingaben
    fließen direkt in das erste "echte" Neuronen-Layout).
    """

    def __init__(self,
                 layer_sizes: list[int],
                 hidden_activation=relu,
                 output_activation=identity,
                 weights: list[np.ndarray] | None = None,
                 biases: list[np.ndarray] | None = None,
                 rng: np.random.Generator | None = None):
        """Erzeugt das Netz.

        Argumente:
            layer_sizes: z.B. [16, 8, 4]. Mindestens 3 Einträge (Eingabe,
                versteckte Schicht, Ausgabe); alle Werte >= 1.
            hidden_activation / output_activation: Aktivierungsfunktionen.
            weights / biases: OPTIONAL - vorhandene Parameter (z.B. aus
                einem Genom). Wenn nicht übergeben, wird zufällig
                initialisiert (siehe random_parameters).

        Fehler:
            ValueError/TypeError bei ungültigen Schichtgrößen,
            passungslosen Parameterformen oder nicht-finiten Werten.

        Reproduzierbarkeit: Über `rng` kann ein NumPy-Generator mit
        festem Seed übergeben werden; ohne rng wird ein frischer
        Generator erzeugt (nicht reproduzierbar).
        """
        if not isinstance(layer_sizes, (list, tuple)) or len(layer_sizes) < 3:
            raise ValueError(
                "NeuralNetwork: layer_sizes braucht mindestens 3 Einträge "
                "(Eingabe, versteckte Schicht, Ausgabe)"
            )
        for size in layer_sizes:
            if not isinstance(size, int) or isinstance(size, bool):
                raise TypeError(
                    f"NeuralNetwork: Schichtgrößen müssen int sein, erhielt {size!r}"
                )
            if size < 1:
                raise ValueError(
                    f"NeuralNetwork: jede Schicht braucht mindestens 1 Neuron, erhielt {size}"
                )
        self.layer_sizes: list[int] = list(layer_sizes)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        # Zufallsgenerator für die Initialisierung (reproduzierbar,
        # wenn ein Seed-Generator übergeben wird).
        self._rng = rng if rng is not None else np.random.default_rng()

        expected_shapes = self.parameter_shapes()

        if weights is None and biases is None:
            # Zufällige, kleine Startwerte (Xavier-ähnlich): große Werte
            # würden die Aktivierungen sofort "sprengen"; kleine Werte
            # geben der Evolution die Chance, Verbindungen schrittweise
            # zu stärken oder zu schwächen.
            self.weights, self.biases = self.random_parameters()
            return

        if weights is None or biases is None:
            raise ValueError(
                "NeuralNetwork: weights und biases müssen gemeinsam übergeben werden"
            )
        # Längenprüfung VOR der Formprüfung (aussagekräftige Reihenfolge).
        if len(weights) != len(expected_shapes) or len(biases) != len(expected_shapes):
            raise ValueError(
                f"NeuralNetwork: erwartet {len(expected_shapes)} Gewichtsmatrizen, "
                f"erhielt {len(weights)} / {len(biases)} Bias-Vektoren"
            )
        # Exakte Form- und Wertprüfung der übergebenen Parameter.
        self.weights = []
        self.biases = []
        for k, (w, (rows, cols)) in enumerate(zip(weights, expected_shapes)):
            w = np.asarray(w, dtype=np.float64)
            if w.shape != (rows, cols):
                raise ValueError(
                    f"NeuralNetwork: weights[{k}] hat Form {w.shape}, erwartet {(rows, cols)}"
                )
            self.weights.append(w)
        for k, (b, (rows, _cols)) in enumerate(zip(biases, expected_shapes)):
            b = np.asarray(b, dtype=np.float64)
            if b.shape != (rows,):
                raise ValueError(
                    f"NeuralNetwork: biases[{k}] hat Form {b.shape}, erwartet {(rows,)}"
                )
            self.biases.append(b)
        self._validate_finite()

    # ------------------------------------------------------------------
    # Parameterstruktur
    # ------------------------------------------------------------------
    def parameter_shapes(self) -> list[tuple[int, int]]:
        """Formen aller Gewichtsmatrizen [(rows, cols), ...].

        cols = Neuronenzahl der Quellschicht, rows = der Zielschicht.
        biases[k] hat Form (rows,). Diese Struktur ist die verbindliche
        "Gestalt" des Genoms (siehe genome() / from_genome()).
        """
        return [
            (self.layer_sizes[k + 1], self.layer_sizes[k])
            for k in range(len(self.layer_sizes) - 1)
        ]

    def random_parameters(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Reproduzierbare zufällige Initialisierung.

        Verwendet den Generator self._rng, der im Konstruktor gesetzt
        wurde. Die Standardnormalverteilung (mean 0, std 0.5) liefert
        kleine, Vorzeichen-symmetrische Startwerte: Das Netz startet
        mit völlig untrainiertem, aber nicht degeneriertem Verhalten.
        """
        rng = self._rng
        weights, biases = [], []
        for rows, cols in self.parameter_shapes():
            # Standardnormalverteilt, mit std 0.5 klein gehalten.
            weights.append(rng.normal(loc=0.0, scale=0.5, size=(rows, cols)))
            biases.append(rng.normal(loc=0.0, scale=0.5, size=rows))
        return weights, biases

    def _validate_finite(self) -> None:
        """Prüft, dass alle Parameter endlich sind (kein NaN/Inf)."""
        for k, (w, b) in enumerate(zip(self.weights, self.biases)):
            if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
                raise ValueError(
                    f"NeuralNetwork: Parameter der Schicht {k} enthalten "
                    f"NaN oder unendliche Werte"
                )

    # ------------------------------------------------------------------
    # Genom-Schnittstelle (für Evolution, Speichern, Laden)
    # ------------------------------------------------------------------
    def genome(self) -> np.ndarray:
        """Alle Parameter als flacher 1D-Vektor (das "Genom").

        Reihenfolge: alle Gewichtsmatrizen hintereinander (Schicht für
        Schicht, zeilenweise), dann alle Bias-Vektoren. Diese flache Form
        macht Mutation trivial (einzelne Gen-Werte ändern) und
        Speichern/Laden eindeutig. Die Form-Information steckt in
        layer_sizes, die Mutation/Load über parameter_shapes() wieder
       herstellt.
        """
        parts = [w.ravel() for w in self.weights] + [b.ravel() for b in self.biases]
        return np.concatenate(parts)

    def from_genome(self, genome: np.ndarray) -> "NeuralNetwork":
        """Erzeugt ein NEUES Netz mit den Parametern aus einem Genom.

        Das Genom muss exakt die erwartete Länge und nur finite Werte
        haben - jede Abweichung führt zu einer aussagekräftigen
        Fehlermeldung (Genomstruktur wird nie stillschweigend verändert).
        """
        genome = np.asarray(genome, dtype=np.float64)
        expected = self.genome_size()
        if genome.shape != (expected,):
            raise ValueError(
                f"from_genome: Genom hat {genome.shape[0]} Werte, "
                f"erwartet {expected} (Netzarchitektur {self.layer_sizes})"
            )
        if not np.all(np.isfinite(genome)):
            raise ValueError("from_genome: Genom enthält NaN oder unendliche Werte")
        weights, biases = [], []
        offset = 0
        for rows, cols in self.parameter_shapes():
            n_w = rows * cols
            weights.append(genome[offset:offset + n_w].reshape(rows, cols))
            offset += n_w
        for rows, _cols in self.parameter_shapes():
            biases.append(genome[offset:offset + rows].copy())
            offset += rows
        return NeuralNetwork(
            layer_sizes=self.layer_sizes,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
            weights=weights,
            biases=biases,
        )

    def genome_size(self) -> int:
        """Anzahl aller Parameter (Gewichte + Bias) = Genomlänge."""
        rows_cols = self.parameter_shapes()
        n_weights = sum(r * c for r, c in rows_cols)
        n_biases = sum(r for r, _c in rows_cols)
        return n_weights + n_biases

    # ------------------------------------------------------------------
    # Vorwärtsberechnung
    # ------------------------------------------------------------------
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Vorwärtsberechnung: Eingabevektor -> Ausgabevektor.

        Mathematisch pro Schicht k:
            z = W_k @ a + b_k      (gewichtete Summe je Neuron)
            a = f(z)               (Aktivierungsfunktion)

        Argument:
            inputs: 1D-Vektor mit genau layer_sizes[0] finiten Werten.

        Rückgabe:
            1D-Vektor mit layer_sizes[-1] Werten (Aktionswerte).

        Fehler:
            ValueError bei falscher Länge, falschem Typ oder
            nicht-finiten Eingaben.
        """
        a = np.asarray(inputs, dtype=np.float64)
        if a.ndim != 1:
            raise ValueError(f"forward: 1D-Eingabe erwartet, erhielt shape {a.shape}")
        if a.shape[0] != self.layer_sizes[0]:
            raise ValueError(
                f"forward: erwartet {self.layer_sizes[0]} Eingabewerte, "
                f"erhielt {a.shape[0]} (Netzarchitektur {self.layer_sizes})"
            )
        if not np.all(np.isfinite(a)):
            raise ValueError(f"forward: Eingaben müssen endlich sein, erhielt {a}")

        n_layers = len(self.layer_sizes)
        for k in range(n_layers - 1):
            # Gewichte: W@a bildet die gewichteten Summen aller Neuronen
            # der nächsten Schicht in einem Schritt (Matrix-Vektor-
            # Produkt, entspricht sum_i w_i * x_i je Neuron).
            z = self.weights[k] @ a + self.biases[k]
            if k == n_layers - 2:
                # letzte Schicht -> Ausgabeaktivierung
                a = self.output_activation(z)
            else:
                a = self.hidden_activation(z)
        return a

    # ------------------------------------------------------------------
    # Darstellung für das Dashboard
    # ------------------------------------------------------------------
    def describe(self) -> str:
        """Lesbare Architekturbeschreibung, z.B. '16-8-4'."""
        return "-".join(str(s) for s in self.layer_sizes)
