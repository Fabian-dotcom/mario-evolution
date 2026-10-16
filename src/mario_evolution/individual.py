"""Individuum: ein Mario-Agent mit eigenem Netzwerk-Genom und Fitness.

Ein Individuum BESITZT:
- seine Netzwerkarchitektur (layer_sizes) und seine Parameter (Genom),
- seine eigene Fitness (nach der Bewertung),
- einen Namen/Label für die GUI.

Ein Individuum ist eine reine Daten-/Logikeinheit OHNE Spielzustand -
der Spielzustand (Position, Schritte, ...) lebt in episode.py. Diese
Trennung macht es möglich, dieselben Genome in vielen unabhängigen
Mario-Instanzen gleichzeitig laufen zu lassen.
"""

from __future__ import annotations

import numpy as np

from .network import NeuralNetwork
from .observation import OBSERVATION_SIZE
from .actions import action_count


class Individual:
    """Ein Agent der Population.

    Attribute:
        network: NeuralNetwork dieses Individuums (Besitz).
        fitness: Fitnesswert (None = noch nicht bewertet).
        label: Anzeigename für die GUI, z.B. "Ind_07".
    """

    def __init__(self,
                 network: NeuralNetwork,
                 label: str = "Individuum",
                 fitness: float | None = None):
        if not isinstance(network, NeuralNetwork):
            raise TypeError(f"Individual: network muss NeuralNetwork sein, erhielt {type(network).__name__}")
        if not isinstance(label, str) or not label:
            raise ValueError("Individual: label muss eine nicht-leere Zeichenkette sein")
        if fitness is not None and not np.isfinite(fitness):
            raise ValueError(f"Individual: fitness muss endlich sein, erhielt {fitness!r}")
        self.network = network
        self.fitness: float | None = fitness
        self.label = label

    # ------------------------------------------------------------------
    # Erzeugung
    # ------------------------------------------------------------------
    @classmethod
    def random(cls,
               hidden_sizes: list[int],
               rng: np.random.Generator,
               label: str = "Individuum") -> "Individual":
        """Erzeugt ein Individuum mit zufällig initialisiertem Netz.

        Argumente:
            hidden_sizes: Neuronenzahlen der versteckten Schichten, z.B.
                [8] für eine Schicht mit 8 Neuronen.
            rng: NumPy-Generator (mit Seed für Reproduzierbarkeit).
            label: Anzeigename.

        Die EIN-/Ausgabegröße ist fest verdrahtet mit der Beobachtungs-
        und Aktionsschnittstelle: Eingabe = OBSERVATION_SIZE, Ausgabe =
        Anzahl der Aktionen. So kann es keine Architektur geben, die
        zur Spielwelt nicht passt.
        """
        if not isinstance(hidden_sizes, list) or not hidden_sizes:
            raise ValueError(
                "Individual.random: hidden_sizes muss eine nicht-leere Liste sein"
            )
        for h in hidden_sizes:
            if not isinstance(h, int) or h < 1:
                raise ValueError(
                    f"Individual.random: versteckte Schichtgrößen müssen int >= 1 sein, erhielt {h!r}"
                )
        layer_sizes = [OBSERVATION_SIZE, *hidden_sizes, action_count()]
        network = NeuralNetwork(layer_sizes=layer_sizes, rng=rng)
        return cls(network=network, label=label)

    # ------------------------------------------------------------------
    # Genom-Zugriff
    # ------------------------------------------------------------------
    def genome(self) -> np.ndarray:
        """Genom (flacher Parametervektor) dieses Individuums."""
        return self.network.genome()

    def clone_with_genome(self, genome: np.ndarray, label: str) -> "Individual":
        """Erzeugt ein neues Individuum mit gleichem Layout und neuem Genom."""
        new_net = self.network.from_genome(genome)
        return Individual(network=new_net, label=label)

    # ------------------------------------------------------------------
    # Aktion
    # ------------------------------------------------------------------
    def act(self, observation: np.ndarray) -> int:
        """Beobachtung -> Aktionsindex für gymnasium env.step(action).

        Trennung Netz <-> Entscheidung: forward() liefert Aktionswerte,
        choose_action() wählt via argmax. Rückgabe ist der INDEX der
        Aktion (gymnasium-Schnittstelle nutzt Integer-Aktionen).
        """
        from .actions import choose_action
        outputs = self.network.forward(observation)
        return choose_action(outputs)
