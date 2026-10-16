"""Population: Erzeugung und Verwaltung einer Menge von Individuen.

Die Population ist eine geordnete Liste von Individuen mit derselben
Netzwerkarchitektur. Sie kapselt die zufällige Erzeugung (mit Seed)
und prüft, dass alle Individuen kompatible Genomformen haben -
Voraussetzung für Mutation und Reproduzierbarkeit.
"""

from __future__ import annotations

import numpy as np

from .individual import Individual
from .observation import OBSERVATION_SIZE
from .actions import ACTION_COUNT


class Population:
    """Eine Generation von Individuen.

    Attribute:
        individuals: Liste der Individuen (Reihenfolge = Reihenfolge
            der Erzeugung; Rang kommt erst durch die Bewertung).
        hidden_sizes: Neuronenzahlen der versteckten Schichten (identisch
            für alle Individuen dieser Population).
    """

    def __init__(self, individuals: list[Individual], hidden_sizes: list[int]):
        if not isinstance(individuals, list) or not individuals:
            raise ValueError("Population: individuals muss eine nicht-leere Liste sein")
        if not isinstance(hidden_sizes, list) or not hidden_sizes:
            raise ValueError("Population: hidden_sizes muss eine nicht-leere Liste sein")
        # Alle Individuen müssen dieselbe Genomlänge haben (sonst wäre
        # Mutation/Crossover nicht definiert) - explizit geprüft statt
        # still anzunehmen.
        sizes = {ind.network.genome_size() for ind in individuals}
        if len(sizes) != 1:
            raise ValueError(
                f"Population: alle Individuen müssen dieselbe Genomlänge haben, "
                f"erhielt {sizes}"
            )
        self.individuals = individuals
        self.hidden_sizes = list(hidden_sizes)

    @classmethod
    def random(cls,
               size: int,
               hidden_sizes: list[int],
               rng: np.random.Generator) -> "Population":
        """Erzeugt eine zufällige Startpopulation (reproduzierbar).

        Argumente:
            size: Anzahl der Individuen (>= 1).
            hidden_sizes: Versteckte Schichten, z.B. [8].
            rng: NumPy-Generator mit Seed.
        """
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(f"Population.random: size muss int >= 1 sein, erhielt {size!r}")
        individuals = [
            Individual.random(hidden_sizes=hidden_sizes, rng=rng, label=f"Ind_{i:03d}")
            for i in range(size)
        ]
        return cls(individuals=individuals, hidden_sizes=hidden_sizes)

    @property
    def size(self) -> int:
        """Anzahl der Individuen."""
        return len(self.individuals)

    @property
    def layer_sizes(self) -> list[int]:
        """Vollständige Netzarchitektur dieser Population."""
        return [OBSERVATION_SIZE, *self.hidden_sizes, ACTION_COUNT]
