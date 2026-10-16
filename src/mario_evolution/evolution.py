"""Evolution: Selektion, Elitismus, Mutation und neue Generationen.

Das ist der Kern der Neuroevolution (Bezug "The Nature of Code",
Kapitel Neuroevolution / genetische Algorithmen):

1. Alle Individuen werden bewertet (Fitness, siehe fitness.py).
2. Die besten werden als ELITEN unverändert in die nächste Generation
   übernommen - ihr Genom bleibt bitidentisch.
3. Der Rest wird aus Eltern erzeugt: Eltern werden mit Wahrscheinlichkeit
   proportional zur Fitness gewählt ("roulette wheel selection",
   im Material als "selection" beschrieben), dann MUTIERT.
4. Mutation ändert zufällig einzelne Gen-Werte (Gewichte/Bias) leicht.
   Das ist der Motor der Variation - ohne ihn gäbe es keine Verbesserung.

NICHT enthalten: Backpropagation / Gradienten. Die Gewichte werden
nicht "gelernt", sondern von der Selektion "veredelt" - das ist der
zentrale Unterschied zwischen Neuroevolution und klassischem
überwachtem Training (siehe README).

Crossover: bewusst NICHT implementiert (Auftrag: "nicht ungeprüft").
Ohne Crossover funktioniert die Evolution vollständig; das Genom ist
als flacher Vektor strukturiert, so dass ein späterer Zuschnitt
möglich ist, solange beide Individuen dieselbe Architektur haben.
"""

from __future__ import annotations

import numpy as np

from .individual import Individual
from .population import Population


class EvolutionConfig:
    """Konfiguration des evolutionären Algorithmus.

    Attribute:
        elite_count: Anzahl unverändert übernommener Bestindividuen
            (>= 0). Elitismus verhindert, dass der beste Fund einer
            Generation durch Mutation wieder verloren geht.
        mutation_rate: Wahrscheinlichkeit pro Gen, mutiert zu werden
            in [0, 1].
        mutation_strength: Standardabweichung der Gauß-Störung pro
            Mutation (>= 0; 0 bedeutet "keine effektive Änderung").
    """

    def __init__(self,
                 elite_count: int = 2,
                 mutation_rate: float = 0.1,
                 mutation_strength: float = 0.3):
        if not isinstance(elite_count, int) or isinstance(elite_count, bool) or elite_count < 0:
            raise ValueError(f"EvolutionConfig: elite_count muss int >= 0 sein, erhielt {elite_count!r}")
        for name in ("mutation_rate", "mutation_strength"):
            value = getattr(self, name, None)  # noqa - wird gleich gesetzt
        # (Werte erst setzen, dann validieren - siehe Zuweisung unten.)

        if not isinstance(mutation_rate, (int, float)) or isinstance(mutation_rate, bool) \
                or not (0.0 <= float(mutation_rate) <= 1.0):
            raise ValueError(
                f"EvolutionConfig: mutation_rate muss in [0, 1] liegen, erhielt {mutation_rate!r}"
            )
        if not isinstance(mutation_strength, (int, float)) or isinstance(mutation_strength, bool) \
                or float(mutation_strength) < 0 or not np.isfinite(mutation_strength):
            raise ValueError(
                f"EvolutionConfig: mutation_strength muss endliche Zahl >= 0 sein, "
                f"erhielt {mutation_strength!r}"
            )
        self.elite_count = elite_count
        self.mutation_rate = float(mutation_rate)
        self.mutation_strength = float(mutation_strength)


def rank_individuals(individuals: list[Individual]) -> list[Individual]:
    """Sortiert Individuen nach Fitness, bester zuerst.

    Individuen ohne Fitness (None) werden ans Ende sortiert. Die
    Reihenfolge der Eingabeliste wird nicht verändert (neue Liste).
    """
    def sort_key(ind: Individual):
        # None -> -unendlich: unbewertete Individuen verlieren immer.
        return -(ind.fitness if ind.fitness is not None else float("-inf"))
    return sorted(individuals, key=sort_key)


def select_parent(sorted_individuals: list[Individual], rng: np.random.Generator) -> Individual:
    """Roulette-Wheel-Selektion eines Elternteils.

    Mathematisch: Jedes Individuum i wird mit Wahrscheinlichkeit
        p_i = f_i / sum(f)
    gewählt (Fitness f_i >= 0 vorausgesetzt). Negative Fitness-Werte
    werden auf 0 geklemmt ("blinde" frühe Individuen bekommen damit
    nur die Basiswahrscheinlichkeit).

    Warum Roulette statt "nimm immer die Besten"? So bleiben auch
    schwächere Individuen eine Chance - ihre (seltenen) Gen-Varianten
    könnten später wertvoll werden. Das ist im Material so beschrieben
    ("fitness proportional selection").
    """
    if not sorted_individuals:
        raise ValueError("select_parent: leere Individuenliste")
    fitness = np.asarray(
        [max(0.0, ind.fitness) if ind.fitness is not None else 0.0
         for ind in sorted_individuals],
        dtype=np.float64,
    )
    total = fitness.sum()
    if total <= 0.0:
        # Alle gleich schlecht (z.B. Startgeneration): gleichverteilte
        # Wahl - dokumentierte, deterministische Ausweichregel.
        return sorted_individuals[int(rng.integers(len(sorted_individuals)))]
    probabilities = fitness / total
    index = int(rng.choice(len(sorted_individuals), p=probabilities))
    return sorted_individuals[index]


def mutate_genome(genome: np.ndarray,
                  config: EvolutionConfig,
                  rng: np.random.Generator) -> np.ndarray:
    """Mutiert ein Genom (flacher Parametervektor).

    Mathematisch: Für jedes Gen g wird mit Wahrscheinlichkeit
    mutation_rate eine Gauß-Störung angewendet:
        g_neu = g + N(0, mutation_strength^2)
    Unmutierte Gene bleiben BITIDENTISCH - so bleibt gutes Wissen
    erhalten, und nur wenige Parameter ändern sich pro Kind.

    Rückgabe: NEUES Array (Eingabe wird nicht verändert).
    """
    genome = np.asarray(genome, dtype=np.float64)
    if genome.ndim != 1:
        raise ValueError(f"mutate_genome: 1D-Genom erwartet, erhielt shape {genome.shape}")
    if not np.all(np.isfinite(genome)):
        raise ValueError("mutate_genome: Genom enthält NaN oder unendliche Werte")
    mask = rng.random(genome.shape[0]) < config.mutation_rate
    mutated = genome.copy()
    if mask.any():
        noise = rng.normal(loc=0.0, scale=config.mutation_strength, size=int(mask.sum()))
        mutated[mask] += noise
    return mutated


class Evolver:
    """Erzeugt neue Generationen aus einer bewerteten Population."""

    def __init__(self, config: EvolutionConfig, rng: np.random.Generator):
        if not isinstance(config, EvolutionConfig):
            raise TypeError(f"Evolver: config muss EvolutionConfig sein, erhielt {type(config).__name__}")
        if not isinstance(rng, np.random.Generator):
            raise TypeError(f"Evolver: rng muss np.random.Generator sein, erhielt {type(rng).__name__}")
        self.config = config
        self.rng = rng

    def next_generation(self, current: Population, fitness_by_label: dict[str, float]) -> Population:
        """Erzeugt die nächste Generation.

        Argumente:
            current: aktuelle Population.
            fitness_by_label: Fitness pro Label (aus dem Experiment).

        Ablauf:
        1. Individuen erhalten ihre Fitness, Rangliste wird gebildet.
        2. Eliten: die config.elite_count Besten werden UNVERÄNDERT
           übernommen (Genom bitidentisch - dafür gibt es einen Test).
        3. Für jedes weitere Kind: Eltern per Roulette wählen, Genom
           mutieren, neues Netz mit gleichem Layout erzeugen.
        """
        if len(fitness_by_label) != current.size:
            raise ValueError(
                f"next_generation: Fitness-Liste hat {len(fitness_by_label)} Einträge, "
                f"erwartet {current.size}"
            )
        for ind in current.individuals:
            if ind.label not in fitness_by_label:
                raise ValueError(f"next_generation: keine Fitness für '{ind.label}'")
            ind.fitness = float(fitness_by_label[ind.label])

        ranked = rank_individuals(current.individuals)
        elite_count = min(self.config.elite_count, current.size)
        elite = ranked[:elite_count]

        # Eliten behalten ihre Genomes exakt (bitidentisch).
        new_individuals: list[Individual] = [
            ind.clone_with_genome(ind.genome(), label=f"{ind.label}_elite")
            for ind in elite
        ]
        template = current.individuals[0]

        while len(new_individuals) < current.size:
            parent = select_parent(ranked, self.rng)
            child_genome = mutate_genome(parent.genome(), self.config, self.rng)
            child = template.clone_with_genome(
                child_genome, label=f"Gen_child_{len(new_individuals):03d}"
            )
            new_individuals.append(child)

        return Population(individuals=new_individuals, hidden_sizes=current.hidden_sizes)
