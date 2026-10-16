"""Experiment: Generationen-Roundtrip, Statistiken, Rekord, Save/Load.

Ein Experiment kapselt den Ablauf ÜBER Generationen hinweg:

    Simulation (alle Individuen spielen parallel im echten Spiel)
        -> Fitness je Individuum
        -> Statistiken dieser Generation berechnen
        -> Rekord-Individuum aktualisieren
        -> Evolver erzeugt die nächste Generation

Alle Zufallsentscheidungen laufen über EINEN Seed-Generator, so dass
ein Experiment mit demselben Seed exakt dieselben Genome liefert
(reproduzierbare Experimente).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .episode import EpisodeStatus
from .evolution import EvolutionConfig, Evolver
from .population import Population
from .simulation import MultiMarioSimulation

# Standard-Environment (Level 1-1, 4 Frames pro Aktion).
DEFAULT_ENV_NAME = "SuperMarioBros-1-1-v0"


@dataclass
class GenerationStats:
    """Statistiken EINER Generation (Basis: alle Episoden nach Abschluss).

    Attribute:
        generation: Generationenzahl (0-basiert).
        population_size: Anzahl der Individuen.
        best_fitness: höchster Fitnesswert der Generation.
        average_fitness: arithmetisches Mittel aller Fitnesswerte.
        worst_fitness: niedrigster Fitnesswert.
        success_count: Anzahl Individuen, die die Flagge erreichten.
        success_rate: success_count / population_size (in [0,1]).
        average_steps: Mittelwert der Schrittzahlen aller Episoden.
        total_blocked_steps: Summe aller blockierten Schritte.
        best_x_pos: größte erreichte x-Position (Fortschritt in Pixeln).
    """

    generation: int
    population_size: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    success_count: int
    success_rate: float
    average_steps: float
    total_blocked_steps: int
    best_x_pos: float

    def as_dict(self) -> dict:
        """Für JSON-Export und den Verlaufsgraphen in der GUI."""
        return {
            "generation": self.generation,
            "population_size": self.population_size,
            "best_fitness": self.best_fitness,
            "average_fitness": self.average_fitness,
            "worst_fitness": self.worst_fitness,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "average_steps": self.average_steps,
            "total_blocked_steps": self.total_blocked_steps,
            "best_x_pos": self.best_x_pos,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationStats":
        """Wiederherstellung aus JSON-Daten mit Validierung."""
        try:
            stats = cls(
                generation=int(data["generation"]),
                population_size=int(data["population_size"]),
                best_fitness=float(data["best_fitness"]),
                average_fitness=float(data["average_fitness"]),
                worst_fitness=float(data["worst_fitness"]),
                success_count=int(data["success_count"]),
                success_rate=float(data["success_rate"]),
                average_steps=float(data["average_steps"]),
                total_blocked_steps=int(data["total_blocked_steps"]),
                best_x_pos=float(data["best_x_pos"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"GenerationStats.from_dict: ungültige Daten ({exc})") from exc
        if stats.population_size < 1:
            raise ValueError("GenerationStats: population_size muss >= 1 sein")
        return stats


class Experiment:
    """Verwaltet den kompletten evolutionären Ablauf.

    Attribute:
        env_name: gym-super-mario-bros-Environment (alle Instanzen
            laufen im selben Level - gleiche Bedingungen für alle).
        population: aktuelle Population.
        evolver: erzeugt die nächste Generation.
        max_steps: Schrittbudget pro Episode.
        generation: Nummer der AKTUELLEN Generation (0-basiert).
        history: Liste der GenerationStats aller abgeschlossenen Runden.
        best_overall_*: Rekord (Fitness, Label, Genom) über alle Generationen.
    """

    def __init__(self,
                 population: Population,
                 evolution_config: EvolutionConfig,
                 rng: np.random.Generator,
                 max_steps: int = 1000,
                 env_name: str = DEFAULT_ENV_NAME,
                 stall_ttl: int | None = 120):
        if not isinstance(population, Population):
            raise TypeError("Experiment: population muss Population sein")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("Experiment: rng muss np.random.Generator sein")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError(f"Experiment: max_steps muss int >= 1 sein, erhielt {max_steps!r}")
        self.population = population
        self.evolver = Evolver(evolution_config, rng)
        self.rng = rng
        self.max_steps = max_steps
        self.stall_ttl = stall_ttl
        self.env_name = env_name
        self.generation = 0
        self.history: list[GenerationStats] = []
        self.evolution_config = evolution_config
        # Rekord: Genom statt Objekt speichern, damit der Rekord von
        # Populationen unabhängig existiert (speicherbar/ladbar).
        self.best_overall_fitness: float | None = None
        self.best_overall_label: str | None = None
        self.best_overall_genome: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Simulation der aktuellen Generation
    # ------------------------------------------------------------------
    def start_round(self) -> MultiMarioSimulation:
        """Startet eine frische Simulation (N eigene Emulator-Instanzen)."""
        return MultiMarioSimulation(
            individuals=self.population.individuals,
            env_name=self.env_name,
            max_steps=self.max_steps,
            stall_ttl=self.stall_ttl,
        )

    def finish_round(self, simulation: MultiMarioSimulation) -> GenerationStats:
        """Bewertet eine abgeschlossene Simulation; aktualisiert Rekord/Statistik.

        Muss NACH Abschluss der Simulation aufgerufen werden; sonst
        Fehlermeldung statt stiller Teilbewertung. Danach werden die
        Emulator-Instanzen freigegeben.
        """
        if not simulation.is_finished():
            raise RuntimeError(
                "finish_round: Simulation ist noch nicht abgeschlossen "
                f"({simulation.active_count} Episoden aktiv)"
            )
        fitness_values = simulation.fitness_values()
        labels = [ind.label for ind in simulation.individuals]
        fitness_by_label = dict(zip(labels, fitness_values))

        stats = self._compute_stats(simulation, fitness_values)
        self.history.append(stats)
        self._update_record(simulation, fitness_by_label)

        # Fitness den Individuen der Population zuweisen.
        for ind in self.population.individuals:
            ind.fitness = fitness_by_label[ind.label]

        simulation.close()  # Emulator-Ressourcen freigeben
        return stats

    def _compute_stats(self, simulation: MultiMarioSimulation,
                       fitness_values: list[float]) -> GenerationStats:
        """Berechnet die Statistiken dieser Generation.

        Datenbasis: ALLE Episoden der Simulation nach Abschluss und die
        zugehörigen Fitnesswerte (Reihenfolge = Individuen).
        """
        n = len(fitness_values)
        success_count = sum(
            1 for e in simulation.episodes if e.status == EpisodeStatus.SUCCESS
        )
        return GenerationStats(
            generation=self.generation,
            population_size=n,
            best_fitness=float(max(fitness_values)),
            average_fitness=float(sum(fitness_values) / n),
            worst_fitness=float(min(fitness_values)),
            success_count=success_count,
            success_rate=success_count / n,
            average_steps=float(sum(e.steps for e in simulation.episodes) / n),
            total_blocked_steps=sum(e.blocked_steps for e in simulation.episodes),
            best_x_pos=max(float(e.last_info["x_pos"]) for e in simulation.episodes),
        )

    def _update_record(self, simulation: MultiMarioSimulation,
                       fitness_by_label: dict[str, float]) -> None:
        """Aktualisiert den Rekord (bester Lauf aller Zeiten).

        Ersetzung nur bei STRICT besserer Fitness - gleich gute Läufe
        ersetzen den Rekord nicht, damit die Anzeige stabil bleibt.
        """
        for ind in simulation.individuals:
            fitness = fitness_by_label[ind.label]
            if self.best_overall_fitness is None or fitness > self.best_overall_fitness:
                self.best_overall_fitness = fitness
                self.best_overall_label = ind.label
                self.best_overall_genome = ind.genome().copy()

    # ------------------------------------------------------------------
    # Generationenwechsel
    # ------------------------------------------------------------------
    def advance_generation(self) -> Population:
        """Erzeugt die nächste Generation (Selektion + Mutation).

        Die Population MUSS vorher bewertet sein (finish_round) - jedes
        Individuum braucht eine Fitness, sonst Fehlermeldung.
        """
        if any(ind.fitness is None for ind in self.population.individuals):
            raise RuntimeError(
                "advance_generation: Population ist nicht bewertet - "
                "erst finish_round() ausführen"
            )
        fitness_by_label = {ind.label: ind.fitness for ind in self.population.individuals}
        self.population = self.evolver.next_generation(self.population, fitness_by_label)
        self.generation += 1
        return self.population

    # ------------------------------------------------------------------
    # Speichern / Laden
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialisierbarer Zustand des Experiments (JSON-fähig)."""
        if self.best_overall_genome is None:
            raise RuntimeError("to_dict: kein Rekord vorhanden (erst finish_round)")
        return {
            "generation": self.generation,
            "hidden_sizes": self.population.hidden_sizes,
            "population_genomes": [ind.genome().tolist() for ind in self.population.individuals],
            "labels": [ind.label for ind in self.population.individuals],
            "fitness": [ind.fitness for ind in self.population.individuals],
            "history": [s.as_dict() for s in self.history],
            "best_overall_fitness": self.best_overall_fitness,
            "best_overall_label": self.best_overall_label,
            "best_overall_genome": self.best_overall_genome.tolist(),
            "env_name": self.env_name,
            "max_steps": self.max_steps,
        }

    def save(self, path: str | Path) -> None:
        """Speichert das Experiment als JSON-Datei."""
        payload = json.dumps(self.to_dict(), indent=2)
        Path(path).write_text(payload, encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path,
             evolution_config: EvolutionConfig, rng: np.random.Generator) -> "Experiment":
        """Lädt ein Experiment aus einer JSON-Datei.

        Fehler:
            ValueError bei fehlenden/inkonsistenten Feldern - kein
            stiller Fallback.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            hidden_sizes = [int(h) for h in data["hidden_sizes"]]
            genomes = data["population_genomes"]
            labels = data["labels"]
            fitness_list = data["fitness"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Experiment.load: Datei-Struktur ungültig ({exc})") from exc
        if not (len(genomes) == len(labels) == len(fitness_list)) or not genomes:
            raise ValueError(
                "Experiment.load: genomes, labels und fitness müssen gleich lang und nicht leer sein"
            )
        from .individual import Individual  # lokaler Import vermeidet Zirkelbezug
        template = Individual.random(hidden_sizes=hidden_sizes, rng=rng, label="tmp")
        individuals = [
            template.clone_with_genome(np.asarray(g, dtype=np.float64), label=labels[i])
            for i, g in enumerate(genomes)
        ]
        exp = cls(
            population=Population(individuals=individuals, hidden_sizes=hidden_sizes),
            evolution_config=evolution_config,
            rng=rng,
            max_steps=int(data.get("max_steps", 1000)),
            env_name=data.get("env_name", DEFAULT_ENV_NAME),
        )
        exp.generation = int(data["generation"])
        exp.history = [GenerationStats.from_dict(s) for s in data["history"]]
        exp.best_overall_fitness = float(data["best_overall_fitness"])
        exp.best_overall_label = data["best_overall_label"]
        exp.best_overall_genome = np.asarray(data["best_overall_genome"], dtype=np.float64)
        for ind, f in zip(individuals, fitness_list):
            ind.fitness = None if f is None else float(f)
        return exp
