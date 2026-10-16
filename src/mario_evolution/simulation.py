"""Parallele Simulation vieler Mario-Instanzen auf gym-super-mario-bros.

Wichtig ist der Unterschied zwischen paralleler DARSTELLUNG (viele
Spielfelder gleichzeitig sichtbar) und paralleler BERECHNUNG (echtes
Multiprocessing). Für die erste Version reicht eine gemeinsame,
effiziente Simulationsschleife ("lockstep"): pro Simulationsschritt
wird für ALLE noch aktiven Episoden je EIN Joypad-Schritt ausgeführt.
Die Emulator-Instanzen sind vollständig voneinander getrennt (jede
Episode besitzt ihr eigenes gymnasium-Environment), so hängt die
Fitness einer Instanz nie von einer anderen ab.

Hinweis zur Last: Jede Episode ist ein NES-Emulator (nes-py). Die
Anzahl der gleichzeitigen Instanzen sollte daher moderat bleiben
(Standard: 8-20). Die GUI zeigt alle Instanzen als Raster an.
"""

from __future__ import annotations

import numpy as np

from .episode import Episode, EpisodeStatus
from .individual import Individual


def make_env(env_name: str = "SuperMarioBros-1-1-v0", frameskip: int = 4):
    """Erzeugt EINE gym-super-mario-bros-Environment-Instanz.

    Argumente:
        env_name: gymnasium-Environment-Name (Standard: Level 1-1,
            4 Frames pro Aktion - jeder Joypad-Schritt fasst 4 Frames
            zusammen, das entspricht dem "Standard"-Modus des Pakets).
        frameskip: Informationell - das Paket -v0/Varianten legen
            frameskip intern fest; der Parameter ist hier dokumentiert
            statt still geändert zu werden.

    Rückgabe: gymnasium-Environment.

    Hinweis: Wir nutzen die RAW-Environment-Schnittstelle (Joypad-Byte).
    Die Joypad-Einschränkung auf SIMPLE_MOVEMENT geschieht NICHT über
    den JoypadSpace-Wrapper, sondern dadurch, dass unser Aktionsindex
    (siehe actions.py) die SIMPLE_MOVEMENT-Bitmasken abbildet. So
    bleibt die Aktionsmenge dokumentiert und an einer Stelle definiert.
    """
    import gym_super_mario_bros  # lokaler Import, schnelle Tests ohne Emulator möglich

    env = gym_super_mario_bros.make(env_name)
    return env


class MultiMarioSimulation:
    """Verwaltet N gleichzeitige Mario-Episoden (lockstep).

    Attribute:
        episodes: Liste mit GENAU EINER Episode pro Individuum.
        individuals: zugehörige Individuen (Reihenfolge stabil).
        max_steps: Schrittbudget pro Episode.
    """

    def __init__(self,
                 individuals: list[Individual],
                 env_name: str = "SuperMarioBros-1-1-v0",
                 max_steps: int = 1000):
        if not isinstance(individuals, list) or not individuals:
            raise ValueError("MultiMarioSimulation: individuals muss nicht-leere Liste sein")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError(f"MultiMarioSimulation: max_steps muss int >= 1 sein, erhielt {max_steps!r}")
        self.individuals = list(individuals)
        self.max_steps = max_steps
        # Für JEDES Individuum eine eigene Episode mit eigenem Emulator.
        self.episodes: list[Episode] = [
            Episode(env=make_env(env_name), max_steps=max_steps)
            for _ in self.individuals
        ]

    @property
    def active_count(self) -> int:
        """Anzahl der noch aktiven Episoden."""
        return sum(1 for e in self.episodes if e.status == EpisodeStatus.ACTIVE)

    def step_once(self) -> None:
        """EIN gemeinsamer Simulationsschritt für alle aktiven Episoden.

        Für jede aktive Episode:
        1. Beobachtung des AKTUELLEN Zustands der Episode erzeugen.
        2. Netzwerk des zugehörigen Individuums entscheidet die Aktion.
        3. Aktion im eigenen Emulator ausführen.

        Beendete Episoden werden übersprungen (nicht weiterbewegt).
        """
        for individual, episode in zip(self.individuals, self.episodes):
            if episode.status != EpisodeStatus.ACTIVE:
                continue
            obs = episode.observation()
            action = individual.act(obs)
            episode.step(action)

    def step_many(self, n: int) -> None:
        """Führt n gemeinsame Simulationsschritte aus (GUI-Tick)."""
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"step_many: n muss int >= 1 sein, erhielt {n!r}")
        for _ in range(n):
            if self.active_count == 0:
                break  # alle fertig -> sauber beenden
            self.step_once()

    def fitness_values(self) -> list[float]:
        """Fitness aller Episoden (Reihenfolge = Individuen)."""
        return [e.fitness() for e in self.episodes]

    def is_finished(self) -> bool:
        """True, wenn alle Episoden beendet sind."""
        return self.active_count == 0

    def close(self) -> None:
        """Gibt ALLE Emulator-Instanzen frei (GUI-Beenden, Generationenwechsel)."""
        for e in self.episodes:
            e.close()
