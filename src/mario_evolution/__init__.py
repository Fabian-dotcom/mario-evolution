"""mario_evolution - Paket für das Mario Neuroevolution Lab.

Grundlage des Spiels: gym-super-mario-bros
(github.com/Kautenja/gym-super-mario-bros) - echte NES-Emulation über
nes-py mit gymnasium-Schnittstelle.

Alle Module mit klaren Grenzen:

- actions.py      : Aktionen (SIMPLE_MOVEMENT als Joypad-Bitmasken)
- observation.py  : Spielzustand (NES-Frame + info) -> Eingabevektor
- network.py      : Feedforward-Netz (reines NumPy, nach "The Nature of Code")
- individual.py   : Individuum = Netzwerk + Genom + Fitness
- fitness.py      : Fitnessbewertung eines Mario-Laufs
- population.py   : Erzeugung von Populationen
- evolution.py    : Selektion, Elitismus, Mutation, neue Generationen
- experiment.py   : Generationen, Statistiken, Rekord, Save/Load
- episode.py      : EIN unabhängiger Mario-Lauf (eigene Emulator-Instanz)
- simulation.py   : Lockstep-Parallelisierung vieler Episoden
- visualizer.py   : Tkinter-Dashboard
"""

__version__ = "2.0.0"
