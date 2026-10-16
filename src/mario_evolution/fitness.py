"""Fitness für gym-super-mario-bros-Läufe.

Die Fitness ist DIE Zielgröße der Evolution. Sie muss
(a) Fortschritt belohnen (damit frühe "blinde" Netze eine Richtung haben),
(b) Zielerreichung klar dominieren lassen,
(c) Effizienz belohnen, ohne willkürliche Strafen, die (b) dominieren.

Aufbau (alle Anteile transparent skaliert):

    fitness =  1.0 * final_x_pos            Fortschritt (Pixel nach rechts)
             + 5000 * (1 wenn Flagge)       Zielbonus
             -    1 * blocked_steps         kleine Strafe (Anlaufen an Wände)
             +  0.5 * reward_sum            interne Belohnungen (Münzen etc.)

Warum final_x_pos als Fortschrittsmaß?
In gym-super-mario-bros ist x_pos Marios horizontale Weltposition in
Pixeln (aus dem NES-RAM gelesen, über die info-Daten geliefert). Das
Ziel liegt bei ~4800 Pixel. Fortschritt ist also direkt und linear
messbar - keine willkürliche Formel. Dies ersetzt die BFS-Distanz des
früheren Gitter-Designs: In Mario gibt es keinen exakt berechenbaren
kürzesten Weg, aber "wie weit rechts" ist das natürliche Fortschritts-
maß des Spiels.

Warum kein zusätzlicher Schritt-Strafterm? Das Environment begrenzt
Schritte bereits über die Spielzeit (time) und den Tod. Ein Pauschal-
Abzug pro Schritt würde Fortschritt indirekt bestrafen. Blockierte
Schritte (keine Vorwärtsbewegung, z.B. Anlaufen an ein Rohr) werden
dagegen bestraft - sie zeigen konkretes Fehlverhalten.

Bezug zu "The Nature of Code":
Kapitel Neuroevolution ("fitness and selection"): Die Fitness misst,
wie gut ein Individuum die Aufgabe löst; Selektion bevorzugt höhere
Werte. Die konkrete Formel ist Anwendungslogik für Mario.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeResult:
    """Ergebnis-Essenz eines Laufs (von episode.py erzeugt).

    Attribute:
        reached_goal: Hat Mario die Flagge erreicht?
        steps: Anzahl der Joypad-Schritte.
        blocked_steps: Anzahl Schritte ohne Vorwärtsbewegung.
        final_x_pos: Horizontale Endposition in Pixeln.
        reward_sum: Summe der Environment-Rewards.
    """

    reached_goal: bool
    steps: int
    blocked_steps: int
    final_x_pos: float
    reward_sum: float


@dataclass
class FitnessWeights:
    """Gewichte der Fitnesskomponenten (konfigurierbar, dokumentiert)."""

    progress_scale: float = 1.0    # Punkte pro Pixel nach rechts
    goal_bonus: float = 5000.0     # Bonus für die Flagge
    blocked_penalty: float = 1.0   # Abzug pro blockiertem Schritt
    reward_scale: float = 0.5      # Faktor für Environment-Reward

    def __post_init__(self) -> None:
        for name in ("progress_scale", "goal_bonus", "blocked_penalty", "reward_scale"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"FitnessWeights.{name} muss eine nicht-negative Zahl sein, erhielt {value!r}"
                )


def compute_fitness(result: EpisodeResult,
                    weights: FitnessWeights | None = None) -> float:
    """Berechnet die Fitness aus einem EpisodeResult.

    Rückgabe: Fitnesswert (kann negativ sein bei sehr schlechten Läufen).
    """
    if weights is None:
        weights = FitnessWeights()
    if not isinstance(result, EpisodeResult):
        raise TypeError(
            f"compute_fitness: result muss EpisodeResult sein, erhielt {type(result).__name__}"
        )

    # 1) Fortschritt: x-Position in Pixeln (Ziel ~4800 -> bei Erfolg ~4800+Bonus).
    progress_points = weights.progress_scale * result.final_x_pos

    # 2) Zielbonus (dominiert alle Strafen klar).
    goal_points = weights.goal_bonus if result.reached_goal else 0.0

    # 3) kleine Strafe: blockierte Schritte (Anlaufen an Hindernisse).
    blocked_points = weights.blocked_penalty * result.blocked_steps

    # 4) internes Reward-Signal (Münzen etc.) als kleines Sekundärmerkmal.
    reward_points = weights.reward_scale * result.reward_sum

    total = progress_points + goal_points - blocked_points + reward_points
    return float(total)
