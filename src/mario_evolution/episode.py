"""Episode: EIN unabhängiger Mario-Lauf auf einer gym-super-mario-bros-Instanz.

Jede Episode BESITZT ein eigenes Environment (eigene Emulator-Instanz
mit eigenem Spielzustand). So ist garantiert, dass die Fitness einer
Instanz niemals von den Bewegungen einer anderen abhängt - die
Emulatoren sind vollständig voneinander getrennt.

Lebenszyklus:
    Episode(env=...)            -> Status ACTIVE
    step(action_index)          -> ein Joypad-Schritt (Beobachtung VOR
                                   der Aktion gehört zum aktuellen Zustand)
    result()/fitness(...)       -> Bewertung nach Abschluss
    close()                     -> Emulator freigeben (wichtig bei N Instanzen!)
"""

from __future__ import annotations

import numpy as np

from .actions import ALL_ACTIONS, gym_action_index
from .observation import NO_PREVIOUS_ACTION, encode


class EpisodeStatus:
    """Konstanten für den Status einer Episode."""
    ACTIVE = "aktiv"
    SUCCESS = "erfolgreich"
    TERMINATED = "beendet"


class Episode:
    """Ein unabhängiger Mario-Lauf auf einer eigenen Emulator-Instanz.

    Attribute:
        env: gymnasium-Environment (gym-super-mario-bros) dieser Instanz.
        max_steps: Maximale Anzahl Joypad-Schritte pro Episode.
        steps / blocked_steps: Zähler (s. unten).
        previous_action_index: letzte Aktion (One-Hot-Eingabe).
        last_frame / last_info: letzter Spielzustand.
        status: EpisodeStatus-Konstante.
    """

    def __init__(self, env, max_steps: int = 1000):
        if env is None:
            raise ValueError("Episode: env darf nicht None sein")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError(f"Episode: max_steps muss int >= 1 sein, erhielt {max_steps!r}")
        self.env = env
        self.max_steps = max_steps
        self.steps = 0
        # "blockierte Bewegung" im originalen Mario: Mario hat sich NICHT
        # nach vorne bewegt (x_pos unverändert), z.B. gegen ein Rohr.
        self.blocked_steps = 0
        self.previous_action_index = NO_PREVIOUS_ACTION
        self._previous_y_pixel: float | None = None
        self.last_frame: np.ndarray | None = None
        self.last_info: dict | None = None
        self.status = EpisodeStatus.ACTIVE
        self.reward_sum = 0.0
        # Erstbeobachtung holen (eigener Emulatorzustand pro Instanz!).
        self.last_frame, self.last_info = self.env.reset()
        if self.last_frame is None or self.last_info is None:
            raise RuntimeError("Episode: env.reset() hat (None, None) geliefert")

    # ------------------------------------------------------------------
    # Beobachtung
    # ------------------------------------------------------------------
    def observation(self) -> np.ndarray:
        """Beobachtungsvektor des AKTUELLEN Zustands dieser Instanz.

        Wichtig (Auftrag): Die Beobachtung gehört zur Episode VOR der
        nächsten Aktion - sie wird aus last_frame/last_info erzeugt,
        nicht aus irgendeinem globalen Zustand.
        """
        return encode(
            self.last_frame,
            self.last_info,
            self.previous_action_index,
            self._previous_y_pixel,
        )

    # ------------------------------------------------------------------
    # Schritt
    # ------------------------------------------------------------------
    def step(self, action_index: int) -> None:
        """Führt EINE Aktion im Emulator aus (nur wenn Episode aktiv).

        Fehler:
            RuntimeError, wenn die Episode bereits beendet ist - eine
            beendete Instanz darf nicht versehentlich weiterbewegt werden.
        """
        if self.status != EpisodeStatus.ACTIVE:
            raise RuntimeError(
                f"Episode.step: Episode ist '{self.status}' und kann nicht weiterbewegt werden"
            )
        if self.steps >= self.max_steps:
            self.status = EpisodeStatus.TERMINATED
            return

        joypad_byte = gym_action_index(action_index)
        x_before = float(self.last_info["x_pos"])
        frame, reward, terminated, truncated, info = self.env.step(joypad_byte)

        # Blockierte Bewegung registrieren: Mario bleibt an gleicher
        # x-Position (z.B. vor einem Rohr) => zählen, nicht verstecken.
        if float(info["x_pos"]) <= x_before:
            self.blocked_steps += 1

        self.reward_sum += float(reward)
        self._previous_y_pixel = float(self.last_info["y_pixel"])
        self.last_frame = frame
        self.last_info = info
        self.steps += 1
        self.previous_action_index = action_index

        if terminated or truncated:
            # Erfolg: flag_get True = Level geschafft.
            self.status = (
                EpisodeStatus.SUCCESS if info.get("flag_get") else EpisodeStatus.TERMINATED
            )
        elif self.steps >= self.max_steps:
            self.status = EpisodeStatus.TERMINATED

    # ------------------------------------------------------------------
    # Ergebnis
    # ------------------------------------------------------------------
    def result(self):
        """EpisodeResult für fitness.compute_fitness (nach Abschluss)."""
        from .fitness import EpisodeResult
        return EpisodeResult(
            reached_goal=self.status == EpisodeStatus.SUCCESS,
            steps=self.steps,
            blocked_steps=self.blocked_steps,
            final_x_pos=float(self.last_info["x_pos"]),
            reward_sum=self.reward_sum,
        )

    def fitness(self, weights=None) -> float:
        """Fitness dieses Laufs (weitergeleitet an fitness.compute_fitness)."""
        from .fitness import compute_fitness
        return compute_fitness(self.result(), weights)

    @property
    def is_finished(self) -> bool:
        """True, wenn die Episode nicht mehr aktiv ist."""
        return self.status != EpisodeStatus.ACTIVE

    def close(self) -> None:
        """Gibt den Emulator frei (zwingend bei vielen Instanzen)."""
        try:
            self.env.close()
        except Exception:
            # Emulator-Ressourcen freigeben darf die Anwendung nicht
            # abstürzen lassen; der Fehler ist hier unkritisch.
            pass
