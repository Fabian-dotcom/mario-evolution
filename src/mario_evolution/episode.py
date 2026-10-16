"""Episode: EIN unabhängiger Mario-Lauf auf einer gym-super-mario-bros-Instanz.

Jede Episode BESITZT ein eigenes Environment (eigene Emulator-Instanz
mit eigenem Spielzustand). So ist garantiert, dass die Fitness einer
Instanz niemals von den Bewegungen einer anderen abhängt - die
Emulatoren sind vollständig voneinander getrennt.

Lebenszyklus:
    Episode(env=...)            -> Status ACTIVE
    step(action_index)          -> ein Joypad-Schritt (Beobachtung VOR
                                   der Aktion gehört zum aktuellen Zustand)
    terminate(status)           -> vorzeitiger Abbruch (Stillstand,
                                   manuelles "alle beenden") - die Episode
                                   wird SAUBER abgeschlossen: Fitness ist
                                   danach berechenbar, Emulator wird
                                   erst bei close() freigegeben.
    result()/fitness(...)       -> Bewertung nach Abschluss
    close()                     -> Emulator freigeben (wichtig bei N Instanzen!)

Stillstand-Erkennung (TTL):
Ein Individuum, das mehr als `stall_ttl` Schritte LANG keine neue
Bestmarke der x-Position erreicht hat, wird beendet (Status
TERMINATED, Grund "stillstand"). Gemessen wird über das Maximum
der x-Position: kurze Rückwärtsbewegungen oder Sprünge auf der Stelle
brechen die TTL NICHT ab, echter Fortschritt schon.

Blockierte Schritte werden NUR gezählt, wenn Mario am Boden steht und
ein Hindernis/Gegner direkt neben ihm ist (präzise Definition statt
"jeder Schritt ohne x-Fortschritt" - Sprünge und Rückwärts-Korrekturen
werden so nicht mehr fälschlich bestraft).
"""

from __future__ import annotations

import numpy as np

from .actions import gym_action_index
from .observation import NO_PREVIOUS_ACTION, encode


class EpisodeStatus:
    """Konstanten für den Status einer Episode."""
    ACTIVE = "aktiv"
    SUCCESS = "erfolgreich"
    TERMINATED = "beendet"


class EndReason:
    """Konstanten für den End-Grund einer Episode (für GUI/Statistik)."""
    GOAL = "flagge_erreicht"
    DEAD = "gestorben"
    TTL = "schrittbudget_erreicht"
    STALL = "stillstand"
    MANUAL = "manuell_beendet"


class Episode:
    """Ein unabhängiger Mario-Lauf auf einer eigenen Emulator-Instanz.

    Attribute:
        env: gymnasium-Environment (gym-super-mario-bros) dieser Instanz.
        max_steps: Maximale Anzahl Joypad-Schritte pro Episode.
        stall_ttl: Wie viele Schritte ohne neue x-Bestmarke geduldet
            werden, bevor die Episode wegen Stillstand endet
            (0 oder None = deaktiviert).
        steps / blocked_steps: Zähler.
        end_reason: End-Grund (EndReason-Konstante), None solange aktiv.
        previous_action_index: letzte Aktion (One-Hot-Eingabe).
        last_frame / last_info: letzter Spielzustand.
        status: EpisodeStatus-Konstante.
    """

    def __init__(self, env, max_steps: int = 1000, stall_ttl: int = 120):
        if env is None:
            raise ValueError("Episode: env darf nicht None sein")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
            raise ValueError(f"Episode: max_steps muss int >= 1 sein, erhielt {max_steps!r}")
        if stall_ttl is not None:
            if not isinstance(stall_ttl, int) or isinstance(stall_ttl, bool) or stall_ttl < 0:
                raise ValueError(
                    f"Episode: stall_ttl muss int >= 0 (oder None) sein, erhielt {stall_ttl!r}"
                )
        self.env = env
        self.max_steps = max_steps
        self.stall_ttl = stall_ttl
        self.steps = 0
        self.blocked_steps = 0
        self.end_reason: str | None = None
        self.previous_action_index = NO_PREVIOUS_ACTION
        self._previous_y_pixel: float | None = None
        self._previous_x_pos: float | None = None
        self.last_frame: np.ndarray | None = None
        self.last_info: dict | None = None
        self.status = EpisodeStatus.ACTIVE
        self.reward_sum = 0.0
        # Stillstand-Buchhaltung: bestes bisher erreichtes x und Zähler
        # der Schritte seitdem.
        self._best_x = -1.0
        self._stall_counter = 0
        # Erstbeobachtung holen (eigener Emulatorzustand pro Instanz!).
        self.last_frame, self.last_info = self.env.reset()
        if self.last_frame is None or self.last_info is None:
            raise RuntimeError("Episode: env.reset() hat (None, None) geliefert")
        self._best_x = float(self.last_info["x_pos"])

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
            self._previous_x_pos,
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
            self._finish(EpisodeStatus.TERMINATED, EndReason.TTL)
            return

        joypad_byte = gym_action_index(action_index)
        frame, reward, terminated, truncated, info = self.env.step(joypad_byte)

        self.reward_sum += float(reward)
        prev_info = self.last_info
        x_before = float(prev_info["x_pos"])
        x_after = float(info["x_pos"])

        # --- Präzise Blockaden-Erkennung ---------------------------------
        # Ein Schritt zählt als "blockiert" NUR wenn:
        #   1. Mario am Boden ist (kein Sprung läuft, y >= ~191),
        #   2. sein x sich NICHT vergrößert hat,
        #   3. ein Hindernis/Gegner direkt neben ihm ist (Pixel-Sensor,
        #      Beobachtung des Zustands VOR diesem Schritt).
        # Sprünge (in der Luft) und Rückwärts-Korrekturen auf freier
        # Bahn fallen damit NICHT mehr unter die Strafe.
        on_ground = float(prev_info["y_pixel"]) >= 191.0
        obs = self.observation()
        obstacle_nearby = bool(obs[9] or obs[10])
        if on_ground and x_after <= x_before and obstacle_nearby:
            self.blocked_steps += 1

        # --- Stillstand-Erkennung (TTL) -----------------------------------
        # Neue x-Bestmarke => Fortschritt => TTL-Zähler zurücksetzen.
        if x_after > self._best_x:
            self._best_x = x_after
            self._stall_counter = 0
        else:
            self._stall_counter += 1

        self._previous_y_pixel = float(prev_info["y_pixel"])
        self._previous_x_pos = x_before
        self.last_frame = frame
        self.last_info = info
        self.steps += 1
        self.previous_action_index = action_index

        # --- Abschluss-Bedingungen ----------------------------------------
        if terminated or truncated:
            if info.get("flag_get"):
                self._finish(EpisodeStatus.SUCCESS, EndReason.GOAL)
            elif info.get("is_dead") or info.get("life", 2) < 2:
                self._finish(EpisodeStatus.TERMINATED, EndReason.DEAD)
            else:
                self._finish(EpisodeStatus.TERMINATED, EndReason.TTL)
        elif self.steps >= self.max_steps:
            self._finish(EpisodeStatus.TERMINATED, EndReason.TTL)
        elif (self.stall_ttl is not None and self.stall_ttl > 0
              and self._stall_counter >= self.stall_ttl):
            # Zu lange ohne Fortschritt -> sauber beenden, um Emulator-
            # Zeit für die anderen Instanzen zu sparen.
            self._finish(EpisodeStatus.TERMINATED, EndReason.STALL)

    # ------------------------------------------------------------------
    # Vorzeitiger, SAUBERER Abbruch
    # ------------------------------------------------------------------
    def terminate(self, reason: str = EndReason.MANUAL) -> None:
        """Beendet die Episode VORZEITIG, aber SAUBER.

        "Sauber": Status wird gesetzt (danach keine Schritte mehr),
        die Fitness ist sofort berechenbar (aktueller Zustand), der
        Emulator bleibt bis close() bestehen - die GUI kann das letzte
        Bild weiter zeigen. Idempotent: ein zweiter Aufruf tut nichts.
        """
        if self.status != EpisodeStatus.ACTIVE:
            return
        self._finish(EpisodeStatus.TERMINATED, reason)

    def _finish(self, status: str, reason: str) -> None:
        """Interner Abschluss: Status + End-Grund setzen (idempotent)."""
        if self.status != EpisodeStatus.ACTIVE:
            return
        self.status = status
        self.end_reason = reason

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

    @property
    def stall_remaining(self) -> int:
        """Restliche Stillstand-Toleranz in Schritten (-1 = deaktiviert)."""
        if self.stall_ttl is None or self.stall_ttl == 0:
            return -1
        return max(0, self.stall_ttl - self._stall_counter)

    def close(self) -> None:
        """Gibt den Emulator frei (zwingend bei vielen Instanzen)."""
        try:
            self.env.close()
        except Exception:
            # Emulator-Ressourcen freigeben darf die Anwendung nicht
            # abstürzen lassen; der Fehler ist hier unkritisch.
            pass
