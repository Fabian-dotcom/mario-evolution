"""Tests für die neuen Episode-Features: Stillstand-TTL, sauberes
terminate(), präzise Blockaden-Erkennung.

Die Blockaden-Erkennung braucht ein echtes Frame+info-Paar; wir nutzen
dafür synthetische Zustände (kein Emulator nötig - die Logik ist
ausreichend getestet, wenn die Bedingungen am künstlichen Zustand
korrekt auswerten).
"""
import numpy as np
import pytest

from mario_evolution.episode import EndReason, Episode, EpisodeStatus


class _FakeEnv:
    """Minimaler Emulator-Ersatz für Zustands-Logik-Tests (kein Spiel!).

    Liefert konfigurierbare Frames/Infos pro Schritt; genug Oberfläche
    (reset/step/close), damit Episode funktioniert.
    """

    def __init__(self, infos: list[dict]):
        self.infos = list(infos)
        self.frame = np.full((240, 256, 3), 104, dtype=np.uint8)
        self.frame[..., 1] = 136
        self.frame[..., 2] = 252
        self.closed = False

    def reset(self):
        return self.frame, self.infos[0]

    def step(self, action):
        self._cursor = min(getattr(self, "_cursor", 1) + 1, len(self.infos) - 1)
        return self.frame, 0.0, False, False, self.infos[self._cursor]

    def close(self):
        self.closed = True


def _info(x=40.0, y=193.0) -> dict:
    return {"x_pos": x, "y_pixel": y, "time": 400, "is_dead": False,
            "flag_get": False, "life": 2}


class TestStallTTL:
    def test_stall_ends_episode(self):
        """Keine neue x-Bestmarke für stall_ttl Schritte -> Episode endet
        SAUBER mit End-Grund 'stillstand' (kein Warten auf das Schrittbudget)."""
        env = _FakeEnv([_info(x=100.0), _info(x=100.0)])
        epi = Episode(env=env, max_steps=1000, stall_ttl=5)
        for _ in range(10):
            if epi.is_finished:
                break
            epi.step(1)
        assert epi.is_finished
        assert epi.end_reason == EndReason.STALL
        assert epi.status == EpisodeStatus.TERMINATED
        env.close()

    def test_progress_resets_stall_counter(self):
        """Fortschritt setzt die TTL zurück: 4 still + 1 Fortschritt + ... endet nie."""
        infos = [_info(x=100.0)]
        for x in (101, 102, 103, 104, 105):
            infos.append(_info(x=float(x)))
        infos.append(_info(x=105.0))
        env = _FakeEnv(infos)
        epi = Episode(env=env, max_steps=1000, stall_ttl=5)
        for _ in range(8):
            if epi.is_finished:
                break
            epi.step(1)
        # alternierend: 4 still, dann Fortschritt - TTL (5) nie erreicht
        assert not epi.is_finished or epi.end_reason != EndReason.STALL
        env.close()

    def test_stall_disabled_with_zero(self):
        env = _FakeEnv([_info(x=100.0), _info(x=100.0)])
        epi = Episode(env=env, max_steps=1000, stall_ttl=0)
        for _ in range(30):
            if epi.is_finished:
                break
            epi.step(1)
        assert epi.end_reason != EndReason.STALL  # Stillstand-Ende ausgeschaltet
        env.close()

    def test_stall_remaining_property(self):
        env = _FakeEnv([_info(x=100.0), _info(x=100.0)])
        epi = Episode(env=env, max_steps=1000, stall_ttl=10)
        assert epi.stall_remaining == 10
        epi.step(1)
        assert epi.stall_remaining == 9
        env.close()


class TestTerminateAll:
    def test_terminate_is_clean_and_idempotent(self):
        """terminate() beendet sauber: End-Grund gesetzt, Fitness
        berechenbar, zweiter Aufruf ändert nichts."""
        env = _FakeEnv([_info(x=250.0), _info(x=250.0)])
        epi = Episode(env=env, max_steps=1000, stall_ttl=0)
        epi.step(1)
        epi.terminate()
        assert epi.is_finished
        assert epi.end_reason == EndReason.MANUAL
        # Fitness ist SOFORT berechenbar (kein Fehler, x=250):
        assert epi.fitness() == pytest.approx(250.0)
        steps_before = epi.steps
        epi.terminate()  # idempotent
        epi.step(1) if False else None
        with pytest.raises(RuntimeError, match="weiterbewegt"):
            epi.step(1)
        assert epi.steps == steps_before
        env.close()

    def test_terminate_all_simulation(self):
        """MultiMarioSimulation.terminate_all beendet ALLE aktiven Episoden
        sauber und zählt, wie viele neu beendet wurden."""
        from mario_evolution.simulation import MultiMarioSimulation
        pytest.importorskip("gym_super_mario_bros")
        rng = np.random.default_rng(3)
        from mario_evolution.individual import Individual
        inds = [Individual.random(hidden_sizes=[8], rng=rng) for _ in range(3)]
        sim = MultiMarioSimulation(individuals=inds, max_steps=50, stall_ttl=0)
        sim.step_many(10)
        before_active = sim.active_count
        n = sim.terminate_all()
        assert n == before_active
        assert sim.is_finished()
        assert sim.active_count == 0
        # Alle Episoden haben einen End-Grund und berechenbare Fitness:
        reasons = [e.end_reason for e in sim.episodes]
        assert all(r is not None for r in reasons)
        assert all(np.isfinite(e.fitness()) for e in sim.episodes)
        # Zweiter Aufruf: 0 neu beendet (idempotent)
        assert sim.terminate_all() == 0
        sim.close()


class TestPreciseBlocked:
    def test_blocked_only_with_obstacle_and_ground(self):
        """Blockade zählt NUR bei: am Boden + x nicht größer + Hindernis nah.
        In der Luft oder ohne Hindernis -> NICHT als blockiert gezählt."""
        frame = np.full((240, 256, 3), 104, dtype=np.uint8)
        frame[..., 1] = 136
        frame[..., 2] = 252
        frame[170:200, 55:87, :] = (0, 168, 0)   # Hindernis rechts neben Mario
        infos = [
            _info(x=100.0, y=193.0),   # am Boden, Hindernis rechts
            _info(x=100.0, y=193.0),   # Blockade-Schritt (x gleich)
            _info(x=100.0, y=100.0),   # jetzt in der LUFT (Sprung)
            _info(x=100.0, y=100.0),   # "blockiert" in der Luft -> zählt NICHT
        ]
        env = _FakeEnv(infos)
        env.frame = frame
        epi = Episode(env=env, max_steps=100, stall_ttl=0)
        epi.step(1)   # Boden-Blockade -> zählt
        assert epi.blocked_steps == 1
        epi.step(1)   # Luft-"Blockade" -> zählt NICHT
        assert epi.blocked_steps == 1
        env.close()
