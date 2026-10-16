"""Tests für Aktionen, Beobachtung und Fitness (gym-super-mario-bros)."""
import numpy as np
import pytest

from mario_evolution.actions import (ALL_ACTIONS, ACTION_COUNT, choose_action,
                                     gym_action_index)
from mario_evolution.fitness import EpisodeResult, FitnessWeights, compute_fitness


class TestActions:
    def test_simple_movement_compatibility(self):
        """Unsere 7 Aktionen == SIMPLE_MOVEMENT des Pakets (Interface-Vertrag)."""
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        assert ACTION_COUNT == len(SIMPLE_MOVEMENT) == 7

    def test_bitmasks_verified_against_emulator_encoding(self):
        """Joypad-Bits: right=128, right+A=129, left=64, A=1, NOOP=0
        (experimentell gegen den Emulator verifiziert, siehe README)."""
        assert gym_action_index(0) == 0     # halten
        assert gym_action_index(1) == 128   # rechts
        assert gym_action_index(2) == 129   # rechts + A
        assert gym_action_index(5) == 1     # springen (nur A)
        assert gym_action_index(6) == 64    # links

    def test_invalid_action_index(self):
        with pytest.raises(ValueError):
            gym_action_index(7)
        with pytest.raises(TypeError):
            gym_action_index(True)

    def test_choose_action_argmax(self):
        """Größter Ausgabewert wählt die Aktion (deterministisch, von Hand prüfbar)."""
        assert choose_action([0.1, 0.9, 0.2, 0.0, 0.0, 0.0, 0.0]) == 1   # rechts
        assert choose_action([0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0]) == 5   # springen
        assert choose_action([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4]) == 6   # links

    def test_choose_action_tie_break_smallest_index(self):
        assert choose_action([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]) == 0

    def test_choose_action_rejects_bad_values(self):
        with pytest.raises(ValueError, match="endlich"):
            choose_action([np.nan, 0, 0, 0, 0, 0, 0])
        with pytest.raises(ValueError, match="Werte"):
            choose_action([0.0, 0.0])   # falsche Anzahl


def _synth_frame(walls: list[str]) -> np.ndarray:
    """Erzeugt ein synthetisches NES-Bild (240x256x3).

    Himmel überall; in den angegebenen Regionen ('links', 'rechts') wird
    ein solider Block (Rohrgrün, wie ein echtes Rohr) in Bodennähe
    platziert, um die Pixel-Sensorik zu testen. 'beige' platziert
    Goomba-Fellfarben (Gegner-Erkennung).
    """
    frame = np.full((240, 256, 3), 104, dtype=np.uint8)
    frame[..., 1] = 136
    frame[..., 2] = 252
    if "links" in walls:
        frame[170:200, 7:39, :] = (0, 168, 0)     # Block links von Mario
    if "rechts" in walls:
        frame[170:200, 55:87, :] = (0, 168, 0)    # Block rechts
    if "beige" in walls:
        frame[190:206, 55:87, :] = (240, 208, 176)  # Goomba-Fell rechts
    return frame


def _base_info(**overrides) -> dict:
    info = {"x_pos": 100.0, "y_pixel": 193.0, "time": 400, "is_dead": False,
            "flag_get": False}
    info.update(overrides)
    return info


class TestObservation:
    def test_observation_size_and_range(self):
        from mario_evolution.observation import encode, OBSERVATION_SIZE
        obs = encode(_synth_frame([]), _base_info())
        assert obs.shape == (OBSERVATION_SIZE,)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    def test_distance_decreases_when_mario_moves_right(self):
        """Reststrecke (Index 0) schrumpft, wenn x_pos wächst - handprüfbar."""
        from mario_evolution.observation import encode
        far = encode(_synth_frame([]), _base_info(x_pos=100.0))
        near = encode(_synth_frame([]), _base_info(x_pos=1100.0))
        assert near[0] < far[0]
        # exakt: (4800-100)/4800 - (4800-1100)/4800 = 1000/4800
        assert far[0] - near[0] == pytest.approx(1000.0 / 4800.0, rel=1e-6)

    def test_wall_detection(self):
        """Pixel-Sensorik: solider Block neben Mario -> Index 7/8 wird 1."""
        from mario_evolution.observation import encode
        clean = encode(_synth_frame([]), _base_info())
        left = encode(_synth_frame(["links"]), _base_info())
        right = encode(_synth_frame(["rechts"]), _base_info())
        assert clean[7] == 0.0 and clean[8] == 0.0
        assert left[7] == 1.0
        assert right[8] == 1.0

    def test_enemy_detection(self):
        """Gegner (Goomba-Fell beige im Streifen) wird als Hindernis erkannt."""
        from mario_evolution.observation import encode
        clean = encode(_synth_frame([]), _base_info())
        enemy = encode(_synth_frame(["beige"]), _base_info())
        assert clean[8] == 0.0
        assert enemy[8] == 1.0

    def test_different_inputs_different_outputs(self):
        """Das Netz kann auf die Beobachtung reagieren: Unterschiedliche
        Zustände erzeugen unterschiedliche Vektoren (kein Star-Code)."""
        from mario_evolution.observation import encode
        o1 = encode(_synth_frame([]), _base_info(x_pos=100.0))
        o2 = encode(_synth_frame(["rechts"]), _base_info(x_pos=500.0, time=200))
        assert not np.allclose(o1, o2)

    def test_one_hot_previous_action(self):
        from mario_evolution.observation import encode
        none = encode(_synth_frame([]), _base_info())               # keine Aktion
        right = encode(_synth_frame([]), _base_info(), previous_action_index=1)
        assert np.allclose(none[9:13], 0.0)
        assert right[10] == 1.0 and right[9] == 0.0

    def test_invalid_inputs_rejected(self):
        from mario_evolution.observation import encode
        with pytest.raises(ValueError, match="Form"):
            encode(np.zeros((10, 10, 3)), _base_info())
        with pytest.raises(ValueError, match="fehlenden|fehlen"):
            encode(_synth_frame([]), {"x_pos": 1.0})  # y_pixel/time fehlen
        with pytest.raises(ValueError, match="previous_action_index"):
            encode(_synth_frame([]), _base_info(), previous_action_index=99)

    def test_labels_match_size(self):
        from mario_evolution.observation import observation_labels, OBSERVATION_SIZE
        assert len(observation_labels()) == OBSERVATION_SIZE


class TestFitness:
    def test_progress_only(self):
        r = EpisodeResult(reached_goal=False, steps=100, blocked_steps=0,
                          final_x_pos=400.0, reward_sum=0.0)
        assert compute_fitness(r) == pytest.approx(400.0)

    def test_goal_dominates(self):
        """Zielbonus (5000) überstrahlt jeden Fortschritt (max ~4800)."""
        r_fail = EpisodeResult(False, 900, 0, 4700.0, 0.0)
        r_win = EpisodeResult(True, 900, 0, 4800.0, 0.0)
        assert compute_fitness(r_win) > compute_fitness(r_fail) + 500.0

    def test_faster_success_scores_higher(self):
        """Effizientere Erfolge unterscheiden sich (nicht alle gleich gut)."""
        slow = EpisodeResult(True, 900, 400, 4800.0, 0.0)
        fast = EpisodeResult(True, 500, 50, 4800.0, 0.0)
        assert compute_fitness(fast) > compute_fitness(slow)

    def test_blocked_steps_penalized(self):
        a = EpisodeResult(False, 100, 0, 300.0, 0.0)
        b = EpisodeResult(False, 100, 20, 300.0, 0.0)
        assert compute_fitness(a) > compute_fitness(b)

    def test_reward_adds_secondary_signal(self):
        a = EpisodeResult(False, 100, 0, 300.0, 10.0)
        b = EpisodeResult(False, 100, 0, 300.0, 0.0)
        assert compute_fitness(a) == pytest.approx(compute_fitness(b) + 5.0)

    def test_negative_fitness_possible(self):
        r = EpisodeResult(False, 1000, 2000, 0.0, 0.0)
        assert compute_fitness(r) < 0

    def test_invalid_weights_rejected(self):
        with pytest.raises(ValueError, match="goal_bonus"):
            FitnessWeights(goal_bonus=-1.0)

    def test_invalid_result_rejected(self):
        with pytest.raises(TypeError):
            compute_fitness({"x_pos": 1})
