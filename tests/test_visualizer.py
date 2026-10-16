"""Tests für das Dashboard (visualizer.py)."""
import tkinter as tk
import numpy as np
import pytest

from mario_evolution.visualizer import MarioEvolutionApp
from mario_evolution.episode import EpisodeStatus


class MockEpisode:
    def __init__(self, x_pos=120, steps=20, blocked=2, status=EpisodeStatus.ACTIVE):
        self.last_frame = np.zeros((240, 256, 3), dtype=np.uint8)
        self.last_info = {"x_pos": x_pos, "y_pixel": 193.0, "time": 400}
        self.steps = steps
        self.blocked_steps = blocked
        self.status = status
        self.end_reason = None
        self.stall_remaining = 10

    def fitness(self):
        return float(self.last_info["x_pos"])

    def observation(self):
        from mario_evolution.observation import OBSERVATION_SIZE
        return np.linspace(0.1, 0.9, OBSERVATION_SIZE)


class MockSimulation:
    def __init__(self, episodes, individuals):
        self.episodes = episodes
        self.individuals = individuals

    def is_finished(self):
        return all(e.status != EpisodeStatus.ACTIVE for e in self.episodes)

    @property
    def active_count(self):
        return sum(1 for e in self.episodes if e.status == EpisodeStatus.ACTIVE)


def test_visualizer_layout_and_network_canvas():
    """Prüft, dass das Dashboard aufbaut und das neuronale Netz den Platz voll einnimmt."""
    root = tk.Tk()
    try:
        app = MarioEvolutionApp(root)
        root.update()

        # Canvas muss eine spürbare Größe haben (> 200px)
        cw = app.canvas_net.winfo_width()
        ch = app.canvas_net.winfo_height()
        assert cw >= 300, f"Canvas-Breite zu klein: {cw}"
        assert ch >= 180, f"Canvas-Höhe zu klein: {ch}"

        # Bounding-Box des gezeichneten Netzes muss den Canvas weitgehend ausfüllen
        bbox = app.canvas_net.bbox("all")
        assert bbox is not None, "Keine Elemente auf canvas_net gezeichnet"
        bx0, by0, bx1, by1 = bbox
        assert bx0 >= 0, f"Elemente ragen links heraus: {bx0}"
        assert by0 >= 0, f"Elemente ragen oben heraus: {by0}"
        drawn_w = bx1 - bx0
        drawn_h = by1 - by0
        assert drawn_w >= cw * 0.8, f"Netzbreite ({drawn_w}) füllt Canvas ({cw}) nicht aus"
        assert drawn_h >= ch * 0.7, f"Netzhöhe ({drawn_h}) füllt Canvas ({ch}) nicht aus"

        # Simulation anhängen und prüfen, dass Netzwerk-Aktivierungen aktualisiert werden
        mock_sim = MockSimulation(
            episodes=[MockEpisode(100), MockEpisode(200)],
            individuals=app.experiment.population.individuals[:2],
        )
        app.simulation = mock_sim
        app._draw_all()
        root.update()

        # Canvas-Bounding-Box nach Simulation prüfen
        bbox_sim = app.canvas_net.bbox("all")
        assert bbox_sim is not None
        assert bbox_sim[2] - bbox_sim[0] >= app.canvas_net.winfo_width() * 0.8
    finally:
        app.running = False
        if app._after_id:
            root.after_cancel(app._after_id)
        root.destroy()
