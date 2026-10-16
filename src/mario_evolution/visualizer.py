"""Tkinter-Dashboard für das Mario Evolution Lab (gym-super-mario-bros).

Aufbau des Fensters:

+--------------------------------------------------------------+
| Kopfzeile: Titel, Generation, Steuerung (Start/Pause/Schritt)|
+---------------------------------------+----------------------+
| LINKS: Raster aller Mario-Instanzen   | RECHTS OBEN:         |
| (Canvas, je Instanz ein Mini-         | bester Lauf (Detail) |
|  Spielfeld mit ECHTEM Spielbild)      +----------------------+
|                                       | RECHTS MITTE:        |
|                                       | neuronales Netz      |
|                                       +----------------------+
|                                       | RECHTS UNTEN:        |
|                                       | Statistiken + Verlauf|
+---------------------------------------+----------------------+

Technische Grundsätze:
- KEINE blockierende Endlosschleife: Simulation läuft über
  Tkinter.after(...) - ein Timer, der periodisch einen schnellen
  Update-Schritt aufruft.
- Simulation, Netz, Evolution und GUI sind getrennt: Diese Datei
  ZEICHNET nur Zustände, die von den Fachmodulen kommen.
- Spielbilder: Die NES-Frames (240x256x3) kommen direkt aus den
  Emulator-Instanzen und werden per PhotoImage skaliert angezeigt -
  ECHTE Spielszenen, keine Dekoration.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, filedialog, ttk

import numpy as np

from .actions import ALL_ACTIONS
from .episode import EpisodeStatus
from .network import NeuralNetwork
from .observation import observation_labels
from .population import Population
from .evolution import EvolutionConfig
from .experiment import Experiment

# Farbschema (konsistentes, dunkles Erscheinungsbild)
BG = "#1e1f24"
PANEL_BG = "#2a2b31"
FG = "#e8e8ea"
ACCENT = "#4fc3f7"
POS_WEIGHT = "#4fc3f7"   # positives Gewicht
NEG_WEIGHT = "#ef5350"   # negatives Gewicht


def frame_to_photo(frame: np.ndarray, scale: int, zoom_cache: dict) -> tk.PhotoImage:
    """Wandelt einen NES-Frame (240x256x3) in ein skaliertes tk.PhotoImage.

    Technische Details:
    - tkinter kann PNG-codierte Bilder direkt aus Base64-Daten erzeugen.
      Wir encodieren den Frame als unkomprimiertes PNG (keine externen
      Bibliotheken nötig: reine Standardbibliothek-Bytes).
    - Skalierung über PhotoImage.zoom (ganzzahlig, nearest neighbor) -
      für 8-Bit-Pixelgrafik die optisch sauberste Methode.
    - zoom_cache verhindert, dass pro Tick identische PNG-Bytes immer
      wieder neu encodiert werden (Leistungsschutz).
    """
    import struct, base64, zlib  # lokal: nur hier nötig

    h, w, _ = frame.shape
    # PNG-Bytes bauen: RGB8, keine Filter (filter type 0 je Zeile)
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # Filtertyp 0 (None)
        row = frame[y]
        raw.extend(row.tobytes())
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 1)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat)
           + chunk(b"IEND", b""))
    b64 = base64.b64encode(png).decode("ascii")
    img = tk.PhotoImage(data=b64)
    if scale > 1:
        img = img.zoom(scale, scale)
    return img


class MarioEvolutionApp:
    """Hauptfenster des Dashboards."""

    TICK_MS = 50          # Timer-Intervall
    STEPS_PER_TICK = 2    # Joypad-Schritte pro Tick (Emulator-Auslastung)

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Mario Neuroevolution Lab - gym-super-mario-bros")
        root.configure(bg=BG)
        root.geometry("1440x900")

        # Standardkonfiguration (im GUI änderbar)
        self.population_size = 8     # 8 Emulator-Instanzen parallel (siehe README)
        self.hidden_sizes = [12]
        self.max_steps = 1000
        self.rng = np.random.default_rng(42)
        self.evolution_config = EvolutionConfig(elite_count=2, mutation_rate=0.1,
                                                mutation_strength=0.3)

        self.experiment: Experiment | None = None
        self.simulation = None
        self.selected_index = 0
        self.running = False
        self._after_id: str | None = None
        self._photo_refs: list[tk.PhotoImage] = []  # tkinter braucht Referenzen!
        self._stats_labels: dict[str, tk.Label] = {}
        self._hist_points: list[tuple[int, float, float]] = []
        self._net_cache: dict = {"key": None, "items": None, "cw": 0, "ch": 0}

        self._build_layout()
        self._create_experiment()
        self._draw_all()
        self._tick()

    # ==================================================================
    # Layout
    # ==================================================================
    def _build_layout(self) -> None:
        header = tk.Frame(self.root, bg=PANEL_BG)
        header.pack(side=tk.TOP, fill=tk.X)

        tk.Label(header, text="Mario Neuroevolution Lab", bg=PANEL_BG, fg=ACCENT,
                 font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT, padx=10, pady=6)

        for text, cmd, width in (
            ("Start", self._on_start, 8),
            ("Pause", self._on_pause, 8),
            ("1 Schritt", self._on_single_step, 9),
            ("Neue Generation", self._on_new_generation, 14),
            ("Speichern", self._on_save, 9),
            ("Laden", self._on_load, 7),
        ):
            tk.Button(header, text=text, width=width, command=cmd
                      ).pack(side=tk.LEFT, padx=3, pady=6)

        tk.Label(header, text="Population:", bg=PANEL_BG, fg=FG).pack(side=tk.LEFT, padx=(12, 2))
        self.spin_pop = tk.Spinbox(header, from_=2, to=40, width=4,
                                   textvariable=tk.StringVar(value=str(self.population_size)))
        self.spin_pop.pack(side=tk.LEFT)
        tk.Label(header, text="Versteckte Neuronen:", bg=PANEL_BG, fg=FG).pack(side=tk.LEFT, padx=(12, 2))
        self.spin_hidden = tk.Spinbox(header, from_=1, to=64, width=4,
                                      textvariable=tk.StringVar(value=str(self.hidden_sizes[0])))
        self.spin_hidden.pack(side=tk.LEFT)

        self.lbl_gen = tk.Label(header, text="Generation 0", bg=PANEL_BG, fg=ACCENT,
                                font=("Segoe UI", 11, "bold"))
        self.lbl_gen.pack(side=tk.RIGHT, padx=12)

        main = tk.Frame(self.root, bg=BG)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # LINKS: Raster aller Instanzen
        left = tk.Frame(main, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(left, text="Alle Mario-Instanzen (echte Spielszenen)", bg=BG, fg=FG,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=2)
        self.canvas_all = tk.Canvas(left, bg=BG, highlightthickness=0)
        self.canvas_all.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # RECHTS: drei Panels
        right = tk.Frame(main, bg=BG, width=620)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        # RECHTS OBEN: Bester Lauf (kompakt mit Vorschau 128x120 und Daten)
        top_right = tk.LabelFrame(right, text="Bester Lauf", bg=PANEL_BG, fg=ACCENT)
        top_right.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        top_inner = tk.Frame(top_right, bg=PANEL_BG)
        top_inner.pack(fill=tk.X, padx=6, pady=4)

        self.label_best_img = tk.Label(top_inner, bg=PANEL_BG)
        self.label_best_img.pack(side=tk.LEFT, padx=(0, 8))

        self.lbl_best_info = tk.Label(top_inner, text="-", bg=PANEL_BG, fg=FG,
                                      font=("Consolas", 9), justify=tk.LEFT, anchor="nw")
        self.lbl_best_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # RECHTS UNTEN: Statistiken & Fitnessverlauf (unten angedockt)
        bottom_right = tk.LabelFrame(right, text="Statistiken & Verlauf", bg=PANEL_BG, fg=ACCENT)
        bottom_right.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=4)

        stats_frame = tk.Frame(bottom_right, bg=PANEL_BG)
        stats_frame.pack(fill=tk.X, padx=6, pady=2)
        stat_entries = [
            ("best_fitness", "Beste Fit:"),
            ("average_fitness", "Ø Fitness:"),
            ("worst_fitness", "Schlechteste:"),
            ("best_x_pos", "Beste x-Pos:"),
            ("success_rate", "Erfolgsquote:"),
            ("success_count", "Erfolge:"),
            ("average_steps", "Ø Schritte:"),
            ("blocked_steps", "Blockiert:"),
        ]
        self._stats_labels = {}
        for idx, (name, display_title) in enumerate(stat_entries):
            col_offset = 0 if idx < 4 else 2
            row = idx % 4
            tk.Label(stats_frame, text=display_title, bg=PANEL_BG, fg=FG,
                     font=("Consolas", 8), width=13, anchor="w").grid(
                column=col_offset, row=row, sticky="w", padx=(0, 2))
            lbl = tk.Label(stats_frame, text="-", bg=PANEL_BG, fg=ACCENT,
                           font=("Consolas", 8, "bold"), width=8, anchor="w")
            lbl.grid(column=col_offset + 1, row=row, sticky="w",
                     padx=(0, 10 if col_offset == 0 else 0))
            self._stats_labels[name] = lbl

        self.canvas_hist = tk.Canvas(bottom_right, bg=PANEL_BG, height=72, highlightthickness=0)
        self.canvas_hist.pack(fill=tk.X, padx=6, pady=3)

        # RECHTS MITTE: Neuronales Netz (erhaelt den maximalen Platz)
        mid_right = tk.LabelFrame(right, text="Neuronales Netz des ausgewählten Individuums",
                                  bg=PANEL_BG, fg=ACCENT)
        mid_right.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

        net_controls = tk.Frame(mid_right, bg=PANEL_BG)
        net_controls.pack(fill=tk.X, padx=6, pady=(2, 2))

        tk.Label(net_controls, text="Individuum:", bg=PANEL_BG, fg=FG,
                 font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
        self.combo_individual = ttk.Combobox(net_controls, state="readonly", width=8)
        self.combo_individual.pack(side=tk.LEFT, padx=4)
        self.combo_individual.bind("<<ComboboxSelected>>", self._on_select_individual)

        self.lbl_net_info = tk.Label(net_controls, text="-", bg=PANEL_BG, fg=FG,
                                     font=("Consolas", 9), justify=tk.LEFT, anchor="w")
        self.lbl_net_info.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)

        self.canvas_net = tk.Canvas(mid_right, bg=PANEL_BG, highlightthickness=0)
        self.canvas_net.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))
        self.canvas_net.bind("<Configure>", self._on_canvas_net_configure)

    # ==================================================================
    # Experiment-/Simulationssteuerung
    # ==================================================================
    def _create_experiment(self) -> None:
        """Erzeugt ein NEUES Experiment (Start/Reset).

        Emulator-Instanzen werden erst in start_round() erzeugt (siehe
        MultiMarioSimulation) - das Anlegen einer Population ist billig.
        """
        self.experiment = Experiment(
            population=Population.random(size=self.population_size,
                                         hidden_sizes=self.hidden_sizes,
                                         rng=self.rng),
            evolution_config=self.evolution_config,
            rng=self.rng,
            max_steps=self.max_steps,
        )
        self.simulation = None
        self.selected_index = 0
        self._hist_points = []
        self._net_cache = {"key": None, "items": None, "cw": 0, "ch": 0}
        self._refresh_individual_list()

    def _apply_gui_parameters(self, confirm: bool) -> None:
        """Übernimmt GUI-Parameter; Architekturwechsel erfordert Bestätigung."""
        try:
            pop_size = int(self.spin_pop.get())
            hidden = int(self.spin_hidden.get())
        except ValueError:
            messagebox.showerror("Ungültige Parameter",
                                 "Populationsgröße und Neuronenzahl müssen ganze Zahlen sein.")
            raise ValueError("ungültige GUI-Parameter")
        if pop_size < 2 or not (1 <= hidden <= 64):
            messagebox.showerror("Ungültige Parameter",
                                 "Populationsgröße >= 2, versteckte Neuronen 1..64.")
            raise ValueError("ungültige GUI-Parameter")

        needs_new = (
            self.experiment is None
            or pop_size != self.experiment.population.size
            or [hidden] != self.experiment.population.hidden_sizes
        )
        if needs_new and confirm and self.experiment is not None:
            if not messagebox.askyesno(
                    "Neue Population",
                    "Populationsgröße oder Netzarchitektur wurden geändert.\n"
                    "Es ist eine neue zufällige Population erforderlich.\n"
                    "Vorhandene Genome verwerfen?"):
                # Änderung NICHT anwenden - GUI auf alte Werte zurücksetzen.
                self.spin_pop.delete(0, tk.END)
                self.spin_pop.insert(0, str(self.experiment.population.size))
                self.spin_hidden.delete(0, tk.END)
                self.spin_hidden.insert(0, str(self.experiment.population.hidden_sizes[0]))
                return
        if needs_new:
            self.population_size = pop_size
            self.hidden_sizes = [hidden]
            if self.simulation is not None:
                self.simulation.close()
                self.simulation = None
            self._create_experiment()

    def _on_start(self) -> None:
        self.running = False  # kurz anhalten, um Parameter sicher anzuwenden
        self._apply_gui_parameters(confirm=True)
        if self.experiment is None:
            self._create_experiment()
        if self.simulation is None:
            self.simulation = self.experiment.start_round()
        self.running = True

    def _on_pause(self) -> None:
        self.running = False

    def _on_single_step(self) -> None:
        self.running = False
        self._apply_gui_parameters(confirm=True)
        if self.simulation is None:
            self.simulation = self.experiment.start_round()
        self.simulation.step_once()
        self._check_finished()
        self._draw_all()

    def _on_new_generation(self) -> None:
        if self.simulation is None:
            messagebox.showinfo("Neue Generation", "Zuerst eine Simulation laufen lassen.")
            return
        if not self.simulation.is_finished():
            messagebox.showwarning(
                "Neue Generation",
                "Die aktuelle Simulation ist noch nicht abgeschlossen.\n"
                "Alle Mario-Instanzen müssen fertig sein (Level geschafft, "
                "tot oder Schrittbudget aufgebraucht)."
            )
            return
        self.experiment.finish_round(self.simulation)
        self._hist_points = [(s.generation, s.best_fitness, s.average_fitness)
                             for s in self.experiment.history]
        self.experiment.advance_generation()
        self.simulation = None
        self._refresh_individual_list()
        self._draw_all()

    def _on_save(self) -> None:
        if self.experiment is None or self.experiment.best_overall_genome is None:
            messagebox.showinfo("Speichern", "Kein bewertetes Experiment zum Speichern.")
            return
        initial_dir = "checkpoints" if Path("checkpoints").is_dir() else "."
        default_name = f"mario_brain_gen{self.experiment.generation}.json"
        path = filedialog.asksaveasfilename(initialdir=initial_dir,
                                            initialfile=default_name,
                                            defaultextension=".json",
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self.experiment.save(path)
            messagebox.showinfo("Speichern", f"Experiment gespeichert:\n{path}")
        except OSError as exc:
            messagebox.showerror("Speichern fehlgeschlagen", str(exc))

    def _on_load(self) -> None:
        initial_dir = "checkpoints" if Path("checkpoints").is_dir() else "."
        path = filedialog.askopenfilename(initialdir=initial_dir,
                                          filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self.experiment = Experiment.load(
                path, evolution_config=self.evolution_config,
                rng=np.random.default_rng(42))
            self.population_size = self.experiment.population.size
            self.hidden_sizes = self.experiment.population.hidden_sizes
            self.spin_pop.delete(0, tk.END); self.spin_pop.insert(0, str(self.population_size))
            self.spin_hidden.delete(0, tk.END); self.spin_hidden.insert(0, str(self.hidden_sizes[0]))
            self.simulation = None
            self._hist_points = [(s.generation, s.best_fitness, s.average_fitness)
                                 for s in self.experiment.history]
            self._net_cache = {"key": None, "items": None, "cw": 0, "ch": 0}
            self._refresh_individual_list()
            self._draw_all()
            messagebox.showinfo("Laden", f"Experiment geladen:\n{path}")
        except (ValueError, OSError, KeyError) as exc:
            messagebox.showerror("Laden fehlgeschlagen",
                                 f"Die Datei konnte nicht geladen werden:\n{exc}")

    def _on_select_individual(self, _event=None) -> None:
        idx = self.combo_individual.current()
        if idx >= 0:
            self.selected_index = idx
            self._net_cache = {"key": None, "items": None, "cw": 0, "ch": 0}
            self._draw_network()

    def _refresh_individual_list(self) -> None:
        if self.experiment is None:
            return
        labels = [ind.label for ind in self.experiment.population.individuals]
        self.combo_individual["values"] = labels
        if 0 <= self.selected_index < len(labels):
            self.combo_individual.current(self.selected_index)

    def _check_finished(self) -> None:
        """Wenn alle Episoden fertig: bewerten (Stats/Rekord) und pausieren."""
        if (self.simulation is not None and self.simulation.is_finished()
                and self.experiment is not None):
            self.experiment.finish_round(self.simulation)
            self._hist_points = [(s.generation, s.best_fitness, s.average_fitness)
                                 for s in self.experiment.history]
            self._refresh_individual_list()
            self.running = False

    # ==================================================================
    # Timer-Loop (ereignisbasiert, nicht blockierend)
    # ==================================================================
    def _tick(self) -> None:
        if self.running and self.simulation is not None:
            self.simulation.step_many(self.STEPS_PER_TICK)
            self._check_finished()
        self._draw_all()
        self._after_id = self.root.after(self.TICK_MS, self._tick)

    def on_close(self) -> None:
        """Sauberes Beenden: Timer stoppen, Emulatoren freigeben, schließen."""
        self.running = False
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if self.simulation is not None:
            self.simulation.close()
        self.root.destroy()

    # ==================================================================
    # Zeichnen
    # ==================================================================
    def _draw_all(self) -> None:
        if self.experiment is None:
            return
        self.lbl_gen.config(text=f"Generation {self.experiment.generation}")
        self._draw_all_instances()
        self._draw_best()
        self._draw_network()
        self._draw_stats()

    def _draw_all_instances(self) -> None:
        """Raster aller Mario-Instanzen mit ECHTEN Spielszenen."""
        canvas = self.canvas_all
        canvas.delete("all")
        self._photo_refs = []
        if self.simulation is None:
            canvas.create_text(20, 20, text="Keine Simulation. 'Start' drücken.",
                               fill=FG, anchor="w", font=("Segoe UI", 11))
            return
        cw = canvas.winfo_width() or 800
        ch = canvas.winfo_height() or 600
        n = len(self.simulation.episodes)
        cols = max(1, int(np.ceil(np.sqrt(n * 2.0))))
        rows = max(1, int(np.ceil(n / cols)))
        cell_w = cw / cols
        cell_h = ch / rows
        # Skalierung so, dass ein NES-Frame (256x240) in die Zelle passt.
        scale = max(1, int(min(cell_w / 256.0, cell_h / 240.0)))
        img_w, img_h = 256 * scale, 240 * scale
        for idx, epi in enumerate(self.simulation.episodes):
            r, c = divmod(idx, cols)
            cx = int(c * cell_w + (cell_w - img_w) / 2)
            cy = int(r * cell_h + (cell_h - img_h) / 2)
            # Echter Frame aus dem Emulator dieser Instanz:
            photo = frame_to_photo(epi.last_frame, scale,
                                   zoom_cache={})  # Caching pro Tick reicht
            self._photo_refs.append(photo)  # Referenz halten (tkinter!)
            canvas.create_image(cx, cy, image=photo, anchor="nw")
            # Rahmen mit Statusfarbe: aktiv gelb / Erfolg grün / Ende rot
            color = {EpisodeStatus.ACTIVE: "#ffca28",
                     EpisodeStatus.SUCCESS: "#66bb6a",
                     EpisodeStatus.TERMINATED: "#ef5350"}[epi.status]
            canvas.create_rectangle(cx - 2, cy - 2, cx + img_w + 2, cy + img_h + 2,
                                    outline=color, width=2)
            # Label: eindeutige Zuordnung Instanz <-> Individuum + x-Position
            label = self.simulation.individuals[idx].label
            x_pos = int(epi.last_info.get("x_pos", 0))
            canvas.create_text(cx + 4, cy + 2, text=f"{label}  x={x_pos}",
                               fill=FG, anchor="nw", font=("Consolas", 8))

    def _on_canvas_net_configure(self, _event=None) -> None:
        """Wird aufgerufen, wenn sich die Größe des Netzwerk-Canvas ändert."""
        if self.experiment is None:
            return
        cw = self.canvas_net.winfo_width()
        ch = self.canvas_net.winfo_height()
        if cw > 10 and ch > 10:
            if cw != self._net_cache.get("cw") or ch != self._net_cache.get("ch"):
                self._draw_network()

    def _draw_best(self) -> None:
        """Detailansicht des aktuell besten Laufs + Rekord über alles."""
        best_idx, best_fit = None, None
        if self.simulation is not None:
            for i, epi in enumerate(self.simulation.episodes):
                f = epi.fitness()
                if best_fit is None or f > best_fit:
                    best_fit, best_idx = f, i
        record_txt = "-"
        if self.experiment is not None and self.experiment.best_overall_fitness is not None:
            record_txt = (f"{self.experiment.best_overall_label}  "
                          f"Fitness {self.experiment.best_overall_fitness:.1f}")

        if best_idx is None:
            info = f"Kein aktiver Lauf.\n\nRekord:\n{record_txt}"
            self.label_best_img.config(image="", text="")
            self._best_photo = None
        else:
            epi = self.simulation.episodes[best_idx]
            ind = self.simulation.individuals[best_idx]
            info = (f"Bester Lauf: {ind.label}\n"
                    f"Fitness:  {best_fit:.1f}\n"
                    f"x_pos:    {int(epi.last_info.get('x_pos', 0))}\n"
                    f"Schritte: {epi.steps} (block: {epi.blocked_steps})\n"
                    f"Status:   {epi.status}\n\n"
                    f"Rekord:\n{record_txt}")
            # Subsample 2 (128x120), damit das neuronale Netz in der Mitte maximalen Platz erhält
            photo = frame_to_photo(epi.last_frame, 1, {}).subsample(2, 2)
            self._best_photo = photo  # Referenz halten
            self.label_best_img.config(image=photo)
        self.lbl_best_info.config(text=info)

    # ------------------------------------------------------------------
    def _draw_network(self) -> None:
        """Zeichnet die TATSÄCHLICHE Netzarchitektur des gewählten Individuums."""
        canvas = self.canvas_net
        if self.experiment is None:
            return
        inds = self.experiment.population.individuals
        if not inds:
            return
        idx = min(self.selected_index, len(inds) - 1)
        ind = inds[idx]
        net = ind.network

        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 10 or ch <= 10:
            cw = 600
            ch = 280

        key = (ind.label, net.describe(), cw, ch)
        if self._net_cache.get("key") != key:
            canvas.delete("all")
            self._net_cache["items"] = self._create_network_items(canvas, net, cw, ch)
            self._net_cache["key"] = key
            self._net_cache["cw"] = cw
            self._net_cache["ch"] = ch
            self.lbl_net_info.config(
                text=f"{ind.label} | {net.describe()} | {net.genome_size()} Parameter"
            )
        self._update_network_activations(canvas, net)

    def _create_network_items(self, canvas: tk.Canvas, net: NeuralNetwork, cw: int, ch: int) -> dict:
        """Erzeugt Struktur-Objekte (Neuronen, gewichtsfarbene Verbindungen)."""
        layers = net.layer_sizes
        n_layers = len(layers)

        margin_left = 75    # Platz für Eingabe-Labels links
        margin_right = 145  # Platz für Aktionsnamen rechts
        margin_top = 28     # Platz für Schichttitel oben
        margin_bottom = 26  # Platz für Legende unten

        avail_w = max(100, cw - margin_left - margin_right)
        avail_h = max(80, ch - margin_top - margin_bottom)

        xs = [margin_left + i * (avail_w / (n_layers - 1)) for i in range(n_layers)]
        positions = []

        max_size = max(layers)
        step_y = avail_h / max(1, max_size - 1)
        # Dynamischer Knotenradius: passt sich automatisch der Neuronenzahl und Canvas-Höhe an
        r = max(5, min(10, int(step_y * 0.38)))

        for li, size in enumerate(layers):
            if size == 1:
                ys = [margin_top + avail_h / 2]
            else:
                ys = [margin_top + ni * (avail_h / (size - 1)) for ni in range(size)]
            positions.append([(xs[li], y) for y in ys])

        # Schicht-Überschriften oben
        canvas.create_text(xs[0], 12, text=f"Eingaben ({layers[0]})",
                           fill=ACCENT, anchor="center", font=("Segoe UI", 9, "bold"))
        for k in range(1, n_layers - 1):
            canvas.create_text(xs[k], 12, text=f"Hidden ({layers[k]})",
                               fill=ACCENT, anchor="center", font=("Segoe UI", 9, "bold"))
        canvas.create_text(xs[-1], 12, text=f"Aktionen ({layers[-1]})",
                           fill=ACCENT, anchor="center", font=("Segoe UI", 9, "bold"))

        # Verbindungen zeichnen
        max_lines = 500
        line_count, reduced = 0, False
        for k in range(n_layers - 1):
            w_mat = net.weights[k]
            for a in range(layers[k]):
                for b in range(layers[k + 1]):
                    if line_count >= max_lines:
                        reduced = True
                        break
                    w = float(w_mat[b, a])
                    color = POS_WEIGHT if w >= 0 else NEG_WEIGHT
                    width = max(1, min(4, int(abs(w) * 2.2) + 1))
                    canvas.create_line(positions[k][a][0], positions[k][a][1],
                                       positions[k + 1][b][0], positions[k + 1][b][1],
                                       fill=color, width=width, tags=("edge",))
                    line_count += 1

        # Knoten zeichnen
        node_ids = []
        for li in range(n_layers):
            ids = []
            for ni in range(layers[li]):
                x, y = positions[li][ni]
                node_r = r + 1 if li in (0, n_layers - 1) else r
                ids.append(canvas.create_oval(
                    x - node_r, y - node_r, x + node_r, y + node_r,
                    fill="#44475a", outline="#191a1f", width=1.5, tags=("node",)
                ))
            node_ids.append(ids)

        # Eingabe-Beschriftungen links
        input_labels = [
            "dx Ziel", "x_pos", "höhe", "fällt", "zeit", "tot",
            "flagge", "hind_l", "hind_r", "alt:rechts", "alt:r_sp", "alt:spr", "alt:links"
        ]
        if layers[0] == len(input_labels):
            for ni, name in enumerate(input_labels):
                x, y = positions[0][ni]
                canvas.create_text(x - (r + 7), y, text=name, fill=FG, anchor="e",
                                   font=("Segoe UI", 8))
        else:
            for ni in range(layers[0]):
                x, y = positions[0][ni]
                canvas.create_text(x - (r + 7), y, text=f"in_{ni}", fill=FG, anchor="e",
                                   font=("Segoe UI", 8))

        # Aktions-Namen an Ausgabeneuronen rechts
        for ni, name in enumerate(ALL_ACTIONS[:layers[-1]]):
            x, y = positions[-1][ni]
            canvas.create_text(x + (r + 8), y, text=name, fill=FG, anchor="w",
                               font=("Segoe UI", 8))

        # Legende unten
        ly = ch - 12
        canvas.create_line(10, ly, 30, ly, fill=POS_WEIGHT, width=2)
        canvas.create_text(34, ly, text="positives Gewicht", fill=FG, anchor="w",
                           font=("Segoe UI", 8))
        canvas.create_line(150, ly, 170, ly, fill=NEG_WEIGHT, width=2)
        canvas.create_text(174, ly, text="negatives Gewicht", fill=FG, anchor="w",
                           font=("Segoe UI", 8))
        canvas.create_oval(290 - 4, ly - 4, 290 + 4, ly + 4, fill="#76ff03", outline="#ffffff")
        canvas.create_text(298, ly, text="gewählte Aktion", fill=FG, anchor="w",
                           font=("Segoe UI", 8))
        if reduced:
            canvas.create_text(cw - 10, ly, text="(Kanten auf 500 begrenzt)",
                               fill="#ffab91", anchor="e", font=("Segoe UI", 8))

        return {"positions": positions, "node_ids": node_ids, "r": r}

    def _update_network_activations(self, canvas: tk.Canvas, net: NeuralNetwork) -> None:
        """Aktualisiert Knotenfarben mit den AKTUELLEN Aktivierungen.

        Quelle: Beobachtung der Episode des gewählten Individuums (falls
        vorhanden) - echte, laufende Werte statt Dekoration.
        """
        items = self._net_cache.get("items")
        if not items:
            return
        node_ids = items["node_ids"]
        obs = None
        if (self.simulation is not None
                and self.selected_index < len(self.simulation.episodes)):
            epi = self.simulation.episodes[self.selected_index]
            obs = epi.observation()

        if obs is None:
            for layer_nodes in node_ids:
                for nid in layer_nodes:
                    canvas.itemconfig(nid, fill="#44475a", outline="#191a1f", width=1.5)
            return

        activations = [obs]
        a = obs
        n_layers = len(net.layer_sizes)
        for k in range(n_layers - 1):
            z = net.weights[k] @ a + net.biases[k]
            a = net.output_activation(z) if k == n_layers - 2 else net.hidden_activation(z)
            activations.append(a)

        # 1. Eingaben: blaues Leuchten je nach Wert
        for i, nid in enumerate(node_ids[0]):
            v = float(np.clip(activations[0][i], 0.0, 1.0))
            c = int(60 + 195 * v)
            canvas.itemconfig(nid, fill=f"#{c:02x}{c:02x}ff", outline="#191a1f", width=1.5)

        # 2. Versteckte Schichten: Cyan/Grün je nach ReLU-Aktivierung
        for li in range(1, n_layers - 1):
            h_act = activations[li]
            h_max = max(1.0, float(np.max(h_act))) if h_act.size > 0 else 1.0
            for i, nid in enumerate(node_ids[li]):
                v = float(np.clip(h_act[i] / h_max, 0.0, 1.0))
                g = int(60 + 195 * v)
                canvas.itemconfig(nid, fill=f"#2a{g:02x}a0", outline="#191a1f", width=1.5)

        # 3. Ausgaben: gewählte Aktion leuchtet hellgrün mit weißem Rand
        chosen_action = int(np.argmax(activations[-1]))
        for i, nid in enumerate(node_ids[-1]):
            val = float(activations[-1][i])
            if i == chosen_action:
                canvas.itemconfig(nid, fill="#76ff03", outline="#ffffff", width=2.5)
            else:
                color = POS_WEIGHT if val >= 0 else NEG_WEIGHT
                canvas.itemconfig(nid, fill=color, outline="#191a1f", width=1.5)

    # ------------------------------------------------------------------
    def _draw_stats(self) -> None:
        """Statistikbereich + Fitnessverlauf aktualisieren."""
        stats = self.experiment.history[-1] if (self.experiment and self.experiment.history) else None
        values = {
            "best_fitness": f"{stats.best_fitness:.1f}" if stats else "-",
            "average_fitness": f"{stats.average_fitness:.1f}" if stats else "-",
            "worst_fitness": f"{stats.worst_fitness:.1f}" if stats else "-",
            "success_rate": f"{stats.success_rate:.0%}" if stats else "-",
            "success_count": str(stats.success_count) if stats else "-",
            "average_steps": f"{stats.average_steps:.1f}" if stats else "-",
            "blocked_steps": str(stats.total_blocked_steps) if stats else "-",
            "best_x_pos": f"{stats.best_x_pos:.0f}" if stats else "-",
        }
        for key, text in values.items():
            self._stats_labels[key].config(text=text)

        canvas = self.canvas_hist
        canvas.delete("all")
        if len(self._hist_points) < 1:
            return
        cw = canvas.winfo_width() or 500
        ch = canvas.winfo_height() or 100
        gens = [p[0] for p in self._hist_points]
        bests = [p[1] for p in self._hist_points]
        avgs = [p[2] for p in self._hist_points]
        lo = min(min(bests), min(avgs), 0)
        hi = max(max(bests), max(avgs), 1)
        def y(v): return ch - 15 - (v - lo) / (hi - lo + 1e-9) * (ch - 25)
        def x(g): return 10 + g / max(1, gens[-1]) * (cw - 20)
        for series, color in ((bests, "#66bb6a"), (avgs, ACCENT)):
            pts = []
            for g, v in zip(gens, series):
                pts += [x(g), y(v)]
            if len(pts) >= 4:
                canvas.create_line(*pts, fill=color, width=2, smooth=True)
        canvas.create_text(10, 8, text=f"beste Fitness (grün) / Ø ({ACCENT})",
                           fill=FG, anchor="w", font=("Segoe UI", 8))


def main() -> None:
    """Startet das Dashboard (Einstiegspunkt)."""
    root = tk.Tk()
    app = MarioEvolutionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
