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
        self._net_cache: dict = {"label": None, "items": None}

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
        right = tk.Frame(main, bg=BG, width=580)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        top_right = tk.LabelFrame(right, text="Bester Lauf", bg=PANEL_BG, fg=ACCENT)
        top_right.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.lbl_best_info = tk.Label(top_right, text="-", bg=PANEL_BG, fg=FG,
                                      font=("Consolas", 10), justify=tk.LEFT, anchor="w")
        self.lbl_best_info.pack(fill=tk.X, padx=6, pady=2)
        self.label_best_img = tk.Label(top_right, bg=PANEL_BG)
        self.label_best_img.pack(padx=6, pady=4)

        mid_right = tk.LabelFrame(right, text="Neuronales Netz des ausgewählten Individuums",
                                  bg=PANEL_BG, fg=ACCENT)
        mid_right.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.lbl_net_info = tk.Label(mid_right, text="-", bg=PANEL_BG, fg=FG,
                                     font=("Consolas", 9), justify=tk.LEFT, anchor="w")
        self.lbl_net_info.pack(fill=tk.X, padx=6)
        self.canvas_net = tk.Canvas(mid_right, bg=PANEL_BG, highlightthickness=0)
        self.canvas_net.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        bottom_right = tk.LabelFrame(right, text="Statistiken", bg=PANEL_BG, fg=ACCENT)
        bottom_right.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        stats_frame = tk.Frame(bottom_right, bg=PANEL_BG)
        stats_frame.pack(fill=tk.X, padx=6, pady=2)
        for name in ("best_fitness", "average_fitness", "worst_fitness",
                     "success_rate", "success_count", "average_steps",
                     "blocked_steps", "best_x_pos"):
            tk.Label(stats_frame, text=name, bg=PANEL_BG, fg=FG,
                     font=("Consolas", 9), width=18, anchor="w").grid(
                column=0, row=len(self._stats_labels), sticky="w")
            lbl = tk.Label(stats_frame, text="-", bg=PANEL_BG, fg=ACCENT,
                           font=("Consolas", 9), anchor="w")
            lbl.grid(column=1, row=len(self._stats_labels), sticky="w")
            self._stats_labels[name] = lbl
        self.canvas_hist = tk.Canvas(bottom_right, bg=PANEL_BG, height=110, highlightthickness=0)
        self.canvas_hist.pack(fill=tk.X, padx=6, pady=4)
        sel_frame = tk.Frame(bottom_right, bg=PANEL_BG)
        sel_frame.pack(fill=tk.X, padx=6, pady=2)
        tk.Label(sel_frame, text="Individuum:", bg=PANEL_BG, fg=FG).pack(side=tk.LEFT)
        self.combo_individual = ttk.Combobox(sel_frame, state="readonly", width=20)
        self.combo_individual.pack(side=tk.LEFT, padx=4)
        self.combo_individual.bind("<<ComboboxSelected>>", self._on_select_individual)

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
        self._net_cache = {"label": None, "items": None}
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
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self.experiment.save(path)
            messagebox.showinfo("Speichern", f"Experiment gespeichert:\n{path}")
        except OSError as exc:
            messagebox.showerror("Speichern fehlgeschlagen", str(exc))

    def _on_load(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
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
            self._net_cache = {"label": None, "items": None}
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
            info = f"Kein aktiver Lauf.\nRekord: {record_txt}"
            self.label_best_img.config(image="", text="")
            self._best_photo = None
        else:
            epi = self.simulation.episodes[best_idx]
            ind = self.simulation.individuals[best_idx]
            info = (f"Aktuell bester Lauf: {ind.label}  Fitness {best_fit:.1f}\n"
                    f"x_pos: {int(epi.last_info['x_pos'])}  Schritte: {epi.steps}  "
                    f"blockiert: {epi.blocked_steps}  Status: {epi.status}\n"
                    f"Rekord über alle Generationen: {record_txt}")
            scale = 2  # 512x480 ist gut lesbar im rechten Panel
            photo = frame_to_photo(epi.last_frame, scale, {})
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
        key = (ind.label, net.describe())
        if self._net_cache["label"] != key:
            canvas.delete("all")
            self._net_cache["items"] = self._create_network_items(canvas, net)
            self._net_cache["label"] = key
            self.lbl_net_info.config(
                text=f"{ind.label}  Architektur: {net.describe()}  "
                     f"Parameter: {net.genome_size()}"
            )
        self._update_network_activations(canvas, net)

    def _create_network_items(self, canvas: tk.Canvas, net: NeuralNetwork) -> dict:
        """Erzeugt Struktur-Objekte (Neuronen, gewichtsfarbene Verbindungen)."""
        cw = canvas.winfo_width() or 540
        ch = canvas.winfo_height() or 240
        layers = net.layer_sizes
        margin_x = 40
        xs = [margin_x + i * (cw - 2 * margin_x) / (len(layers) - 1) for i in range(len(layers))]
        positions = []
        for li, size in enumerate(layers):
            if size == 1:
                ys = [ch / 2]
            else:
                ys = [15 + ni * (ch - 45) / (size - 1) for ni in range(size)]
            positions.append([(xs[li], y) for y in ys])

        # Verbindungen: bei > 400 Kanten reduzierte Darstellung (klar
        # gekennzeichnet, damit niemand Vollständigkeit annimmt).
        max_lines = 400
        line_count, reduced = 0, False
        for k in range(len(layers) - 1):
            for a in range(layers[k]):
                for b in range(layers[k + 1]):
                    if line_count >= max_lines:
                        reduced = True
                        break
                    w = net.weights[k][b, a]
                    color = POS_WEIGHT if w >= 0 else NEG_WEIGHT
                    width = max(1, min(3, abs(w) * 2 + 0.5))
                    canvas.create_line(positions[k][a][0], positions[k][a][1],
                                       positions[k + 1][b][0], positions[k + 1][b][1],
                                       fill=color, width=width, tags=("edge",))
                    line_count += 1
        node_ids = []
        for li in range(len(layers)):
            ids = []
            for ni in range(layers[li]):
                x, y = positions[li][ni]
                r = 7 if li in (0, len(layers) - 1) else 6
                ids.append(canvas.create_oval(x - r, y - r, x + r, y + r,
                                              fill="#555", outline="#000", tags=("node",)))
            node_ids.append(ids)

        # Legende
        ly = ch - 12
        canvas.create_line(10, ly, 30, ly, fill=POS_WEIGHT, width=2)
        canvas.create_text(34, ly, text="positives Gewicht", fill=FG, anchor="w",
                           font=("Segoe UI", 8))
        canvas.create_line(160, ly, 180, ly, fill=NEG_WEIGHT, width=2)
        canvas.create_text(184, ly, text="negatives Gewicht", fill=FG, anchor="w",
                           font=("Segoe UI", 8))
        if reduced:
            canvas.create_text(cw - 10, ly, text="(reduziert: >400 Verbindungen)",
                               fill="#ffab91", anchor="e", font=("Segoe UI", 8))
        # Aktion-Namen an die Ausgabeneuronen schreiben
        for ni, name in enumerate(ALL_ACTIONS):
            x, y = positions[-1][ni]
            canvas.create_text(x + 12, y, text=name, fill=FG, anchor="w",
                               font=("Segoe UI", 7))
        return {"positions": positions, "node_ids": node_ids}

    def _update_network_activations(self, canvas: tk.Canvas, net: NeuralNetwork) -> None:
        """Aktualisiert Knotenfarben mit den AKTUELLEN Aktivierungen.

        Quelle: Beobachtung der Episode des gewählten Individuums (falls
        vorhanden) - echte, laufende Werte statt Dekoration.
        """
        items = self._net_cache["items"]
        if not items:
            return
        obs = out = None
        if (self.simulation is not None
                and self.selected_index < len(self.simulation.episodes)):
            epi = self.simulation.episodes[self.selected_index]
            obs = epi.observation()
            out = net.forward(obs)
        node_ids = items["node_ids"]
        if obs is not None:
            for i, nid in enumerate(node_ids[0]):
                v = float(np.clip(obs[i], 0.0, 1.0))
                c = int(80 + 175 * v)
                canvas.itemconfig(nid, fill=f"#{c:02x}{c:02x}ff")
        if out is not None:
            for i, nid in enumerate(node_ids[-1]):
                canvas.itemconfig(nid, fill=POS_WEIGHT if float(out[i]) >= 0 else NEG_WEIGHT)

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
