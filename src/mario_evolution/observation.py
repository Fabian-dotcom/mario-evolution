"""Beobachtungsschnittstelle: gym-super-mario-bros-Zustand -> Eingabevektor.

Das neuronale Netz bekommt einen 1D-Vektor aus Gleitkommazahlen. Diese
Datei definiert VERBINDLICH, wie der Vektor aus dem Spielzustand
(NES-Bild + `info`-Daten + Vor-Schritt-Zustand) aufgebaut ist.

Sensor-Design (17 Eingaben, alle in [0, 1]):

  Index 0:  reststrecke_x    - horizontale Reststrecke bis zum Level-Ende,
                               normalisiert mit 1/4800.
  Index 1:  x_position       - absolute Position, normalisiert mit 1/4800.
  Index 2:  geschwindigkeit  - horizontale Geschwindigkeit: (x - x_vor) / 8,
                               geklemmt auf [0,1]. WICHTIG fuer Spruenge
                               mit Anlauf: Ohne Geschwindigkeit kann das
                               Netz "laufe ich schon schnell?" nicht von
                               "stehe ich noch?" unterscheiden.
  Index 3:  hoehe            - Hoehe ueber dem Boden, normalisiert 1/180.
  Index 4:  sprungphase      - 0.0 am Boden, 0.5 steigend, 1.0 fallend.
                               Damit kann das Netz "ich bin im Aufstieg"
                               von "ich lande gleich" unterscheiden.
  Index 5:  faellt           - 1.0, wenn y_pixel gegenueber Vor-Schritt waechst.
  Index 6:  restzeit         - Restzeit / 400.
  Index 7:  tot              - 1.0, wenn Mario tot ist.
  Index 8:  flagge           - 1.0, wenn die Flagge erreicht wurde.
  Index 9:  hindernis_links  - solides Hindernis/Gegner im linken Streifen.
  Index 10: hindernis_rechts - solides Hindernis/Gegner im rechten Streifen.
  Index 11: distanz_nahe     - Anteil solider Flaeche im NAHEN Streifen
                               rechts (kontinuierlich in [0,1]): je hoeher,
                               desto NAEHER ist das Hindernis.
  Index 12: distanz_fern     - Anteil solider Flaeche im FERNEN Streifen
                               (weitere 32 px): 1 = Hindernis kommt,
                               0 = freie Bahn. Zusammen mit Index 11 kann
                               das Netz die ENTFERNUNG abschaetzen und
                               frueh genug den Anlauf-Sprung starten.
  Index 13-16: One-Hot der letzten NAVIGATIONS-Aktion (4 Spalten:
                               rechts / rechts_sprung / springen / links).

One-Hot-Korrektur (Bugfix): Die Doku beschrieb die Aktionen als
"rechts / rechts_sprung / springen / links", der Code nutzte aber die
ERSTEN VIER Indizes (halten/rechts/rechts_sprung/rennen_rechts). Jetzt
wird explizit das dokumentierte Mapping [1, 2, 5, 6] verwendet - Code
und Legende stimmen wieder ueberein (dafuer gibt es einen Test).

Warum Pixel-Sensorik?
Der NES-Himmel von Level 1-1 hat exakt die Farbe (104, 136, 252).
Mario wird im Bild ueber seine rote Kleidung (248,56,0) gefunden (der
Viewport scrollt mit!). Rohre/Bloecke werden ueber den solide-Anteil
erkannt (kalibriert: Busch ~0.07, Rohr ~0.30), Gegner (Goombas) ueber
deren Fellfarbe (240,208,176), die sonst nirgends vorkommt.

Bezug zu "The Nature of Code":
Kapitel 10 beschreibt Perzeptron-Eingaben x_i, die mit Gewichten w_i
multipliziert und summiert werden. Hier liefern wir genau diese x_i:
die "Sensorik" des Agenten. Die Wahl der Eingaben legt fest, WELCHE
Information das Netz ueberhaupt sehen kann.
"""

from __future__ import annotations

import numpy as np

from .actions import ACTION_COUNT

# ---------------------------------------------------------------------
# Kalibrierte Farben (an echten Frames von SuperMarioBros-1-1-v0)
# ---------------------------------------------------------------------
SKY_RGB: tuple[int, int, int] = (104, 136, 252)    # Himmel
GROUND_RGB: tuple[int, int, int] = (228, 92, 16)   # Boden-Ziegel
MARIO_RGB: tuple[int, int, int] = (248, 56, 0)     # Marios rote Kleidung
MARIO_RGB_TOLERANCE: int = 10
ENEMY_RGB: tuple[int, int, int] = (240, 208, 176)  # Goomba-Fell

# Schwelle fuer "solides Hindernis" im Streifen (kalibriert an echten
# Frames: Hintergrundbusch ~0.07 nicht-himmelfarben, Rohr ~0.30 ->
# Schwelle 0.20 trennt beides).
OBSTACLE_MIN_FRACTION: float = 0.20

# ---------------------------------------------------------------------
# Normalisierungen (aus dem Spiel abgelesen, dokumentiert statt geraten)
# ---------------------------------------------------------------------
LEVEL_WIDTH_PX: float = 4800.0     # horizontale Breite eines Levels
MAX_FALL_HEIGHT_PX: float = 180.0  # sinnvoller Vertikalbereich
MAX_TIME: float = 400.0            # Startzeit eines Levels
GROUND_Y_PIXEL: float = 193.0      # y_pixel von Mario am Boden (1-1)
# Geschwindigkeits-Skalierung: bei Frameskip 4 sind realistisch bis zu
# ~8 Pixel pro Joypad-Schritt erreichbar (rennen). /8 klemmt die
# maximale Laufgeschwindigkeit auf 1.0.
SPEED_SCALE_PX: float = 8.0
# Fallback-Bildschirm-x, falls Mario nicht gefunden wird.
DEFAULT_MARIO_SCREEN_X: int = 47

# Groesse des Beobachtungsvektors. Das Netz prueft die Laenge zur Laufzeit.
OBSERVATION_SIZE: int = 17

# One-Hot-Mapping: die vier NAVIGATIONS-Aktionen und ihre Indizes aus
# actions.py (Bugfix: vorher wurden die ersten vier Indizes benutzt,
# was zur Dokumentation "rechts/rechts_sprung/springen/links" nicht passte).
ONE_HOT_ACTION_INDICES: tuple[int, int, int, int] = (1, 2, 5, 6)

NO_PREVIOUS_ACTION: int = -1

# Schwelle fuer "am Boden" (<= 2 px Hoehe ueber der Bodenlinie).
GROUND_THRESHOLD_PX: float = 2.0


def _find_mario_screen_x(frame: np.ndarray) -> int:
    """Findet Marios Bildschirm-x ueber seine roten Kleidungspixel.

    Warum suchen statt fixe Position? Der NES-Viewport scrollt mit:
    Mario "wandert" von screen-x 47 (Levelanfang) zur Bildmitte (~120).
    Eine feste Position wuerde falsche Hindernis-Regionen lesen.

    Rueckgabe: Mittelwert der roten Pixel-x (int). Fallback:
    DEFAULT_MARIO_SCREEN_X, wenn keine Pixel gefunden werden.
    """
    r, g, b = MARIO_RGB
    mask = ((np.abs(frame[:, :, 0].astype(int) - r) <= MARIO_RGB_TOLERANCE)
            & (np.abs(frame[:, :, 1].astype(int) - g) <= MARIO_RGB_TOLERANCE)
            & (frame[:, :, 2] <= 32))
    xs = np.nonzero(mask)[1]
    if xs.size == 0:
        return DEFAULT_MARIO_SCREEN_X
    return int(xs.mean())


def _is_solid(region: np.ndarray) -> bool:
    """True, wenn die Region ein solides Hindernis oder einen Gegner enthaelt.

    Definition "solide": Kein Himmel, kein Boden, UND der nicht-freie
    Anteil >= OBSTACLE_MIN_FRACTION. Das unterscheidet Rohre/Bloecke
    (voller Block) von Hintergrundbueschen (gleiche Farbe, aber mit
    Himmel-Luecken). Zusaetzlich gilt die Region als gefaehrlich, wenn
    Goomba-Pixel enthalten sind.
    """
    sky = np.array(SKY_RGB, dtype=np.uint8)
    ground = np.array(GROUND_RGB, dtype=np.uint8)
    is_sky = np.all(region == sky, axis=-1)
    is_ground = np.all(region == ground, axis=-1)
    passable = is_sky | is_ground
    solid_fraction = 1.0 - float(passable.mean())
    enemy = np.all(np.abs(region.astype(int) - np.array(ENEMY_RGB)) <= 10, axis=-1)
    return solid_fraction >= OBSTACLE_MIN_FRACTION or bool(enemy.any())


def _solid_fraction(region: np.ndarray) -> float:
    """Anteil der nicht-himmel-/nicht-bodenfarben Flaeche (0..1).

    Wird fuer die KONTINUIERLICHE Distanz-Sensoren benutzt (Indizes
    11/12): Buesche verfaelschen den Wert leicht, aber der Hauptin-
    formationstransfer ("etwas Massives kommt naeher") bleibt erhalten.
    """
    sky = np.array(SKY_RGB, dtype=np.uint8)
    ground = np.array(GROUND_RGB, dtype=np.uint8)
    is_sky = np.all(region == sky, axis=-1)
    is_ground = np.all(region == ground, axis=-1)
    passable = is_sky | is_ground
    return float(1.0 - passable.mean())


def _sample_region(frame: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Schneidet eine Bildregion heraus (Clipping an den Raendern)."""
    h, w, _ = frame.shape
    return frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1), :]


def encode(frame: np.ndarray,
           info: dict,
           previous_action_index: int = NO_PREVIOUS_ACTION,
           previous_y_pixel: float | None = None,
           previous_x_pos: float | None = None) -> np.ndarray:
    """Erzeugt den numerischen Eingabevektor fuer das neuronale Netz.

    Argumente:
        frame: NES-Bild (240, 256, 3), uint8 - von env.reset()/env.step().
        info: gymnasium-Info-Dictionary (x_pos, y_pixel, time, ...).
        previous_action_index: Index der letzten Aktion (0..6) oder
            NO_PREVIOUS_ACTION am Episodenanfang.
        previous_y_pixel: y_pixel des VORHERIGEN Schritts (fuer Fall-
            und Sprungphasen-Erkennung). None am Anfang.
        previous_x_pos: x_pos des VORHERIGEN Schritts (fuer die
            Geschwindigkeitsmessung). None am Anfang -> Speed 0.

    Rueckgabe:
        np.ndarray mit OBSERVATION_SIZE Eintraegen, alle in [0, 1].
    """
    if getattr(frame, "shape", None) != (240, 256, 3):
        raise ValueError(
            f"encode: frame muss die Form (240, 256, 3) haben, erhielt "
            f"{None if frame is None else frame.shape}"
        )
    if frame.dtype != np.uint8:
        raise ValueError(f"encode: frame muss uint8 sein, erhielt {frame.dtype}")
    required = ("x_pos", "y_pixel", "time")
    missing = [k for k in required if k not in info]
    if missing:
        raise ValueError(f"encode: im info-Dictionary fehlen Schluessel {missing}")
    if not isinstance(previous_action_index, int):
        raise TypeError(
            f"encode: previous_action_index muss int sein, "
            f"erhielt {type(previous_action_index).__name__}"
        )
    if not (NO_PREVIOUS_ACTION <= previous_action_index < ACTION_COUNT):
        raise ValueError(
            f"encode: previous_action_index muss in [-1, {ACTION_COUNT - 1}] liegen, "
            f"erhielt {previous_action_index}"
        )

    values: list[float] = []

    # --- Block A: horizontale Lage und Bewegung -------------------------
    x_pos = float(info["x_pos"])
    values.append(min(1.0, max(0.0, (LEVEL_WIDTH_PX - x_pos) / LEVEL_WIDTH_PX)))  # 0
    values.append(min(1.0, max(0.0, x_pos / LEVEL_WIDTH_PX)))                     # 1
    # Geschwindigkeit: Differenz zum Vor-Schritt, skaliert und geklemmt.
    if previous_x_pos is not None:
        speed = max(0.0, min(1.0, (x_pos - float(previous_x_pos)) / SPEED_SCALE_PX))
    else:
        speed = 0.0
    values.append(speed)                                                          # 2

    # --- Block B: vertikale Lage und Sprungphase -------------------------
    y_pixel = float(info["y_pixel"])
    height = max(0.0, GROUND_Y_PIXEL - y_pixel)
    values.append(min(1.0, height / MAX_FALL_HEIGHT_PX))                          # 3
    # Sprungphase: am Boden (0.0), steigend/Scheitel (0.5), fallend (1.0).
    falling = previous_y_pixel is not None and y_pixel > previous_y_pixel
    rising = previous_y_pixel is not None and y_pixel < previous_y_pixel
    if height <= GROUND_THRESHOLD_PX:
        jump_phase = 0.0
    elif falling:
        jump_phase = 1.0
    else:
        # steigend oder Scheitelpunkt (in der Luft, konstant)
        jump_phase = 0.5
    values.append(jump_phase)                                                     # 4
    values.append(1.0 if falling else 0.0)                                        # 5

    # --- Block C: Spielstatus -------------------------------------------
    values.append(max(0.0, min(1.0, float(info["time"]) / MAX_TIME)))             # 6
    values.append(1.0 if info.get("is_dead") else 0.0)                            # 7
    values.append(1.0 if info.get("flag_get") else 0.0)                           # 8

    # --- Block D: Pixel-Sensorik (Hindernis/Gegner/Entfernung) -----------
    # y_pixel aus den info-Daten ist die OBERKANTE von Mario; die Fueße
    # liegen ~30 Pixel tiefer (verifiziert: Boden y_pixel=176 -> Fueße
    # bei screen-y 206). Der Suchstreifen liegt 8 Pixel UEBER den Fueßen
    # (Boden ausgeschlossen) bis 78 Pixel darueber (Sprunghoehe).
    # NEU: ZWEI Streifen je Seite - NAH (8..40 px) und FERN (40..72 px).
    # So kann das Netz die Entfernung abschaetzen und frueh genug den
    # Anlauf-Sprung starten (Kernproblem "steht am grossen Hindernis").
    foot_screen_y = int(y_pixel) + 30
    top_y = max(0, foot_screen_y - 78)
    bottom_y = max(0, foot_screen_y - 8)
    mx = _find_mario_screen_x(frame)
    left_near = _sample_region(frame, top_y, bottom_y, mx - 40, mx - 8)
    right_near = _sample_region(frame, top_y, bottom_y, mx + 8, mx + 40)
    right_far = _sample_region(frame, top_y, bottom_y, mx + 40, mx + 72)
    values.append(1.0 if _is_solid(left_near) else 0.0)                           # 9
    values.append(1.0 if _is_solid(right_near) else 0.0)                          # 10
    values.append(_solid_fraction(right_near))                                    # 11
    values.append(_solid_fraction(right_far))                                     # 12

    # --- Block E: One-Hot der letzten NAVIGATIONS-Aktion (Bugfix) --------
    # Explizites Mapping der dokumentierten Indizes -> 4 Spalten.
    for mapped_index in ONE_HOT_ACTION_INDICES:
        values.append(1.0 if mapped_index == previous_action_index else 0.0)      # 13..16

    obs = np.asarray(values, dtype=np.float64)
    if obs.shape != (OBSERVATION_SIZE,):
        raise RuntimeError(
            f"encode: interner Fehler - Vektor hat {obs.shape[0]} Eintraege, "
            f"erwartet {OBSERVATION_SIZE}"
        )
    if not np.all(np.isfinite(obs)):
        raise RuntimeError("encode: interner Fehler - Vektor enthaelt NaN/Inf")
    return obs


def observation_labels() -> list[str]:
    """Menschlich lesbare Namen aller Eingaben (GUI-Legende, Reihenfolge = encode())."""
    return [
        "reststrecke_x (normiert)",
        "x_position (normiert)",
        "geschwindigkeit",
        "hoehe_ueber_boden",
        "sprungphase (0 Boden/0,5 steig/1 fall)",
        "faellt_gerade",
        "restzeit",
        "ist_tot",
        "flagge_erreicht",
        "hindernis_links",
        "hindernis_rechts",
        "distanz_hindernis_nahe",
        "distanz_hindernis_fern",
        "letzte_aktion_rechts",
        "letzte_aktion_rechts_sprung",
        "letzte_aktion_springen",
        "letzte_aktion_links",
    ]
