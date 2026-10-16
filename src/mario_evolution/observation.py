"""Beobachtungsschnittstelle: gym-super-mario-bros-Zustand -> Eingabevektor.

Das neuronale Netz versteht keine Python-Objekte. Es bekommt einen
1D-Vektor aus Gleitkommazahlen. Diese Datei definiert VERBINDLICH,
wie der Vektor aus dem Spielzustand (gymnasium `info`-Dictionary plus
NES-Bild) aufgebaut ist.

Sensor-Design (13 Eingaben, alle normalisiert auf [0, 1]):

  Index 0: reststrecke_x    - horizontale Reststrecke bis zum Level-Ende,
                              normalisiert mit 1/4800 (Levelbreite ~4800 px).
  Index 1: x_position       - absolute Position, normalisiert mit 1/4800.
  Index 2: hoehe            - Höhe über dem Boden, normalisiert mit 1/180.
  Index 3: faellt           - 1.0, wenn Mario gerade fällt (y_pixel wächst).
  Index 4: restzeit         - Restzeit / 400.
  Index 5: tot              - 1.0, wenn Mario tot ist.
  Index 6: flagge           - 1.0, wenn die Flagge erreicht wurde.
  Index 7: hindernis_links  - solides Hindernis (Rohr/Block) im linken
                              Streifen neben Mario (Kalibrierung siehe
                              OBSTACLE_MIN_FRACTION).
  Index 8: hindernis_rechts - solides Hindernis im rechten Streifen.
  Index 9..12: One-Hot der letzten Aktion (4 Navigationsaktionen:
                              rechts / rechts_sprung / springen / links).

Warum Pixel-Sensorik (Himmelfarbe)?
Der NES-Himmel von Level 1-1 hat die exakte Farbe (104, 136, 252).
Mario wird im Bild über seine rote Kleidung gesucht (der Viewport
scrollt mit!). In einem 32 Pixel breiten Streifen neben ihm (Bodennähe
bis Sprunghöhe) wird geprüft, ob ein SOLIDES Hindernis (Rohr, Block)
steht: der nicht-Himmel/nicht-Boden-Anteil liegt beim ersten Rohr bei
~0.30, bei Hintergrundbüschen (gleiche Rohr-Farbe, aber mit Himmel-
Lücken) nur bei ~0.07 - die Schwelle 0.20 trennt beides.
Gegner (Goombas) werden über deren dunkle Pixel (0,0,0) erkannt und
zählen ebenfalls als Hindernis (Index 7/8 = 1).
Das macht die Beobachtung wirklich reaktiv: Das Netz sieht, WO ein
Hindernis oder Gegner steht, nicht nur, WO Mario ist.

Warum One-Hot der letzten Aktion (4 von 7)?
Die 4 Navigationsaktionen decken die Fortbewegung ab; "halten" und die
Renn-Varianten sind Modifikationen derselben Richtung. 4 statt 7 Spalten
halten den Eingaberaum klein ohne erkennbaren Informationsverlust.

Bezug zu "The Nature of Code":
Kapitel 10 beschreibt Perzeptron-Eingaben x_i, die mit Gewichten w_i
multipliziert und summiert werden. Hier liefern wir genau diese x_i:
die "Sensorik" des Agenten. Die Wahl der Eingaben legt fest, WELCHE
Information das Netz überhaupt sehen kann.
"""

from __future__ import annotations

import numpy as np

from .actions import ACTION_COUNT

# Farbe des NES-Himmels (an Frames von SuperMarioBros-1-1-v0 verifiziert:
# blauer Himmel in Level 1-1).
SKY_RGB: tuple[int, int, int] = (104, 136, 252)

# Der BODEN (Ziegel) hat die Farbe (228, 92, 16) - der Sensor darf den
# Boden NICHT als Hindernis melden (Mario läuft darauf!). Der vertikale
# Suchstreifen endet deshalb über der Bodenlinie.
GROUND_RGB: tuple[int, int, int] = (228, 92, 16)

# Farbe von Marios roter Kleidung (verifiziert an echten Frames):
# RGB-Kanal r~248, g~56, b~0. Wird genutzt, um Mario IM BILD zu finden -
# der Viewport scrollt mit, eine feste screen-x ist also falsch.
MARIO_RGB: tuple[int, int, int] = (248, 56, 0)
MARIO_RGB_TOLERANCE: int = 10

# Grün der Rohre/Büsche (0,168,0). Beide haben dieselbe NES-Palette.
# Ein solider Block im Streifen (Anteil > OBSTACLE_MIN_FRACTION) ist
# ein echtes Hindernis. Werte an echten Frames kalibriert:
#   Hintergrundbusch: Anteil ~0.07 (Himmel-Lücken zwischen Büschen)
#   Rohr 16px vor Mario: Anteil ~0.30 (Rohr füllt Band von unten)
#   Schwelle dazwischen: 0.20.
OBSTACLE_MIN_FRACTION: float = 0.20

# Goomba-Körperfell (beige). An echten Frames verifiziert: Diese Farbe
# kommt am Start/Busch NICHT vor, nur an Gegnern - dadurch zuverlässige
# Gegner-Erkennung ohne KI-Bibliothek.
ENEMY_RGB: tuple[int, int, int] = (240, 208, 176)

# Normalisierungen (aus dem Spiel abgelesen, dokumentiert statt geraten):
LEVEL_WIDTH_PX: float = 4800.0     # horizontale Breite eines Levels
MAX_FALL_HEIGHT_PX: float = 180.0  # sinnvoller Vertikalbereich
MAX_TIME: float = 400.0            # Startzeit eines Levels
GROUND_Y_PIXEL: float = 193.0      # y_pixel von Mario am Boden (1-1)
# Fallback-Bildschirm-x, falls Mario (z.B. kleines Mario ohne rote Pixel)
# nicht gefunden wird: Position am Levelanfang.
DEFAULT_MARIO_SCREEN_X: int = 47

OBSERVATION_SIZE: int = 13

# Anzahl der One-Hot-Aktionsspalten (die ersten 4 Aktionen).
_ONE_HOT_ACTIONS: int = 4

NO_PREVIOUS_ACTION: int = -1


def _find_mario_screen_x(frame: np.ndarray) -> int:
    """Findet Marios Bildschirm-x über seine roten Kleidungspixel.

    Warum suchen statt fixe Position? Der NES-Viewport scrollt mit:
    Mario "wandert" von screen-x 47 (Levelanfang) zur Bildmitte (~120).
    Eine feste Position würde falsche Hindernis-Regionen lesen.

    Rückgabe: Mittelwert der roten Pixel-x (int). Fallback:
    DEFAULT_MARIO_SCREEN_X, wenn keine Pixel gefunden werden (z.B.
    kurzes Mario mit anderen Farben in Zwischenzuständen).
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
    """True, wenn die Region ein solides Hindernis oder einen Gegner enthält.

    Definition "solide": Kein Himmel, kein Boden, UND der nicht-freie
    Anteil >= OBSTACLE_MIN_FRACTION der Region. Das unterscheidet
    Rohre/Blöcke (voller Block) von Hintergrundbüschen (gleiche Farbe,
    aber mit Himmel-Lücken zwischen den Büscheln).
    Zusätzlich gilt die Region als gefährlich, wenn sie GOOMBA-Pixel
    enthält (Körperfelle beige (240,208,176), an echten Frames verifi-
    ziert: am Start/Busch kommen diese Farben nicht vor) - sonst läuft
    das Netz blind in Gegner hinein.

    Warum auch Boden prüfen? Der Boden ist kein Hindernis (Mario läuft
    darauf); ohne Ausnahme würde der Sensor jedes Feld als blockiert
    melden.
    """
    sky = np.array(SKY_RGB, dtype=np.uint8)
    ground = np.array(GROUND_RGB, dtype=np.uint8)
    is_sky = np.all(region == sky, axis=-1)
    is_ground = np.all(region == ground, axis=-1)
    passable = is_sky | is_ground
    solid_fraction = 1.0 - float(passable.mean())
    # Gegner-Erkennung: Goomba-Fell beige (240,208,176) im Streifen.
    enemy = np.all(np.abs(region.astype(int) - np.array(ENEMY_RGB)) <= 10, axis=-1)
    return solid_fraction >= OBSTACLE_MIN_FRACTION or bool(enemy.any())


def _sample_region(frame: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Schneidet eine Bildregion heraus (Clipping an den Rändern)."""
    h, w, _ = frame.shape
    return frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1), :]


def encode(frame: np.ndarray,
           info: dict,
           previous_action_index: int = NO_PREVIOUS_ACTION,
           previous_y_pixel: float | None = None) -> np.ndarray:
    """Erzeugt den numerischen Eingabevektor für das neuronale Netz.

    Argumente:
        frame: NES-Bild (240, 256, 3), uint8 - von env.reset()/env.step().
        info: gymnasium-Info-Dictionary (x_pos, y_pixel, time, ...).
        previous_action_index: Index der letzten Aktion (0..6) oder
            NO_PREVIOUS_ACTION am Episodenanfang.
        previous_y_pixel: y_pixel des VORHERIGEN Schritts (für die
            Fall-Erkennung). None am Anfang -> faellt = 0.

    Rückgabe:
        np.ndarray mit OBSERVATION_SIZE Einträgen, alle in [0, 1].
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
        raise ValueError(f"encode: im info-Dictionary fehlen Schlüssel {missing}")
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

    # --- Block A: horizontale Lage --------------------------------------
    x_pos = float(info["x_pos"])
    values.append(min(1.0, max(0.0, (LEVEL_WIDTH_PX - x_pos) / LEVEL_WIDTH_PX)))  # 0
    values.append(min(1.0, max(0.0, x_pos / LEVEL_WIDTH_PX)))                     # 1

    # --- Block B: vertikale Lage ------------------------------------------
    y_pixel = float(info["y_pixel"])
    # Höhe über Boden: am Boden ist y_pixel ~ 193 (abgelesen). Die
    # Differenz zur Bodenlinie ist die aktuelle Sprunghöhe.
    height = max(0.0, GROUND_Y_PIXEL - y_pixel)
    values.append(min(1.0, height / MAX_FALL_HEIGHT_PX))                          # 2
    # y_pixel wächst = Mario sinkt = er fällt (Vergleich mit Vor-Schritt).
    falling = 1.0 if (previous_y_pixel is not None and y_pixel > previous_y_pixel) else 0.0
    values.append(falling)                                                        # 3

    # --- Block C: Spielstatus -----------------------------------------------
    values.append(max(0.0, min(1.0, float(info["time"]) / MAX_TIME)))             # 4
    values.append(1.0 if info.get("is_dead") else 0.0)                            # 5
    values.append(1.0 if info.get("flag_get") else 0.0)                           # 6

    # --- Block D: Pixel-Sensorik (Gefahr/Hindernis neben Mario) ------------
    # y_pixel aus den info-Daten ist die OBERKANTE von Mario; die Füße
    # liegen ~30 Pixel tiefer (verifiziert: Boden y_pixel=176 -> Füße
    # bei screen-y 206). Der Suchstreifen liegt 8 Pixel ÜBER den Füßen
    # (Boden ausgeschlossen) bis 78 Pixel darüber (Sprunghöhe).
    # Bandbreite: 32 Pixel je Seite (verifiziert: das erste Rohr meldet
    # sich 24 Pixel vor Mario; ein Goomba füllt das Band mit dunklen
    # Pixeln, Hintergrundbüsche nicht).
    foot_screen_y = int(y_pixel) + 30
    top_y = max(0, foot_screen_y - 78)
    bottom_y = max(0, foot_screen_y - 8)
    mx = _find_mario_screen_x(frame)
    left_region = _sample_region(frame, top_y, bottom_y, mx - 40, mx - 8)
    right_region = _sample_region(frame, top_y, bottom_y, mx + 8, mx + 40)
    # Hindernis (Rohr/Block): solide Fläche ohne Himmel-Lücken ODER
    # Gegner (dunkle Pixel im Streifen, z.B. Goomba schwarz (0,0,0)).
    # Kalibrierung: Busch ~0.07, Rohr ~0.30, Schwelle 0.20 (siehe oben).
    values.append(1.0 if _is_solid(left_region) else 0.0)                         # 7
    values.append(1.0 if _is_solid(right_region) else 0.0)                        # 8

    # --- Block E: One-Hot der letzten Aktion ---------------------------------
    for i in range(_ONE_HOT_ACTIONS):
        values.append(1.0 if i == previous_action_index else 0.0)                 # 9..12

    obs = np.asarray(values, dtype=np.float64)
    if obs.shape != (OBSERVATION_SIZE,):
        raise RuntimeError(
            f"encode: interner Fehler - Vektor hat {obs.shape[0]} Einträge, "
            f"erwartet {OBSERVATION_SIZE}"
        )
    if not np.all(np.isfinite(obs)):
        raise RuntimeError("encode: interner Fehler - Vektor enthält NaN/Inf")
    return obs


def observation_labels() -> list[str]:
    """Menschlich lesbare Namen aller Eingaben (GUI-Legende, Reihenfolge = encode())."""
    return [
        "reststrecke_x (normiert)",
        "x_position (normiert)",
        "höhe_über_boden",
        "fällt_gerade",
        "restzeit",
        "ist_tot",
        "flagge_erreicht",
        "hindernis_links",
        "hindernis_rechts",
        "letzte_aktion_rechts",
        "letzte_aktion_rechts_sprung",
        "letzte_aktion_springen",
        "letzte_aktion_links",
    ]
