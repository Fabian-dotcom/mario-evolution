"""Aktionen für gym-super-mario-bros (github.com/Kautenja/gym-super-mario-bros).

Das nes-py-Environment nimmt pro Schritt einen BYTE entgegen, dessen
Bits den NES-Joypad-Tasten entsprechen (binary encoding):

    right=128, left=64, down=32, up=16, start=8, select=4, B=2, A=1

Das gym-super-mario-bros-Paket definiert daraus fertige Aktionsmengen
(SIMPLE_MOVEMENT usw.). Wir legen fest: Unser Aktionsindex i entspricht
SIMPLE_MOVEMENT[i], und die dazugehörige Joypad-Bitmaske steht in
GYM_ACTION_IDS (experimentell verifiziert: right+A = 129 bewegt Mario
vorwärts, siehe Tests).

Bezug zu "The Nature of Code":
Der Aktionsindex ist die Interpretation der Netzausgabe (Kapitel 10:
Ausgabeneuronen -> Entscheidung). Die Auswahlregel selbst (argmax) ist
Anwendungslogik und liegt hier.
"""

from __future__ import annotations

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Bits des NES-Joypads im nes-py "binary encoding" (verifiziert).
_BUTTON_BITS: dict[str, int] = {
    "right": 1 << 7,   # 128
    "left": 1 << 6,    # 64
    "down": 1 << 5,    # 32
    "up": 1 << 4,      # 16
    "start": 1 << 3,   # 8
    "select": 1 << 2,  # 4
    "B": 1 << 1,       # 2
    "A": 1 << 0,       # 1
    "NOOP": 0,         # keine Taste
}


def _bitmask(buttons: list[str]) -> int:
    """Kombiniert Tastennamen zur Joypad-Bitmaske (OR-Verknüpfung)."""
    value = 0
    for b in buttons:
        value |= _BUTTON_BITS[b]
    return value


# FESTE Reihenfolge! Index i im Netzausgabevektor == ALL_ACTIONS[i] ==
# SIMPLE_MOVEMENT[i]. GYM_ACTION_IDS[i] ist der Byte-Wert, den
# env.step() erwartet (SIMPLE_MOVEMENT-Kombinationen als Bitmasken).
ALL_ACTIONS: tuple[str, ...] = (
    "halten",               # 0: NOOP
    "rechts",               # 1: right
    "rechts_sprung",        # 2: right + A
    "rennen_rechts",        # 3: right + B
    "rennen_sprung_rechts", # 4: right + A + B
    "springen",             # 5: A (auf der Stelle)
    "links",                # 6: left
)

# Joypad-Bitmasken, parallel zu ALL_ACTIONS (aus SIMPLE_MOVEMENT abgeleitet).
GYM_ACTION_IDS: tuple[int, ...] = tuple(
    _bitmask(list(buttons)) for buttons in SIMPLE_MOVEMENT
)

ACTION_COUNT: int = len(ALL_ACTIONS)


def action_count() -> int:
    """Anzahl der verfügbaren Aktionen = Größe der Netzausgabeschicht."""
    return ACTION_COUNT


def gym_action_index(action_index: int) -> int:
    """Übersetzt unseren Aktionsindex in den Joypad-Byte für env.step()."""
    if not isinstance(action_index, int) or isinstance(action_index, bool):
        raise TypeError(f"gym_action_index: Index muss int sein, erhielt {action_index!r}")
    if not (0 <= action_index < ACTION_COUNT):
        raise ValueError(
            f"gym_action_index: Index muss in [0, {ACTION_COUNT - 1}] liegen, erhielt {action_index}"
        )
    return GYM_ACTION_IDS[action_index]


def choose_action(values) -> int:
    """Übersetzt Netzausgabewerte in einen Aktionsindex.

    Mathematisch: argmax_a values[a] - der größte Ausgabewert bestimmt
    die Aktion (Tie-Break: kleinster Index, wie numpy.argmax). Die
    Werte sind rohe "Aktionswerte" (keine Wahrscheinlichkeiten).
    """
    import numpy as np  # lokal, damit actions.py ohne numpy nutzbar bleibt

    arr = np.asarray(values, dtype=float)
    if arr.shape != (ACTION_COUNT,):
        raise ValueError(
            f"choose_action: erwartet {ACTION_COUNT} Werte (einer pro Aktion), "
            f"erhielt shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"choose_action: Ausgabewerte müssen endlich sein, erhielt {arr}")
    return int(np.argmax(arr))
