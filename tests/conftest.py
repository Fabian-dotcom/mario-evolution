"""Stellt sicher, dass `src` im Pfad liegt (Projekt nutzt src-Layout)."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
