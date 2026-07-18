"""
EchoForge AI — Canonical data paths.

Import ``VOCAB_PATH`` from here everywhere you need the phonemizer vocab file.
Never hardcode the path in any other module — this is the single source of truth.
"""

from pathlib import Path

# Project root is the directory that contains both ``src/`` and ``data/``.
# This file lives at  src/echoforge_tts/config/paths.py  → 3 levels up.
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

# --------------------------------------------------------------------------- #
#  Single fixed location for the Hindi + English phonemizer vocab file.       #
#  • Training (prepare_csv_wavs.py) writes here after processing a dataset.   #
#  • Inference  (infer/utils_infer.py)  reads from here.                      #
#  • get_tokenizer() uses this path when tokenizer == "phonemizer".           #
# --------------------------------------------------------------------------- #
VOCAB_PATH: Path = _PROJECT_ROOT / "data" / "echoforge_hindi_en_phonemizer" / "vocab.txt"
