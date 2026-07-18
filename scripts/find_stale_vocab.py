#!/usr/bin/env python3
"""
EchoForge AI — Stale vocab.txt scanner / cleaner.

Scans the entire project for any vocab.txt files that live outside the
canonical VOCAB_PATH location, reports them, and deletes them.

Can be imported and called programmatically, or run as a standalone script:

    python scripts/find_stale_vocab.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

# Allow running from any working directory
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR.parent / "src"))

from echoforge_tts.config.paths import VOCAB_PATH


def scan_and_clean_stale_vocabs(dry_run: bool = False) -> list[Path]:
    """Find all vocab.txt files in the project that are NOT the canonical VOCAB_PATH.

    Prints a warning for each stale file found, then deletes them (unless
    ``dry_run=True``).  Skips anything inside ``runtime/triton_trtllm/``.

    Returns the list of stale paths that were found (and deleted if not dry-run).
    """
    project_root = VOCAB_PATH.parents[2]  # …/data/echoforge_hindi_en_phonemizer/vocab.txt → root

    stale: list[Path] = []
    for vocab_file in project_root.rglob("vocab.txt"):
        # Skip triton runtime (out of scope)
        if "triton_trtllm" in vocab_file.parts:
            continue
        # Skip the canonical location itself
        if vocab_file.resolve() == VOCAB_PATH.resolve():
            continue
        stale.append(vocab_file)

    if not stale:
        print("✅ No stale vocab.txt files found outside canonical path.")
        return []

    for path in stale:
        print(f"⚠️  Old/misplaced vocab.txt found at: {path}")
        if not dry_run:
            try:
                path.unlink()
                print(f"   🗑️  Deleted: {path}")
            except OSError as e:
                print(f"   ❌ Could not delete {path}: {e}")

    if dry_run:
        print(f"\n[dry-run] {len(stale)} stale file(s) would be deleted.")
    else:
        print(f"\n🧹 Cleaned {len(stale)} stale vocab.txt file(s).")

    return stale


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan and clean stale vocab.txt files.")
    parser.add_argument("--dry-run", action="store_true", help="Report without deleting.")
    args = parser.parse_args()
    scan_and_clean_stale_vocabs(dry_run=args.dry_run)
