"""CI entry point for semantic training-output reproducibility."""

from __future__ import annotations

import argparse
from pathlib import Path

from faturaiz.reproducibility import assert_reproducible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("candidate_root", type=Path)
    args = parser.parse_args()
    assert_reproducible(args.reference_root, args.candidate_root)
    print("Training outputs are semantically reproducible and integrity-verified.")


if __name__ == "__main__":
    main()
