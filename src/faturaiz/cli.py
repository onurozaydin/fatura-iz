"""Command-line entry points for reproducible training and local scoring."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from faturaiz.config import load_settings
from faturaiz.model import load_verified_bundle
from faturaiz.service import triage_batch
from faturaiz.training import run_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="faturaiz", description="Duplicate-invoice triage")
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="generate synthetic data, train, and evaluate")
    train.add_argument("--output-dir", default=".")
    score = subparsers.add_parser("score", help="score a CSV batch using a verified artifact")
    score.add_argument("csv_path")
    score.add_argument("--reference-pepper", default="demo-only-pepper")
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = load_settings(args.config)
    if args.command == "train":
        metrics = run_training(settings, Path(args.output_dir))
        print(json.dumps(metrics, indent=2))
        return
    frame = pd.read_csv(args.csv_path, keep_default_na=False)
    if "group_id" not in frame.columns:
        frame["group_id"] = frame["invoice_id"]
    bundle = load_verified_bundle(settings.service.artifact_path, settings.service.manifest_path)
    results = triage_batch(
        frame,
        bundle=bundle,
        candidate_config=settings.candidates,
        reference_pepper=args.reference_pepper,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
