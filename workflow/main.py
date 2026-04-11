from __future__ import annotations

import argparse
import sys
from pathlib import Path

from workflow.classes.pipeline import MoleculeClusteringPipeline
from workflow.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the molecule clustering workflow.")
    parser.add_argument(
        "--config",
        default="workflow/config.toml",
        help="Path to the TOML config file.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config and exit.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate config and create output directories without running the pipeline.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    if args.print_config:
        print(config.to_json())
        return

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        print(f"Prepared output directory: {output_dir}")
        return

    pipeline = MoleculeClusteringPipeline(config=config, config_path=Path(args.config))
    try:
        outputs = pipeline.run()
    except RuntimeError as exc:
        print(f"Workflow failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(outputs.to_json())


if __name__ == "__main__":
    main()
