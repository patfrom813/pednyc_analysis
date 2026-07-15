"""Run one participant/scenario through the PedNYC pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DatasetConfig, PipelineConfig
from src.pipeline import ScenarioPipeline


def parse_args() -> argparse.Namespace:
    """Parse single-scenario command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--participant", type=int, required=True)
    parser.add_argument("-s", "--scenario", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Execute one scenario and print its saved outputs."""
    args = parse_args()
    root = Path(__file__).resolve().parent
    config = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    result = ScenarioPipeline(DatasetConfig(root), config).process(args.participant, args.scenario, not args.no_plots)
    print(f"Processed participant {args.participant}, scenario {args.scenario}")
    for name, path in result["output_paths"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
