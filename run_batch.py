"""Discover, select, and process PedNYC scenarios in a batch."""

from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

from src.config import DatasetConfig, PipelineConfig
from src.pipeline import ScenarioPipeline


def parse_args() -> argparse.Namespace:
    """Parse batch selection arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--participant", type=int, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true")
    selection.add_argument("-s", "--scenarios", nargs="+")
    selection.add_argument("--count", type=int)
    selection.add_argument("--pattern")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def select_scenarios(available: list[str], args: argparse.Namespace) -> list[str]:
    """Select discovered scenarios according to parsed arguments."""
    if args.scenarios:
        missing = [value for value in args.scenarios if value not in available]
        if missing:
            raise ValueError(f"Unavailable scenarios: {missing}")
        return args.scenarios
    if args.count is not None:
        if args.count < 0:
            raise ValueError("--count must be non-negative")
        return available[: args.count]
    if args.pattern:
        return [value for value in available if fnmatch.fnmatch(value, args.pattern)]
    return available


def main() -> int:
    """Discover and process selected scenarios, reporting each result."""
    args = parse_args()
    root = Path(__file__).resolve().parent
    dataset = DatasetConfig(root)
    available = dataset.discover_scenarios(args.participant)
    if args.list:
        print("\n".join(available) if available else "No scenarios found")
        return 0
    selected = select_scenarios(available, args)
    config = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    pipeline = ScenarioPipeline(dataset, config)
    failures = 0
    for scenario in selected:
        try:
            pipeline.process(args.participant, scenario, not args.no_plots)
            print(f"OK participant={args.participant} scenario={scenario}")
        except Exception as exc:
            failures += 1
            print(f"FAIL participant={args.participant} scenario={scenario}: {exc}")
    print(f"Completed {len(selected) - failures}/{len(selected)} scenarios")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
