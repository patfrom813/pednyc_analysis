"""Run one participant/scenario through the PedNYC pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_v6_batch import process_scenario


def parse_args() -> argparse.Namespace:
    """Parse single-scenario command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--participant", type=int, required=True)
    parser.add_argument("-s", "--scenario", required=True)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Execute one scenario and print its saved outputs."""
    args = parse_args()
    root = Path(__file__).resolve().parent
    output_dir = process_scenario(root, args.participant, args.scenario, args.no_plots)
    print(f"Processed exact-v6 participant {args.participant}, scenario {args.scenario}")
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
