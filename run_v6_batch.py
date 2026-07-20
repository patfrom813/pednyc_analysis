"""Run the established metrics-v1, macro-v2, and micro-v6 pipeline unchanged."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from src.config import DatasetConfig


def parse_args() -> argparse.Namespace:
    """Parse participant and scenario selection for the exact v6 workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--participant", type=int, required=True)
    parser.add_argument("-s", "--scenarios", nargs="+", required=True)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def run_command(arguments: list[str], root: Path) -> None:
    """Run one pipeline stage with the active Python interpreter."""
    subprocess.run([sys.executable, *arguments], cwd=root, check=True)


def process_scenario(root: Path, participant: int, scenario: str, no_plots: bool) -> Path:
    """Execute the original four-stage algorithm for one discovered scenario."""
    dataset = DatasetConfig(root)
    raw_csv = dataset.get_scenario_csv(participant, scenario)
    identity = f"PedNYC{participant}_scenario{scenario}"
    processed_dir = root / "data" / "processed" / f"pednyc{participant}" / f"scenario{scenario}"
    graph_dir = root / "outputs" / "graphs" / "exact_v6" / f"pednyc{participant}" / f"scenario{scenario}"
    decoded_csv = processed_dir / f"decoded_clean_{identity}.csv"
    features_csv = processed_dir / f"features_{identity}_metrics_v1.csv"
    segments_v1_csv = processed_dir / f"segments_{identity}_v1.csv"
    metrics_dir = graph_dir / "metrics_v1"
    macro_dir = graph_dir / "macro_v2"
    macro_csv = macro_dir / f"macro_segments_{identity}_v2.csv"
    macro_png = macro_dir / f"macro_segments_{identity}_v2.png"
    micro_dir = graph_dir / "micro"
    fourframe_dir = root / "4frame_view" / f"pednyc{participant}" / f"scenario{scenario}" / "metrics_v1"

    run_command(["src/preprocess/decode_first.py", "--input-csv", str(raw_csv), "--output-csv", str(decoded_csv)], root)
    run_command([
        "src/features/create_metrics_and_graph.py",
        "--input-csv", str(decoded_csv),
        "--output-csv", str(features_csv),
        "--segment-csv", str(segments_v1_csv),
        "--graph-dir", str(metrics_dir),
        "--fourframe-dir", str(fourframe_dir),
    ], root)
    run_command([
        "src/segmentation/macro_segmentation.py",
        "--input-csv", str(features_csv),
        "--output-dir", str(macro_dir),
        "--segments-csv", str(macro_csv),
        "--plot-path", str(macro_png),
    ], root)
    micro_command = [
        "src/segmentation/micro_segmentation.py",
        "--feature-csv", str(features_csv),
        "--macro-csv", str(macro_csv),
        "--output-dir", str(micro_dir),
        "--participant", str(participant),
        "--scenario", str(scenario),
    ]
    if no_plots:
        micro_command.append("--no-plot")
    run_command(micro_command, root)
    return graph_dir


def main() -> int:
    """Process every requested scenario and report exact-v6 output locations."""
    args = parse_args()
    root = Path(__file__).resolve().parent
    failures = 0
    for scenario in args.scenarios:
        try:
            output_dir = process_scenario(root, args.participant, scenario, args.no_plots)
            print(f"OK exact-v6 participant={args.participant} scenario={scenario} outputs={output_dir}")
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            failures += 1
            print(f"FAIL exact-v6 participant={args.participant} scenario={scenario}: {exc}")
    print(f"Completed {len(args.scenarios) - failures}/{len(args.scenarios)} exact-v6 scenarios")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
