"""Create a human-annotation template from merged PedNYC events."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import DatasetConfig


def parse_args() -> argparse.Namespace:
    """Parse validation command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--participant", type=int, required=True)
    parser.add_argument("-s", "--scenario", required=True)
    return parser.parse_args()


def main() -> int:
    """Load merged events and write an annotation-ready CSV."""
    args = parse_args()
    dataset = DatasetConfig(Path(__file__).resolve().parent)
    output_dir = dataset.get_output_dir(args.participant, args.scenario)
    events_path = output_dir / "merged_events.csv"
    if not events_path.is_file():
        raise FileNotFoundError(f"Run the scenario pipeline first; missing {events_path}")
    events = pd.read_csv(events_path)
    template = events[["start_time", "end_time", "label"]].rename(columns={"label": "predicted_label"})
    template["human_label"] = ""
    template["correct"] = ""
    template["notes"] = ""
    template_path = output_dir / "validation_template.csv"
    template.to_csv(template_path, index=False)
    print(events.to_string(index=False))
    print(f"Validation template: {template_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
