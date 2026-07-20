"""Compare exact micro-v6 label distributions across processed scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse participant and scenarios to compare."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--participant", type=int, required=True)
    parser.add_argument("-s", "--scenarios", nargs="+", required=True)
    return parser.parse_args()


def build_comparison(root: Path, participant: int, scenarios: list[str]) -> pd.DataFrame:
    """Summarize counts and durations for every v6 tag dimension."""
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        directory = root / "outputs" / "graphs" / "exact_v6" / f"pednyc{participant}" / f"scenario{scenario}" / "micro" / "v6"
        matches = sorted(directory.glob("micro_segments_descriptive_*_v6.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one exact-v6 segment CSV in {directory}; found {len(matches)}")
        segments = pd.read_csv(matches[0])
        total_segments = len(segments)
        inferred_segments = int(segments["has_inferred_behavior"].sum())
        for dimension, column in (("motion", "motion_tag"), ("head", "head_tag"), ("car_context", "car_tag")):
            grouped = segments.groupby(column, dropna=False)["duration_sec"].agg(["size", "sum"]).reset_index()
            for _, label in grouped.iterrows():
                rows.append({
                    "participant": participant,
                    "scenario": scenario,
                    "micro_segments": total_segments,
                    "inferred_segments": inferred_segments,
                    "tag_dimension": dimension,
                    "tag": label[column],
                    "segment_count": int(label["size"]),
                    "total_duration_sec": float(label["sum"]),
                    "percent_of_segments": 100.0 * float(label["size"]) / total_segments if total_segments else 0.0,
                })
    return pd.DataFrame(rows)


def main() -> int:
    """Write and display the cross-scenario v6 label comparison."""
    args = parse_args()
    root = Path(__file__).resolve().parent
    comparison = build_comparison(root, args.participant, args.scenarios)
    scenario_token = "_".join(args.scenarios)
    output = root / "outputs" / "graphs" / "exact_v6" / f"pednyc{args.participant}" / f"v6_label_comparison_{scenario_token}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output, index=False)
    print(comparison.to_string(index=False))
    print(f"Comparison CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
