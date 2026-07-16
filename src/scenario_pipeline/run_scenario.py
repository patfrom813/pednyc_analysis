"""Run the canonical PedNYC v2/v6 pipeline for one or all raw scenario CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scenario_pipeline.runner import ScenarioRunner
from src.scenario_pipeline.scenario import ScenarioPaths, discover_raw_scenarios, parse_scenario_number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the scenario-agnostic command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_csv", nargs="?", type=Path, help="Compatible raw CSV path")
    parser.add_argument("--raw-path", type=Path, help="Explicit alternative to the positional raw CSV")
    parser.add_argument("--scenario-id", type=int, help="Find one scenario under data/raw")
    parser.add_argument("--all", action="store_true", help="Process every compatible CSV under data/raw")
    parser.add_argument("--participant", type=int, default=1)
    parser.add_argument("--output-root", type=Path, help="Relocate generated data/outputs beneath this root")
    parser.add_argument("--skip-render", action="store_true", help="Skip MP4 and renderer preview generation")
    parser.add_argument("--preview-only", action="store_true", help="Generate previews instead of an MP4")
    parser.add_argument("--preview-time", type=float, help="Generate a preview at one explicit elapsed second")
    parser.add_argument("--ffmpeg", type=Path, help="Explicit ffmpeg executable")
    parser.add_argument("--force", action="store_true", help="Explicitly allow overwriting existing scenario outputs")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue processing remaining --all inputs")
    args = parser.parse_args(argv)
    selected = sum(bool(value) for value in (args.raw_csv, args.raw_path, args.scenario_id is not None, args.all))
    if selected != 1:
        parser.error("provide exactly one of a positional raw CSV, --raw-path, --scenario-id, or --all")
    if args.preview_time is not None:
        args.preview_only = True
    if args.skip_render and args.preview_only:
        parser.error("--skip-render cannot be combined with --preview-only/--preview-time")
    return args


def resolve_raw_inputs(args: argparse.Namespace, project_root: Path = PROJECT_ROOT) -> list[Path]:
    """Resolve CLI selection to one or more unambiguous compatible raw CSV paths."""
    raw_dir = project_root / "data" / "raw"
    if args.all:
        paths = discover_raw_scenarios(raw_dir)
        if not paths:
            raise FileNotFoundError(f"No compatible raw scenario CSVs found in {raw_dir}")
        return paths
    if args.scenario_id is not None:
        matches = [path for path in discover_raw_scenarios(raw_dir) if parse_scenario_number(path) == args.scenario_id]
        if not matches:
            raise FileNotFoundError(f"No raw CSV found for scenario {args.scenario_id} in {raw_dir}")
        if len(matches) > 1:
            raise ValueError(f"Multiple raw CSVs found for scenario {args.scenario_id}: {matches}")
        return matches
    selected = args.raw_path or args.raw_csv
    path = Path(selected)
    if not path.is_absolute():
        path = project_root / path
    return [path.resolve()]


def main(argv: list[str] | None = None) -> int:
    """Process selected scenarios and report each scenario-specific report path."""
    args = parse_args(argv)
    inputs = resolve_raw_inputs(args)
    failures = 0
    for raw_csv in inputs:
        try:
            paths = ScenarioPaths.from_raw(
                raw_csv,
                project_root=PROJECT_ROOT,
                output_root=args.output_root,
                participant=args.participant,
            )
            result = ScenarioRunner(
                paths,
                skip_render=args.skip_render,
                preview_only=args.preview_only,
                preview_time=args.preview_time,
                ffmpeg=args.ffmpeg,
                force=args.force,
            ).run()
            print(
                f"OK scenario={paths.scenario_number} stem={paths.scenario_stem} "
                f"macro={result.statistics['macro_segments']} micro={result.statistics['micro_segments']}"
            )
            print(f"Report: {paths.report_md}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {raw_csv}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
