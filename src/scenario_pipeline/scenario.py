"""Scenario identity parsing and artifact path derivation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


RAW_NAME_RE = re.compile(
    r"^CSV_Scenario-Ped-(?P<scenario>\d+)_Session-.*\.csv$",
    re.IGNORECASE,
)


def parse_scenario_number(path_or_name: str | Path) -> int:
    """Return the PedNYC scenario number encoded in a compatible raw filename."""
    name = Path(path_or_name).name
    match = RAW_NAME_RE.match(name)
    if not match:
        raise ValueError(
            "Raw filename must match CSV_Scenario-Ped-<N>_Session-*.csv; "
            f"received {name!r}"
        )
    return int(match.group("scenario"))


def discover_raw_scenarios(raw_dir: Path) -> list[Path]:
    """Return every compatible raw scenario CSV in stable scenario/name order."""
    candidates = []
    for path in Path(raw_dir).glob("CSV_Scenario-Ped-*_Session-*.csv"):
        try:
            scenario = parse_scenario_number(path)
        except ValueError:
            continue
        candidates.append((scenario, path.name.lower(), path.resolve()))
    return [path for _, _, path in sorted(candidates)]


@dataclass(frozen=True)
class ScenarioPaths:
    """All inputs and outputs for one canonical PedNYC scenario run."""

    project_root: Path
    artifact_root: Path
    raw_csv: Path
    participant: int
    scenario_number: int
    scenario_stem: str
    processed_dir: Path
    decoded_csv: Path
    metrics_csv: Path
    legacy_segments_csv: Path
    graph_root: Path
    metrics_graph_dir: Path
    fourframe_dir: Path
    macro_dir: Path
    macro_csv: Path
    macro_png: Path
    micro_dir: Path
    micro_v6_dir: Path
    micro_segments_csv: Path
    micro_frame_csv: Path
    micro_summary_csv: Path
    micro_boundaries_csv: Path
    tag_definitions_csv: Path
    micro_standard_png: Path
    micro_stacked_png: Path
    micro_gantt_png: Path
    gantt_mp4: Path
    preview_dir: Path
    report_md: Path

    @classmethod
    def from_raw(
        cls,
        raw_csv: Path,
        project_root: Path,
        output_root: Path | None = None,
        participant: int = 1,
    ) -> "ScenarioPaths":
        """Derive a complete scenario path set from one compatible raw CSV."""
        project = Path(project_root).resolve()
        raw = Path(raw_csv)
        if not raw.is_absolute():
            raw = project / raw
        raw = raw.resolve()
        scenario = parse_scenario_number(raw)
        artifact_root = Path(output_root).resolve() if output_root else project
        stem = f"PedNYC{int(participant)}_scenario{scenario}"
        processed = artifact_root / "data" / "processed" / f"pednyc{participant}" / f"scenario{scenario}"
        graph_root = artifact_root / "outputs" / "graphs" / "exact_v6" / f"pednyc{participant}" / f"scenario{scenario}"
        metrics_dir = graph_root / "metrics_v1"
        macro_dir = graph_root / "macro_v2"
        micro_dir = graph_root / "micro"
        micro_v6 = micro_dir / "v6"
        fourframe = artifact_root / "4frame_view" / f"pednyc{participant}" / f"scenario{scenario}" / "metrics_v1"
        return cls(
            project_root=project,
            artifact_root=artifact_root,
            raw_csv=raw,
            participant=int(participant),
            scenario_number=scenario,
            scenario_stem=stem,
            processed_dir=processed,
            decoded_csv=processed / f"decoded_clean_{stem}.csv",
            metrics_csv=processed / f"features_{stem}_metrics_v1.csv",
            legacy_segments_csv=processed / f"segments_{stem}_v1.csv",
            graph_root=graph_root,
            metrics_graph_dir=metrics_dir,
            fourframe_dir=fourframe,
            macro_dir=macro_dir,
            macro_csv=macro_dir / f"macro_segments_{stem}_v2.csv",
            macro_png=macro_dir / f"macro_segments_{stem}_v2.png",
            micro_dir=micro_dir,
            micro_v6_dir=micro_v6,
            micro_segments_csv=micro_v6 / f"micro_segments_descriptive_{stem}_v6.csv",
            micro_frame_csv=micro_v6 / f"features_with_descriptive_micro_segments_{stem}_v6.csv",
            micro_summary_csv=micro_v6 / f"micro_segment_summary_by_macro_{stem}_v6.csv",
            micro_boundaries_csv=micro_v6 / f"micro_change_boundaries_{stem}_v6.csv",
            tag_definitions_csv=micro_v6 / "micro_event_tag_definitions_v6.csv",
            micro_standard_png=micro_v6 / f"micro_standard_3panel_{stem}_v6.png",
            micro_stacked_png=micro_v6 / f"micro_stacked_observable_inferred_{stem}_v6.png",
            micro_gantt_png=micro_v6 / f"micro_gantt_observable_inferred_{stem}_v6.png",
            gantt_mp4=micro_v6 / f"micro_gantt_observable_inferred_{stem}_v6.mp4",
            preview_dir=micro_v6 / "previews",
            report_md=graph_root / f"{stem}_report.md",
        )

    def core_csvs(self) -> tuple[Path, ...]:
        """Return the nonempty CSVs required for a complete Stage 0-3 run."""
        return (
            self.decoded_csv,
            self.metrics_csv,
            self.legacy_segments_csv,
            self.macro_csv,
            self.micro_segments_csv,
            self.micro_frame_csv,
            self.micro_summary_csv,
            self.micro_boundaries_csv,
            self.tag_definitions_csv,
        )

    def expected_plots(self) -> tuple[Path, ...]:
        """Return canonical macro and micro plots (Stage 1 plots are discovered)."""
        return (
            self.macro_png,
            self.micro_standard_png,
            self.micro_stacked_png,
            self.micro_gantt_png,
        )
