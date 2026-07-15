"""Central filesystem discovery and algorithm configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
import ast
import re
from typing import Any


@dataclass
class DatasetConfig:
    """Discover PedNYC participants, scenarios, and output locations."""

    base_dir: Path = Path(".")
    participant_pattern: str = "csv_pednyc{}"

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir).resolve()

    def discover_participants(self) -> list[int]:
        """Return sorted participant IDs found below ``base_dir``."""
        prefix, suffix = self.participant_pattern.split("{}", 1)
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$", re.IGNORECASE)
        participants = []
        if not self.base_dir.exists():
            return participants
        for path in self.base_dir.iterdir():
            match = pattern.match(path.name) if path.is_dir() else None
            if match:
                participants.append(int(match.group(1)))
        return sorted(set(participants))

    def get_csv_dir(self, pid: int) -> Path:
        """Return the flat CSV directory for a participant."""
        return self.base_dir / self.participant_pattern.format(int(pid)) / "csv"

    def discover_scenarios(self, pid: int) -> list[str]:
        """Return scenarios parsed from participant CSV filenames."""
        csv_dir = self.get_csv_dir(pid)
        if not csv_dir.is_dir():
            return []
        scenarios: set[str] = set()
        numeric = re.compile(r"^CSV_Scenario-Ped-(\d+)_Session-.*\.csv$", re.IGNORECASE)
        practice = re.compile(r"^CSV_Scenario-Practice_Session-.*\.csv$", re.IGNORECASE)
        for path in csv_dir.iterdir():
            match = numeric.match(path.name) if path.is_file() else None
            if match:
                scenarios.add(match.group(1))
            elif path.is_file() and practice.match(path.name):
                scenarios.add("Practice")
        return sorted(scenarios, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value.lower()))

    def get_scenario_csv(self, pid: int, scenario: str) -> Path:
        """Resolve one scenario CSV, rejecting missing or ambiguous matches."""
        value = str(scenario).strip()
        if not value:
            raise ValueError("Scenario cannot be empty")
        stem = "Practice" if value.lower() == "practice" else f"Ped-{int(value)}"
        matches = sorted(self.get_csv_dir(pid).glob(f"CSV_Scenario-{stem}_Session-*.csv"))
        if not matches:
            raise FileNotFoundError(f"No CSV found for participant {pid}, scenario {value}")
        if len(matches) > 1:
            raise ValueError(f"Multiple CSVs found for participant {pid}, scenario {value}: {matches}")
        return matches[0]

    def get_output_dir(self, pid: int, scenario: str) -> Path:
        """Create and return the processed scenario output directory."""
        return self._mkdir(self.base_dir / "outputs" / "processed" / f"pednyc{int(pid)}" / f"scenario{scenario}")

    def get_graphs_dir(self, pid: int) -> Path:
        """Create and return the participant graph directory."""
        return self._mkdir(self.base_dir / "outputs" / "graphs" / f"pednyc{int(pid)}")

    def get_animations_dir(self, pid: int) -> Path:
        """Create and return the participant animation directory."""
        return self._mkdir(self.base_dir / "outputs" / "animations" / f"pednyc{int(pid)}")

    @staticmethod
    def _mkdir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class PipelineConfig:
    """All tunable cleaning, segmentation, and behavior thresholds."""

    speed_spike_threshold: float = 8.0
    smoothing_window: int = 9
    acceleration_smoothing_window: int = 11
    distance_smoothing_window: int = 9
    time_start_zero: bool = True
    speed_threshold: float = 0.15
    walking_speed: float = 0.30
    min_duration: float = 0.45
    acceleration_slope: float = 0.12
    deceleration_slope: float = -0.25
    macro_smoothing_seconds: float = 0.45
    macro_accel_search_seconds: float = 4.0
    final_deceleration_fraction: float = 0.50
    window_size: float = 1.0
    window_step: float = 0.5
    window_min: float = 0.25
    window_max: float = 2.50
    hesitation_speed_var: float = 0.22
    hesitation_min_reversals: int = 2
    hesitation_max_progress: float = 0.70
    hesitation_drop_min: float = 0.18
    hesitation_recovery_fraction: float = 0.35
    hesitation_max_duration: float = 1.50
    yielding_distance: float = 12.0
    head_check_yaw: float = 20.0
    head_check_turn_rate: float = 120.0
    head_check_max_duration: float = 0.50
    head_active_turn_rate: float = 50.0
    near_stationary_speed: float = 0.15
    behavior_walking_speed: float = 0.25
    steady_speed_cv_max: float = 0.22
    closing_threshold: float = 0.0
    slowing_acceleration: float = -0.25
    starting_acceleration: float = 0.25
    speeding_acceleration: float = 0.35
    pause_speed: float = 0.10
    pause_min_duration: float = 0.12
    merge_gap: float = 0.15
    merge_gap_threshold: float = 0.15
    min_event_duration: float = 0.25
    confidence_default: float = 0.75

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """Load known configuration fields from a YAML mapping."""
        try:
            import yaml
        except ImportError:
            values = _read_flat_yaml(Path(path))
        else:
            with Path(path).open("r", encoding="utf-8") as handle:
                values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ValueError("Pipeline configuration must be a YAML mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown pipeline configuration fields: {unknown}")
        return cls(**values)

    def to_yaml(self, path: Path) -> None:
        """Write this configuration to YAML."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            import yaml
        except ImportError:
            with output.open("w", encoding="utf-8") as handle:
                for key, value in asdict(self).items():
                    rendered = str(value).lower() if isinstance(value, bool) else str(value)
                    handle.write(f"{key}: {rendered}\n")
        else:
            with output.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(asdict(self), handle, sort_keys=False)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the configuration."""
        return asdict(self)


def _read_flat_yaml(path: Path) -> dict[str, Any]:
    """Parse the flat scalar YAML format used by ``PipelineConfig``."""
    values: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Invalid YAML at line {line_number}: {raw.rstrip()}")
            key, raw_value = (part.strip() for part in line.split(":", 1))
            lowered = raw_value.lower()
            if lowered in {"true", "false"}:
                value: Any = lowered == "true"
            elif lowered in {"null", "none", "~"}:
                value = None
            else:
                try:
                    value = ast.literal_eval(raw_value)
                except (SyntaxError, ValueError):
                    value = raw_value.strip("\"'")
            values[key] = value
    return values
