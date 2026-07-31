"""Pedestrian and vehicle X/Z trajectory analysis.

The raw diagnostic phase deliberately applies no spatial translation, rotation,
reflection, scaling, or progress normalization. It reuses the validated loader,
column normalization, cleaning, smoothing, and movement-onset definition from
the existing feature pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.analysis.scenario_metadata import SCENARIO_DESCRIPTIONS, scenario_description
from src.config import PipelineConfig
from src.features.column_map import normalize_columns
from src.features.metrics_engine import MetricsEngine
from src.preprocess.cleaner import clean_metrics
from src.preprocess.loader import load_scenario, validate_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIOS = tuple(SCENARIO_DESCRIPTIONS)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs/scenario_pattern_expanded/manifests/processing_success_manifest.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/trajectory_analysis"


@dataclass(frozen=True)
class RecordingDiagnostic:
    recording_identifier: str
    participant_id: str
    scenario: int
    source_path: str
    row_count: int
    alignment_event: str
    alignment_time: float
    ped_start_x: float
    ped_start_z: float
    ped_end_x: float
    ped_end_z: float
    veh_start_x: float
    veh_start_z: float
    veh_end_x: float
    veh_end_z: float
    ped_displacement: float
    veh_displacement: float
    ped_direction_deg: float
    veh_direction_deg: float
    ped_path_length: float
    veh_path_length: float
    duration: float
    source_time_column: str
    source_ped_x_column: str
    source_ped_z_column: str
    source_veh_x_column: str
    source_veh_z_column: str
    coordinate_transform: str = "none"
    coordinate_unit: str = "Unity world units (treated as meters by existing pipeline)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios", nargs="+", type=int, default=list(DEFAULT_SCENARIOS)
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress-points", type=int, default=101)
    parser.add_argument(
        "--phase",
        choices=("raw", "full"),
        default="raw",
        help="Run raw diagnostics only or continue after coordinate validation.",
    )
    return parser.parse_args()


def _source_column(frame: pd.DataFrame, standard: str) -> str:
    aliases = {
        "time": ("scenario_time", "time", "elapsed_time_sec"),
        "ped_x": (
            "b_vr_pos_x", "pedestrianpositionx", "ped_pos_x",
            "playerposition_x", "ped_x", "b_vr_pos",
        ),
        "ped_z": (
            "b_vr_pos_z", "pedestrianpositionz", "ped_pos_z",
            "playerposition_z", "ped_z", "b_vr_pos",
        ),
        "veh_x": (
            "a_car_pos_x", "vehiclepositionx", "car_x", "veh_x", "a_car_pos",
        ),
        "veh_z": (
            "a_car_pos_z", "vehiclepositionz", "car_z", "veh_z", "a_car_pos",
        ),
    }
    lookup = {
        "".join(character for character in str(column).lower() if character.isalnum()):
        str(column)
        for column in frame.columns
    }
    for alias in aliases[standard]:
        token = "".join(character for character in alias.lower() if character.isalnum())
        if token in lookup:
            return lookup[token]
    return "decoded vector source"


def _path_length(x: pd.Series, z: pd.Series) -> float:
    return float(np.nansum(np.hypot(np.diff(x.to_numpy()), np.diff(z.to_numpy()))))


def _direction_degrees(x: pd.Series, z: pd.Series) -> float:
    return float(np.degrees(np.arctan2(z.iloc[-1] - z.iloc[0], x.iloc[-1] - x.iloc[0])))


def load_recording(row: pd.Series, config: PipelineConfig) -> tuple[pd.DataFrame, RecordingDiagnostic]:
    """Load one recording with the repository's validated feature pipeline."""
    source = Path(row.source_csv_path)
    raw = load_scenario(source)
    source_columns = {
        name: _source_column(raw, name)
        for name in ("time", "ped_x", "ped_z", "veh_x", "veh_z")
    }
    normalized = normalize_columns(raw)
    validation = validate_scenario(normalized)
    if not validation["valid"]:
        raise ValueError(f"Input validation failed for {source}: {validation}")
    cleaned = clean_metrics(normalized, config)
    metrics = MetricsEngine(config).compute(cleaned).reset_index(drop=True)
    if len(metrics) < 2:
        raise ValueError(f"Fewer than two usable rows in {source}")
    onset_indices = np.flatnonzero(
        metrics["ped_speed"].ge(config.walking_speed).to_numpy()
    )
    if len(onset_indices):
        alignment_time = float(metrics["time"].iloc[onset_indices[0]])
        alignment_event = "pedestrian_movement_onset"
    else:
        alignment_time = float(metrics["time"].iloc[0])
        alignment_event = "scenario_relative_start_fallback"
    participant_id = f"PedNYC{int(row.pednyc_number)}"
    recording_id = str(row.recording_identifier)
    scenario = int(row.scenario_number)
    traces = pd.DataFrame({
        "participant_id": participant_id,
        "pednyc_number": int(row.pednyc_number),
        "scenario": scenario,
        "recording_identifier": recording_id,
        "scenario_time": metrics["time"],
        "aligned_time": metrics["time"] - alignment_time,
        "ped_x": metrics["ped_x"],
        "ped_z": metrics["ped_z"],
        "veh_x": metrics["veh_x"],
        "veh_z": metrics["veh_z"],
        "ped_speed": metrics["ped_speed"],
        "veh_speed": metrics["veh_speed"],
        "ped_acceleration": metrics["ped_acceleration"],
        "veh_acceleration": metrics["veh_acceleration"],
    })
    diagnostic = RecordingDiagnostic(
        recording_identifier=recording_id,
        participant_id=participant_id,
        scenario=scenario,
        source_path=str(source.resolve()),
        row_count=len(traces),
        alignment_event=alignment_event,
        alignment_time=alignment_time,
        ped_start_x=float(traces.ped_x.iloc[0]),
        ped_start_z=float(traces.ped_z.iloc[0]),
        ped_end_x=float(traces.ped_x.iloc[-1]),
        ped_end_z=float(traces.ped_z.iloc[-1]),
        veh_start_x=float(traces.veh_x.iloc[0]),
        veh_start_z=float(traces.veh_z.iloc[0]),
        veh_end_x=float(traces.veh_x.iloc[-1]),
        veh_end_z=float(traces.veh_z.iloc[-1]),
        ped_displacement=float(np.hypot(
            traces.ped_x.iloc[-1] - traces.ped_x.iloc[0],
            traces.ped_z.iloc[-1] - traces.ped_z.iloc[0],
        )),
        veh_displacement=float(np.hypot(
            traces.veh_x.iloc[-1] - traces.veh_x.iloc[0],
            traces.veh_z.iloc[-1] - traces.veh_z.iloc[0],
        )),
        ped_direction_deg=_direction_degrees(traces.ped_x, traces.ped_z),
        veh_direction_deg=_direction_degrees(traces.veh_x, traces.veh_z),
        ped_path_length=_path_length(traces.ped_x, traces.ped_z),
        veh_path_length=_path_length(traces.veh_x, traces.veh_z),
        duration=float(traces.scenario_time.iloc[-1] - traces.scenario_time.iloc[0]),
        source_time_column=source_columns["time"],
        source_ped_x_column=source_columns["ped_x"],
        source_ped_z_column=source_columns["ped_z"],
        source_veh_x_column=source_columns["veh_x"],
        source_veh_z_column=source_columns["veh_z"],
    )
    return traces, diagnostic


LINKED_CLICK_SCRIPT = r"""
(function () {
  const plot = document.getElementById('{plot_id}');
  let selected = null;
  const participantIndices = () => plot.data
    .map((trace, index) => trace.meta?.role === 'participant' ? index : -1)
    .filter(index => index >= 0);
  function reset() {
    const indices = participantIndices();
    Plotly.restyle(plot, {
      'line.width': indices.map(index =>
        plot.data[index].meta.actor === 'pedestrian' ? 2 : 1.5),
      'opacity': indices.map(() => 0.32)
    }, indices);
    selected = null;
  }
  plot.on('plotly_click', event => {
    if (!event.points.length) return;
    const trace = plot.data[event.points[0].curveNumber];
    if (trace.meta?.role !== 'participant') return;
    const participant = trace.meta.participant_id;
    if (selected === participant) return reset();
    const indices = participantIndices();
    Plotly.restyle(plot, {
      'line.width': indices.map(index =>
        plot.data[index].meta.participant_id === participant ? 5 : 0.7),
      'opacity': indices.map(index =>
        plot.data[index].meta.participant_id === participant ? 1 : 0.05)
    }, indices);
    selected = participant;
  });
  plot.on('plotly_doubleclick', () => { reset(); return false; });
  plot.setAttribute('data-linked-highlight-ready', 'true');
})();
"""


def _add_direction_marker(
    figure: go.Figure, one: pd.DataFrame, actor: str, color: str, symbol: str
) -> None:
    x_column, z_column = f"{actor[:3]}_x", f"{actor[:3]}_z"
    index = min(len(one) - 1, max(1, len(one) // 2))
    figure.add_trace(go.Scatter(
        x=[one[x_column].iloc[index]],
        y=[one[z_column].iloc[index]],
        mode="markers",
        marker={"color": color, "size": 8, "symbol": symbol},
        showlegend=False,
        hoverinfo="skip",
        meta={"role": "direction_marker"},
    ))


def build_raw_figure(traces: pd.DataFrame, scenario: int) -> go.Figure:
    """Build an untransformed joint pedestrian/vehicle X-Z diagnostic."""
    description = scenario_description(scenario)
    figure = go.Figure()
    for recording_id, one in traces.groupby("recording_identifier", sort=True):
        one = one.sort_values("scenario_time")
        participant = str(one.participant_id.iloc[0])
        custom = np.column_stack([
            np.repeat(participant, len(one)),
            np.repeat(description, len(one)),
            one.scenario_time,
            one.aligned_time,
        ])
        for actor, x_column, z_column, color, dash, width in (
            ("pedestrian", "ped_x", "ped_z", "#2A6FBB", "solid", 2),
            ("vehicle", "veh_x", "veh_z", "#D1495B", "dash", 1.5),
        ):
            figure.add_trace(go.Scatter(
                x=one[x_column],
                y=one[z_column],
                mode="lines",
                name=actor.title(),
                legendgroup=actor,
                showlegend=not any(
                    trace.name == actor.title() for trace in figure.data
                ),
                line={"color": color, "dash": dash, "width": width},
                opacity=0.32,
                customdata=custom,
                meta={
                    "role": "participant",
                    "participant_id": participant,
                    "recording_identifier": recording_id,
                    "scenario": scenario,
                    "actor": actor,
                },
                hovertemplate=(
                    "Participant: %{customdata[0]}<br>"
                    "Condition: %{customdata[1]}<br>"
                    f"Actor: {actor}<br>"
                    "X: %{x:.3f} m<br>Z: %{y:.3f} m<br>"
                    "Scenario time: %{customdata[2]:.3f} s<br>"
                    "Seconds from pedestrian movement onset: "
                    "%{customdata[3]:.3f} s<extra></extra>"
                ),
            ))
            figure.add_trace(go.Scatter(
                x=[one[x_column].iloc[0]],
                y=[one[z_column].iloc[0]],
                mode="markers",
                marker={
                    "color": color, "size": 8,
                    "symbol": "circle" if actor == "pedestrian" else "square",
                },
                showlegend=False,
                hovertemplate=(
                    f"{participant}<br>{actor.title()} starting point"
                    "<br>X: %{x:.3f} m<br>Z: %{y:.3f} m<extra></extra>"
                ),
                meta={"role": "start_marker", "participant_id": participant},
            ))
        _add_direction_marker(figure, one, "pedestrian", "#2A6FBB", "triangle-up")
        _add_direction_marker(figure, one, "vehicle", "#D1495B", "triangle-up")
    figure.update_layout(
        title={
            "text": (
                f"<b>{description.replace(' — ', ' —<br>')}</b>"
                f"<br><span style='font-size:14px'>Scenario {scenario} · "
                "Raw pedestrian and vehicle X–Z trajectories</span>"
            ),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.985,
            "yanchor": "top",
            "font": {"size": 21},
        },
        template="plotly_white",
        height=820,
        hovermode="closest",
        xaxis={"title": "World X position (m)", "scaleanchor": "y", "scaleratio": 1},
        yaxis={"title": "World Z position (m)", "constrain": "domain"},
        legend={
            "orientation": "h", "x": 0, "xanchor": "left",
            "y": 1.02, "yanchor": "bottom",
        },
        margin={"l": 85, "r": 45, "t": 170, "b": 80},
    )
    return figure


def circular_range_degrees(values: pd.Series) -> float:
    """Return the smallest circular arc containing all directions."""
    radians = np.sort(np.mod(np.radians(values.dropna()), 2 * np.pi))
    if not len(radians):
        return math.nan
    gaps = np.diff(np.r_[radians, radians[0] + 2 * np.pi])
    return float(np.degrees(2 * np.pi - gaps.max()))


def summarize_coordinate_frames(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Quantify within-scenario start, direction, scale, and geometry agreement."""
    rows = []
    for scenario, one in diagnostics.groupby("scenario", sort=True):
        rows.append({
            "scenario": int(scenario),
            "description": scenario_description(int(scenario)),
            "recordings": len(one),
            "ped_start_x_range": float(one.ped_start_x.max() - one.ped_start_x.min()),
            "ped_start_z_range": float(one.ped_start_z.max() - one.ped_start_z.min()),
            "veh_start_x_range": float(one.veh_start_x.max() - one.veh_start_x.min()),
            "veh_start_z_range": float(one.veh_start_z.max() - one.veh_start_z.min()),
            "ped_direction_arc_deg": circular_range_degrees(one.ped_direction_deg),
            "veh_direction_arc_deg": circular_range_degrees(one.veh_direction_deg),
            "ped_displacement_median": float(one.ped_displacement.median()),
            "ped_displacement_iqr": float(
                one.ped_displacement.quantile(.75) - one.ped_displacement.quantile(.25)
            ),
            "veh_displacement_median": float(one.veh_displacement.median()),
            "veh_displacement_iqr": float(
                one.veh_displacement.quantile(.75) - one.veh_displacement.quantile(.25)
            ),
            "common_origin_supported": bool(
                one.ped_start_x.max() - one.ped_start_x.min() <= 2.0
                and one.ped_start_z.max() - one.ped_start_z.min() <= 2.0
                and one.veh_start_x.max() - one.veh_start_x.min() <= 5.0
                and one.veh_start_z.max() - one.veh_start_z.min() <= 5.0
            ),
        })
    return pd.DataFrame(rows)


def interaction_interval(one: pd.DataFrame, movement_threshold: float = 0.15) -> tuple[float, float]:
    """Return onset through the last observed pedestrian movement sample."""
    ordered = one.sort_values("scenario_time")
    onset_rows = ordered[ordered.aligned_time.ge(-1e-9)]
    start = float(onset_rows.scenario_time.iloc[0]) if len(onset_rows) else float(
        ordered.scenario_time.iloc[0]
    )
    moving = ordered[
        ordered.scenario_time.ge(start) & ordered.ped_speed.ge(movement_threshold)
    ]
    end = float(moving.scenario_time.iloc[-1]) if len(moving) else float(
        ordered.scenario_time.iloc[-1]
    )
    if end <= start:
        end = float(ordered.scenario_time.iloc[-1])
    if end <= start:
        raise ValueError(f"Nonpositive interaction interval for {ordered.recording_identifier.iloc[0]}")
    return start, end


def interpolate_observed(
    source_progress: np.ndarray, values: np.ndarray, target_progress: np.ndarray
) -> np.ndarray:
    """Interpolate only inside the finite observed progress range."""
    valid = np.isfinite(source_progress) & np.isfinite(values)
    x = source_progress[valid]
    y = values[valid]
    if len(x) < 2:
        return np.full(len(target_progress), np.nan)
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique, indices = np.unique(x, return_index=True)
    y = y[indices]
    return np.interp(target_progress, unique, y, left=np.nan, right=np.nan)


def normalize_trajectories(
    traces: pd.DataFrame, progress_points: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample the onset-to-last-movement interval without changing duration."""
    grid = np.linspace(0.0, 100.0, progress_points)
    rows: list[pd.DataFrame] = []
    intervals: list[dict[str, object]] = []
    for recording, one in traces.groupby("recording_identifier", sort=True):
        one = one.sort_values("scenario_time")
        start, end = interaction_interval(one)
        interval = one[one.scenario_time.between(start, end)].copy()
        progress = 100.0 * (interval.scenario_time.to_numpy() - start) / (end - start)
        result = pd.DataFrame({
            "participant_id": one.participant_id.iloc[0],
            "recording_identifier": recording,
            "scenario": int(one.scenario.iloc[0]),
            "progress_percent": grid,
            "ped_x": interpolate_observed(progress, interval.ped_x.to_numpy(), grid),
            "ped_z": interpolate_observed(progress, interval.ped_z.to_numpy(), grid),
            "veh_x": interpolate_observed(progress, interval.veh_x.to_numpy(), grid),
            "veh_z": interpolate_observed(progress, interval.veh_z.to_numpy(), grid),
        })
        rows.append(result)
        intervals.append({
            "participant_id": one.participant_id.iloc[0],
            "recording_identifier": recording,
            "scenario": int(one.scenario.iloc[0]),
            "interaction_start_time": start,
            "interaction_end_time": end,
            "interaction_duration": end - start,
        })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(intervals)


def path_summary(normalized: pd.DataFrame, actor: str) -> pd.DataFrame:
    """Return pointwise mean, median, and radial-deviation quantiles."""
    x_column, z_column = f"{actor}_x", f"{actor}_z"
    rows = []
    for progress, one in normalized.groupby("progress_percent", sort=True):
        mean_x, mean_z = float(one[x_column].mean()), float(one[z_column].mean())
        deviations = np.hypot(one[x_column] - mean_x, one[z_column] - mean_z)
        rows.append({
            "progress_percent": progress,
            "mean_x": mean_x,
            "mean_z": mean_z,
            "median_x": float(one[x_column].median()),
            "median_z": float(one[z_column].median()),
            "deviation_q25": float(deviations.quantile(.25)),
            "deviation_median": float(deviations.median()),
            "deviation_q75": float(deviations.quantile(.75)),
            "cov_xx": float(one[x_column].var(ddof=1)),
            "cov_xz": float(one[[x_column, z_column]].cov().iloc[0, 1]),
            "cov_zz": float(one[z_column].var(ddof=1)),
            "contributors": int(one[[x_column, z_column]].dropna().shape[0]),
        })
    return pd.DataFrame(rows)


def add_covariance_ellipses(
    figure: go.Figure, summary: pd.DataFrame, color: str
) -> None:
    """Add one-standard-deviation covariance ellipses at 20% increments."""
    for target in (0, 20, 40, 60, 80, 100):
        row = summary.iloc[(summary.progress_percent - target).abs().argmin()]
        covariance = np.array([[row.cov_xx, row.cov_xz], [row.cov_xz, row.cov_zz]])
        if not np.isfinite(covariance).all():
            continue
        values, vectors = np.linalg.eigh(covariance)
        values = np.maximum(values, 0)
        angles = np.linspace(0, 2 * np.pi, 80)
        ellipse = vectors @ (np.sqrt(values)[:, None] * np.vstack(
            [np.cos(angles), np.sin(angles)]
        ))
        figure.add_trace(go.Scatter(
            x=row.mean_x + ellipse[0], y=row.mean_z + ellipse[1],
            mode="lines", line={"color": color, "width": 1},
            fill="toself", fillcolor=color.replace(")", ",0.10)").replace("rgb", "rgba"),
            showlegend=False, hoverinfo="skip",
            meta={"role": "spatial_variability", "progress_percent": target},
        ))


def build_normalized_path_figure(
    normalized: pd.DataFrame, scenario: int, actor: str
) -> go.Figure:
    summary = path_summary(normalized, actor)
    color = "rgb(42,111,187)" if actor == "ped" else "rgb(209,73,91)"
    label = "Pedestrian" if actor == "ped" else "Vehicle"
    figure = go.Figure()
    for recording, one in normalized.groupby("recording_identifier", sort=True):
        figure.add_trace(go.Scatter(
            x=one[f"{actor}_x"], y=one[f"{actor}_z"], mode="lines",
            line={"color": color, "width": 1}, opacity=.2, showlegend=False,
            customdata=np.column_stack([one.participant_id, one.progress_percent]),
            meta={
                "role": "participant", "participant_id": one.participant_id.iloc[0],
                "recording_identifier": recording, "actor": label.lower(),
            },
            hovertemplate=(
                "Participant: %{customdata[0]}<br>"
                "Progress: %{customdata[1]:.0f}%<br>"
                "X: %{x:.3f} m<br>Z: %{y:.3f} m<extra></extra>"
            ),
        ))
    add_covariance_ellipses(figure, summary, color)
    figure.add_trace(go.Scatter(
        x=summary.mean_x, y=summary.mean_z, mode="lines",
        name=f"Mean {label.lower()} path",
        line={"color": color, "width": 4},
        customdata=summary.progress_percent,
        meta={"role": "group_summary"},
        hovertemplate="Progress: %{customdata:.0f}%<br>X: %{x:.3f} m<br>Z: %{y:.3f} m<extra></extra>",
    ))
    description = scenario_description(scenario)
    figure.update_layout(
        title={
            "text": f"<b>{description.replace(' — ', ' —<br>')}</b><br>"
                    f"<span style='font-size:14px'>Scenario {scenario} · "
                    f"Normalized-progress {label.lower()} paths</span>",
            "x": .5, "xanchor": "center", "y": .985, "yanchor": "top",
        },
        template="plotly_white", height=820,
        xaxis={"title": "World X position (m)", "scaleanchor": "y", "scaleratio": 1},
        yaxis={"title": "World Z position (m)"},
        margin={"l": 85, "r": 45, "t": 170, "b": 80},
    )
    return figure


def deviation_frame(
    normalized: pd.DataFrame, summary: pd.DataFrame, actor: str
) -> pd.DataFrame:
    merged = normalized.merge(
        summary[["progress_percent", "mean_x", "mean_z"]],
        on="progress_percent", how="left",
    )
    merged["deviation_m"] = np.hypot(
        merged[f"{actor}_x"] - merged.mean_x,
        merged[f"{actor}_z"] - merged.mean_z,
    )
    return merged


def build_deviation_figure(
    deviation: pd.DataFrame, scenario: int, actor_label: str
) -> go.Figure:
    figure = go.Figure()
    for recording, one in deviation.groupby("recording_identifier", sort=True):
        figure.add_trace(go.Scatter(
            x=one.progress_percent, y=one.deviation_m, mode="lines",
            line={"color": "#4C78A8", "width": 1}, opacity=.25,
            showlegend=False,
            customdata=one.participant_id,
            meta={
                "role": "participant", "participant_id": one.participant_id.iloc[0],
                "recording_identifier": recording, "actor": actor_label.lower(),
            },
            hovertemplate="Participant: %{customdata}<br>Progress: %{x:.0f}%<br>Deviation: %{y:.3f} m<extra></extra>",
        ))
    group = deviation.groupby("progress_percent").deviation_m
    median = group.median()
    q25, q75 = group.quantile(.25), group.quantile(.75)
    figure.add_trace(go.Scatter(
        x=q25.index, y=q25, mode="lines", line={"width": 0},
        showlegend=False, hoverinfo="skip", meta={"role": "variability_band"},
    ))
    figure.add_trace(go.Scatter(
        x=q75.index, y=q75, mode="lines", line={"width": 0},
        fill="tonexty", fillcolor="rgba(245,133,24,.22)",
        name="Interquartile range", hoverinfo="skip",
        meta={"role": "variability_band"},
    ))
    figure.add_trace(go.Scatter(
        x=median.index, y=median, mode="lines", name="Median deviation",
        line={"color": "#D1495B", "width": 3}, meta={"role": "group_summary"},
    ))
    description = scenario_description(scenario)
    figure.update_layout(
        title={
            "text": f"<b>{description.replace(' — ', ' —<br>')}</b><br>"
                    f"<span style='font-size:14px'>Scenario {scenario} · "
                    f"{actor_label} deviation from mean path</span>",
            "x": .5, "xanchor": "center", "y": .985, "yanchor": "top",
        },
        template="plotly_white", height=760,
        xaxis={"title": "Normalized interaction progress (%)"},
        yaxis={"title": "Distance from scenario mean path (m)", "rangemode": "tozero"},
        legend={"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
        margin={"l": 85, "r": 45, "t": 165, "b": 75},
    )
    return figure


def trajectory_metrics(
    normalized: pd.DataFrame, intervals: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    summaries = {actor: path_summary(normalized, actor) for actor in ("ped", "veh")}
    deviations = {
        actor: deviation_frame(normalized, summaries[actor], actor)
        for actor in ("ped", "veh")
    }
    for recording, interval in intervals.set_index("recording_identifier").iterrows():
        row: dict[str, object] = {
            "participant_id": interval.participant_id,
            "recording_identifier": recording,
            "scenario": int(interval.scenario),
            "original_interaction_duration": interval.interaction_duration,
            "data_quality_flags": "",
        }
        one_norm = normalized[normalized.recording_identifier.eq(recording)]
        for actor, label in (("ped", "pedestrian"), ("veh", "vehicle")):
            one = deviations[actor][deviations[actor].recording_identifier.eq(recording)]
            max_index = one.deviation_m.idxmax()
            row[f"{label}_mean_path_deviation"] = float(one.deviation_m.mean())
            row[f"{label}_median_path_deviation"] = float(one.deviation_m.median())
            row[f"{label}_maximum_path_deviation"] = float(one.deviation_m.max())
            row[f"{label}_progress_at_maximum_deviation"] = float(
                one.loc[max_index, "progress_percent"]
            )
            row[f"{label}_path_length"] = _path_length(
                one_norm[f"{actor}_x"], one_norm[f"{actor}_z"]
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    outlier_columns = [
        "pedestrian_maximum_path_deviation", "vehicle_maximum_path_deviation",
        "pedestrian_path_length", "vehicle_path_length",
        "original_interaction_duration",
    ]
    flags = {index: [] for index in result.index}
    for column in outlier_columns:
        q25, q75 = result[column].quantile([.25, .75])
        iqr = q75 - q25
        lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        for index in result.index[result[column].lt(lower) | result[column].gt(upper)]:
            flags[index].append(f"robust_outlier:{column}")
    result["data_quality_flags"] = [
        ";".join(flags[index]) for index in result.index
    ]
    return result


def run_full_trajectory_analysis(
    output_dir: Path, scenarios: list[int], progress_points: int
) -> pd.DataFrame:
    all_metrics = []
    validation = []
    for scenario in scenarios:
        scenario_dir = output_dir / f"scenario_{scenario:02d}"
        traces = pd.read_csv(scenario_dir / "raw_trajectory_samples.csv")
        normalized, intervals = normalize_trajectories(traces, progress_points)
        normalized.to_csv(scenario_dir / "normalized_trajectory_samples.csv", index=False)
        intervals.to_csv(scenario_dir / "interaction_intervals.csv", index=False)
        # Standardization is the identity transform after the common-frame audit.
        build_raw_figure(traces, scenario).write_html(
            scenario_dir / "standardized_joint_trajectories.html",
            include_plotlyjs=True, full_html=True, post_script=LINKED_CLICK_SCRIPT,
            config={"displaylogo": False, "responsive": True},
        )
        transformation = {
            "scenario": scenario,
            "description": scenario_description(scenario),
            "translation": {"x_m": 0.0, "z_m": 0.0},
            "rotation_degrees": 0.0,
            "scale": 1.0,
            "reflection": False,
            "reason": "Raw diagnostics support a shared within-scenario Unity world frame.",
            "distance_preservation_max_absolute_error_m": 0.0,
        }
        (scenario_dir / "transformation_metadata.json").write_text(
            json.dumps(transformation, indent=2) + "\n", encoding="utf-8"
        )
        for actor, filename in (
            ("ped", "normalized_pedestrian_paths.html"),
            ("veh", "normalized_vehicle_paths.html"),
        ):
            summary = path_summary(normalized, actor)
            summary.to_csv(scenario_dir / f"{actor}_mean_path.csv", index=False)
            build_normalized_path_figure(normalized, scenario, actor).write_html(
                scenario_dir / filename, include_plotlyjs=True, full_html=True,
                post_script=LINKED_CLICK_SCRIPT,
                config={"displaylogo": False, "responsive": True},
            )
            dev = deviation_frame(normalized, summary, actor)
            dev.to_csv(scenario_dir / f"{actor}_path_deviation.csv", index=False)
            build_deviation_figure(
                dev, scenario, "Pedestrian" if actor == "ped" else "Vehicle"
            ).write_html(
                scenario_dir / (
                    "pedestrian_deviation_from_mean.html"
                    if actor == "ped" else "vehicle_deviation_from_mean.html"
                ),
                include_plotlyjs=True, full_html=True,
                post_script=LINKED_CLICK_SCRIPT,
                config={"displaylogo": False, "responsive": True},
            )
        metrics = trajectory_metrics(normalized, intervals)
        metrics.to_csv(scenario_dir / "trajectory_metrics.csv", index=False)
        all_metrics.append(metrics)
        validation.append({
            "scenario": scenario,
            "recordings": int(metrics.recording_identifier.nunique()),
            "progress_points": progress_points,
            "transform": "identity",
            "duration_preserved": bool(np.allclose(
                intervals.interaction_end_time - intervals.interaction_start_time,
                intervals.interaction_duration,
            )),
            "no_extrapolation": bool(
                normalized.groupby("recording_identifier").progress_percent.agg(
                    ["min", "max"]
                ).eq([0.0, 100.0]).all().all()
            ),
        })
    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(output_dir / "combined_trajectory_summary.csv", index=False)
    (output_dir / "validation_summary.json").write_text(
        json.dumps({
            "normalized_interval": (
                "First smoothed pedestrian-speed sample >= 0.30 m/s through "
                "the final subsequent sample >= 0.15 m/s."
            ),
            "progress_grid": f"0–100% inclusive at {progress_points} points",
            "spatial_summary": "Pointwise arithmetic mean X/Z path.",
            "variability": (
                "Radial deviation quartiles plus one-standard-deviation X/Z "
                "covariance ellipses at 20% progress increments."
            ),
            "scenarios": validation,
        }, indent=2) + "\n", encoding="utf-8"
    )
    return combined


def run_raw_diagnostics(
    manifest_path: Path, output_dir: Path, scenarios: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    selected = manifest[
        manifest.scenario_number.isin(scenarios)
        & manifest.processing_status.eq("success")
    ].copy()
    preexisting_failures = manifest[
        manifest.scenario_number.isin(scenarios)
        & ~manifest.processing_status.eq("success")
    ][
        ["recording_identifier", "scenario_number", "processing_status",
         "failure_exclusion_reason"]
    ].to_dict(orient="records")
    absent = sorted(set(scenarios) - set(selected.scenario_number))
    if absent:
        raise ValueError(f"No successful recordings for scenarios: {absent}")
    config = PipelineConfig()
    all_traces: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        try:
            traces, diagnostic = load_recording(pd.Series(row._asdict()), config)
        except Exception as error:  # preserve every failed recording in validation output
            failures.append({
                "recording_identifier": row.recording_identifier,
                "scenario": int(row.scenario_number),
                "error": f"{type(error).__name__}: {error}",
            })
            continue
        all_traces.append(traces)
        diagnostics.append(asdict(diagnostic))
    if not all_traces:
        raise RuntimeError("No recordings could be loaded for trajectory diagnostics.")
    trace_frame = pd.concat(all_traces, ignore_index=True)
    diagnostic_frame = pd.DataFrame(diagnostics)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_frame.to_csv(output_dir / "coordinate_diagnostics.csv", index=False)
    frame_summary = summarize_coordinate_frames(diagnostic_frame)
    frame_summary.to_csv(output_dir / "coordinate_frame_summary.csv", index=False)
    for scenario in scenarios:
        scenario_dir = output_dir / f"scenario_{scenario:02d}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        one = trace_frame[trace_frame.scenario.eq(scenario)]
        one.to_csv(scenario_dir / "raw_trajectory_samples.csv", index=False)
        figure = build_raw_figure(one, scenario)
        figure.write_html(
            scenario_dir / "raw_joint_trajectories.html",
            include_plotlyjs=True,
            full_html=True,
            post_script=LINKED_CLICK_SCRIPT,
            config={"displaylogo": False, "responsive": True},
        )
    metadata = {
        "phase": "raw_coordinate_diagnostics",
        "scenarios": scenarios,
        "requested_recordings": int(len(selected)),
        "successful_recordings": int(len(diagnostic_frame)),
        "failed_recordings": failures,
        "preexisting_failed_or_excluded_recordings": preexisting_failures,
        "coordinate_columns": {
            "time": "ScenarioTime normalized internally as time",
            "pedestrian_x": "B VR Pos X (or decoded B VR Pos component 0)",
            "pedestrian_z": "B VR Pos Z (or decoded B VR Pos component 2)",
            "vehicle_x": "A car Pos X (or decoded A car Pos component 0)",
            "vehicle_z": "A car Pos Z (or decoded A car Pos component 2)",
            "pedestrian_speed": "ped_speed: 9-sample centered rolling mean",
            "vehicle_speed": "veh_speed: 9-sample centered rolling mean",
            "movement_onset": "first ped_speed >= 0.30 m/s; scenario-start fallback",
        },
        "transformations": [],
        "unit_statement": (
            "The existing pipeline treats Unity X/Z world-coordinate differences "
            "as meters. No independent calibration artifact was found in the repository."
        ),
        "frame_summary": frame_summary.to_dict(orient="records"),
    }
    (output_dir / "raw_validation_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return diagnostic_frame, frame_summary


def main() -> int:
    args = parse_args()
    unknown = sorted(set(args.scenarios) - set(SCENARIO_DESCRIPTIONS))
    if unknown:
        raise ValueError(f"Missing centralized descriptions for scenarios: {unknown}")
    if args.progress_points < 2:
        raise ValueError("--progress-points must be at least 2.")
    _, frame_summary = run_raw_diagnostics(
        args.input.resolve(), args.output_dir.resolve(), args.scenarios
    )
    print(frame_summary.to_string(index=False))
    if args.phase == "full" and not frame_summary.common_origin_supported.all():
        failed = frame_summary.loc[
            ~frame_summary.common_origin_supported, "scenario"
        ].tolist()
        raise RuntimeError(
            "Coordinate compatibility was not supported for scenarios "
            f"{failed}; stopping before standardization and mean trajectories."
        )
    if args.phase == "full":
        run_full_trajectory_analysis(
            args.output_dir.resolve(), args.scenarios, args.progress_points
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
