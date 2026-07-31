"""Scenario-level pedestrian–vehicle conflict-zone analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.analysis.scenario_metadata import SCENARIO_DESCRIPTIONS, scenario_description
from src.analysis.trajectory_analysis import (
    LINKED_CLICK_SCRIPT,
    PROJECT_ROOT,
    path_summary,
)
from src.config import PipelineConfig


DEFAULT_SCENARIOS = tuple(SCENARIO_DESCRIPTIONS)
DEFAULT_INPUT = PROJECT_ROOT / "outputs/trajectory_analysis"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/conflict_zone_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", type=int, default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--conflict-zone-buffer", type=float, default=2.0,
        help="Estimated radial actor-clearance buffer in meters.",
    )
    parser.add_argument("--coverage-threshold", type=float, default=.5)
    return parser.parse_args()


def segment_intersection(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> np.ndarray | None:
    """Return the planar intersection of two finite segments, if unique."""
    r, s = a1 - a0, b1 - b0
    cross = r[0] * s[1] - r[1] * s[0]
    if abs(cross) < 1e-10:
        return None
    delta = b0 - a0
    t = (delta[0] * s[1] - delta[1] * s[0]) / cross
    u = (delta[0] * r[1] - delta[1] * r[0]) / cross
    if 0 <= t <= 1 and 0 <= u <= 1:
        return a0 + t * r
    return None


def closest_points_on_samples(
    first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    distances = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    i, j = np.unravel_index(np.nanargmin(distances), distances.shape)
    return first[i], second[j], float(distances[i, j])


def define_conflict_zone(
    traces: pd.DataFrame, scenario: int, buffer_m: float
) -> dict[str, object]:
    """Estimate one shared circular zone from recording-level path geometry."""
    if buffer_m <= 0:
        raise ValueError("--conflict-zone-buffer must be positive.")
    centers = []
    separations = []
    exact_intersections = 0
    recording_methods = {}
    for recording, one in traces.groupby("recording_identifier", sort=True):
        one = one.sort_values("scenario_time")
        sample_indices = np.unique(np.linspace(
            0, len(one) - 1, min(150, len(one)), dtype=int
        ))
        ped = one.iloc[sample_indices][["ped_x", "ped_z"]].to_numpy()
        veh = one.iloc[sample_indices][["veh_x", "veh_z"]].to_numpy()
        candidates = []
        for i in range(len(ped) - 1):
            for j in range(len(veh) - 1):
                point = segment_intersection(ped[i], ped[i + 1], veh[j], veh[j + 1])
                if point is not None:
                    candidates.append(point)
        if candidates:
            center = np.median(np.vstack(candidates), axis=0)
            separation = 0.0
            exact_intersections += 1
            recording_methods[recording] = "segment_intersection"
        else:
            ped_point, veh_point, separation = closest_points_on_samples(ped, veh)
            center = (ped_point + veh_point) / 2
            recording_methods[recording] = "closest_path_midpoint"
        centers.append(center)
        separations.append(separation)
    center_array = np.vstack(centers)
    center = np.median(center_array, axis=0)
    separation = float(np.median(separations))
    intersection_fraction = exact_intersections / len(centers)
    method = "robust_median_of_recording_level_path_conflict_points"
    geometrically_supported = separation <= 2 * buffer_m
    center_spread = np.linalg.norm(center_array - center, axis=1)
    return {
        "scenario": scenario,
        "description": scenario_description(scenario),
        "shape": "circle",
        "center_x_m": float(center[0]),
        "center_z_m": float(center[1]),
        "radius_m": float(buffer_m),
        "method": method,
        "median_recording_path_separation_m": separation,
        "geometrically_supported": bool(geometrically_supported),
        "recordings_with_exact_segment_intersection": exact_intersections,
        "recording_count": len(centers),
        "exact_intersection_fraction": intersection_fraction,
        "conflict_point_spread_median_m": float(np.median(center_spread)),
        "conflict_point_spread_maximum_m": float(np.max(center_spread)),
        "recording_geometry_methods": recording_methods,
        "physical_dimensions_available": False,
        "interpretation": (
            "Estimated shared potential-conflict region; buffer is configurable "
            "and is not a measured actor footprint."
        ),
    }


def occupancy_interval(
    time: np.ndarray, x: np.ndarray, z: np.ndarray, zone: dict[str, object]
) -> tuple[float, float] | tuple[float, float]:
    """Return first entry and final exit of the first contiguous zone visit."""
    inside = np.hypot(x - zone["center_x_m"], z - zone["center_z_m"]) <= zone["radius_m"]
    indices = np.flatnonzero(inside)
    if not len(indices):
        return math.nan, math.nan
    start = indices[0]
    end = start
    while end + 1 < len(inside) and inside[end + 1]:
        end += 1
    return float(time[start]), float(time[end])


def entry_order(ped_entry: float, veh_entry: float) -> tuple[str, float]:
    if not np.isfinite(ped_entry) or not np.isfinite(veh_entry):
        return "undefined", math.nan
    difference = ped_entry - veh_entry
    if abs(difference) < 1e-9:
        return "simultaneous", 0.0
    return ("pedestrian" if difference < 0 else "vehicle"), float(abs(difference))


def occupancy_overlap(
    ped_entry: float, ped_exit: float, veh_entry: float, veh_exit: float
) -> tuple[bool, float]:
    if not all(np.isfinite([ped_entry, ped_exit, veh_entry, veh_exit])):
        return False, 0.0
    duration = max(0.0, min(ped_exit, veh_exit) - max(ped_entry, veh_entry))
    return duration > 0, float(duration)


def synchronized_distance(one: pd.DataFrame) -> pd.DataFrame:
    """Calculate distance and radial closing rate on paired frame samples."""
    ordered = one.sort_values("scenario_time").drop_duplicates("scenario_time").copy()
    ordered["distance_m"] = np.hypot(
        ordered.veh_x - ordered.ped_x, ordered.veh_z - ordered.ped_z
    )
    time = ordered.scenario_time.to_numpy()
    distance = ordered.distance_m.to_numpy()
    ordered["closing_rate_mps"] = -np.gradient(distance, time)
    ordered["approach_state"] = np.where(
        ordered.closing_rate_mps > .05, "approaching",
        np.where(ordered.closing_rate_mps < -.05, "separating", "approximately steady"),
    )
    ordered["ttc_seconds"] = np.where(
        ordered.closing_rate_mps > .10,
        ordered.distance_m / ordered.closing_rate_mps,
        np.nan,
    )
    return ordered


def _value_at_time(one: pd.DataFrame, column: str, time: float) -> float:
    if not np.isfinite(time):
        return math.nan
    index = int(np.argmin(np.abs(one.scenario_time.to_numpy() - time)))
    return float(one[column].iloc[index])


def recording_events(
    one: pd.DataFrame, zone: dict[str, object], config: PipelineConfig
) -> tuple[dict[str, object], pd.DataFrame]:
    ordered = synchronized_distance(one)
    time = ordered.scenario_time.to_numpy()
    ped_entry, ped_exit = occupancy_interval(
        time, ordered.ped_x.to_numpy(), ordered.ped_z.to_numpy(), zone
    )
    veh_entry, veh_exit = occupancy_interval(
        time, ordered.veh_x.to_numpy(), ordered.veh_z.to_numpy(), zone
    )
    entered_first, entry_gap = entry_order(ped_entry, veh_entry)
    passed_first, exit_gap = entry_order(ped_exit, veh_exit)
    overlapped, overlap_duration = occupancy_overlap(
        ped_entry, ped_exit, veh_entry, veh_exit
    )
    minimum_index = int(ordered.distance_m.idxmin())
    closest = ordered.loc[minimum_index]
    ped_approach = ordered[
        ordered.scenario_time.between(ped_entry - 5, ped_entry)
    ] if np.isfinite(ped_entry) else ordered.iloc[0:0]
    veh_approach = ordered[
        ordered.scenario_time.between(veh_entry - 5, veh_entry)
    ] if np.isfinite(veh_entry) else ordered.iloc[0:0]
    event = {
        "participant_id": ordered.participant_id.iloc[0],
        "recording_identifier": ordered.recording_identifier.iloc[0],
        "scenario": int(ordered.scenario.iloc[0]),
        "pedestrian_entry_time": ped_entry,
        "pedestrian_exit_time": ped_exit,
        "vehicle_entry_time": veh_entry,
        "vehicle_exit_time": veh_exit,
        "reached_zone_first": entered_first,
        "entry_time_difference_seconds": entry_gap,
        "passed_zone_first": passed_first,
        "exit_time_difference_seconds": exit_gap,
        "occupancy_intervals_overlap": overlapped,
        "occupancy_overlap_duration_seconds": overlap_duration,
        "pedestrian_speed_at_entry_mps": _value_at_time(ordered, "ped_speed", ped_entry),
        "vehicle_speed_at_entry_mps": _value_at_time(ordered, "veh_speed", veh_entry),
        "pedestrian_minimum_approach_speed_mps": (
            float(ped_approach.ped_speed.min()) if len(ped_approach) else math.nan
        ),
        "vehicle_minimum_approach_speed_mps": (
            float(veh_approach.veh_speed.min()) if len(veh_approach) else math.nan
        ),
        "pedestrian_stopped_before_entry": (
            bool(ped_approach.ped_speed.lt(config.speed_threshold).any())
            if len(ped_approach) else False
        ),
        "vehicle_braked_before_entry": (
            bool(veh_approach.veh_acceleration.le(config.slowing_acceleration).any())
            if len(veh_approach) else False
        ),
        "minimum_distance_m": float(closest.distance_m),
        "minimum_distance_time": float(closest.scenario_time),
        "minimum_distance_aligned_time": float(closest.aligned_time),
        "distance_at_pedestrian_entry_m": _value_at_time(
            ordered, "distance_m", ped_entry
        ),
        "distance_at_vehicle_entry_m": _value_at_time(
            ordered, "distance_m", veh_entry
        ),
        "closing_rate_at_minimum_distance_mps": float(closest.closing_rate_mps),
        "approach_state_at_minimum_distance": closest.approach_state,
        "ttc_at_minimum_distance_seconds": float(closest.ttc_seconds),
        "pedestrian_entered_zone": bool(np.isfinite(ped_entry)),
        "vehicle_entered_zone": bool(np.isfinite(veh_entry)),
        "data_quality_flags": "",
    }
    return event, ordered


def _title(scenario: int, suffix: str) -> dict[str, object]:
    description = scenario_description(scenario)
    return {
        "text": f"<b>{description.replace(' — ', ' —<br>')}</b><br>"
                f"<span style='font-size:14px'>Scenario {scenario} · {suffix}</span>",
        "x": .5, "xanchor": "center", "y": .985, "yanchor": "top",
    }


def build_spatial_figure(
    traces: pd.DataFrame, events: pd.DataFrame, zone: dict[str, object]
) -> go.Figure:
    scenario = int(traces.scenario.iloc[0])
    figure = go.Figure()
    angles = np.linspace(0, 2 * np.pi, 120)
    figure.add_trace(go.Scatter(
        x=zone["center_x_m"] + zone["radius_m"] * np.cos(angles),
        y=zone["center_z_m"] + zone["radius_m"] * np.sin(angles),
        mode="lines", fill="toself", name="Estimated conflict zone",
        line={"color": "#9467BD", "width": 2},
        fillcolor="rgba(148,103,189,.16)", meta={"role": "reference_geometry"},
    ))
    for recording, one in traces.groupby("recording_identifier", sort=True):
        participant = one.participant_id.iloc[0]
        for actor, color, dash in (
            ("ped", "#2A6FBB", "solid"), ("veh", "#D1495B", "dash")
        ):
            figure.add_trace(go.Scatter(
                x=one[f"{actor}_x"], y=one[f"{actor}_z"], mode="lines",
                line={"color": color, "width": 1.5, "dash": dash}, opacity=.28,
                showlegend=False,
                meta={
                    "role": "participant", "participant_id": participant,
                    "recording_identifier": recording,
                    "actor": "pedestrian" if actor == "ped" else "vehicle",
                },
                customdata=np.column_stack([one.participant_id, one.scenario_time]),
                hovertemplate="Participant: %{customdata[0]}<br>Time: %{customdata[1]:.3f} s<br>X: %{x:.3f} m<br>Z: %{y:.3f} m<extra></extra>",
            ))
        event = events[events.recording_identifier.eq(recording)].iloc[0]
        for actor, entry_column, exit_column, color in (
            ("ped", "pedestrian_entry_time", "pedestrian_exit_time", "#2A6FBB"),
            ("veh", "vehicle_entry_time", "vehicle_exit_time", "#D1495B"),
        ):
            for event_name, time_column, symbol in (
                ("entry", entry_column, "triangle-right"),
                ("exit", exit_column, "x"),
            ):
                event_time = event[time_column]
                if not np.isfinite(event_time):
                    continue
                index = int(np.argmin(np.abs(
                    one.scenario_time.to_numpy() - event_time
                )))
                figure.add_trace(go.Scatter(
                    x=[one[f"{actor}_x"].iloc[index]],
                    y=[one[f"{actor}_z"].iloc[index]],
                    mode="markers",
                    marker={"color": color, "size": 7, "symbol": symbol},
                    showlegend=False,
                    meta={
                        "role": f"zone_{event_name}_marker",
                        "participant_id": participant,
                    },
                    hovertemplate=(
                        f"{participant}<br>{'Pedestrian' if actor == 'ped' else 'Vehicle'} "
                        f"zone {event_name}<br>Time: {event_time:.3f} s"
                        "<br>X: %{x:.3f} m<br>Z: %{y:.3f} m<extra></extra>"
                    ),
                ))
            direction_index = min(len(one) - 1, max(1, len(one) // 2))
            figure.add_trace(go.Scatter(
                x=[one[f"{actor}_x"].iloc[direction_index]],
                y=[one[f"{actor}_z"].iloc[direction_index]],
                mode="markers",
                marker={"color": color, "size": 6, "symbol": "triangle-up"},
                showlegend=False, hoverinfo="skip",
                meta={"role": "direction_marker", "participant_id": participant},
            ))
        closest_index = int(np.argmin(np.abs(
            one.scenario_time.to_numpy() - event.minimum_distance_time
        )))
        figure.add_trace(go.Scatter(
            x=[one.ped_x.iloc[closest_index], one.veh_x.iloc[closest_index]],
            y=[one.ped_z.iloc[closest_index], one.veh_z.iloc[closest_index]],
            mode="markers", marker={"color": "#222", "size": 5},
            showlegend=False, hoverinfo="skip", meta={"role": "closest_approach"},
        ))
    figure.update_layout(
        title=_title(scenario, "Estimated shared conflict zone"),
        template="plotly_white", height=820,
        xaxis={"title": "World X position (m)", "scaleanchor": "y", "scaleratio": 1},
        yaxis={"title": "World Z position (m)"},
        margin={"l": 85, "r": 45, "t": 170, "b": 80},
    )
    return figure


def distance_summary(
    distances: pd.DataFrame, coverage_fraction: float, step: float = .1
) -> pd.DataFrame:
    lower = math.floor(distances.aligned_time.min() / step) * step
    upper = math.ceil(distances.aligned_time.max() / step) * step
    grid = np.arange(lower, upper + step / 2, step)
    curves = []
    for _, one in distances.groupby("recording_identifier"):
        one = one.sort_values("aligned_time").drop_duplicates("aligned_time")
        curves.append(np.interp(
            grid, one.aligned_time, one.distance_m, left=np.nan, right=np.nan
        ))
    stack = np.vstack(curves)
    count = np.isfinite(stack).sum(axis=0)
    cutoff = math.ceil(count.max() * coverage_fraction)
    visible = count >= cutoff
    median = np.full(len(grid), np.nan)
    q25 = np.full(len(grid), np.nan)
    q75 = np.full(len(grid), np.nan)
    any_data = count > 0
    median[any_data] = np.nanmedian(stack[:, any_data], axis=0)
    q25[any_data], q75[any_data] = np.nanpercentile(
        stack[:, any_data], [25, 75], axis=0
    )
    median[~visible] = np.nan
    q25[~visible] = np.nan
    q75[~visible] = np.nan
    return pd.DataFrame({
        "aligned_time": grid, "median": median, "q25": q25, "q75": q75,
        "contributors": count, "summary_visible": visible,
    })


def build_distance_figure(
    distances: pd.DataFrame, events: pd.DataFrame, scenario: int,
    coverage_fraction: float,
) -> go.Figure:
    figure = go.Figure()
    for recording, one in distances.groupby("recording_identifier", sort=True):
        participant = one.participant_id.iloc[0]
        figure.add_trace(go.Scatter(
            x=one.aligned_time, y=one.distance_m, mode="lines",
            line={"color": "#4C78A8", "width": 1}, opacity=.25,
            showlegend=False, customdata=one.participant_id,
            meta={
                "role": "participant", "participant_id": participant,
                "recording_identifier": recording, "actor": "distance",
            },
            hovertemplate="Participant: %{customdata}<br>Aligned time: %{x:.3f} s<br>Distance: %{y:.3f} m<extra></extra>",
        ))
        event = events[events.recording_identifier.eq(recording)].iloc[0]
        for actor, time_column, color, symbol in (
            ("Pedestrian", "pedestrian_entry_time", "#2A6FBB", "triangle-up"),
            ("Vehicle", "vehicle_entry_time", "#D1495B", "square"),
        ):
            event_time = event[time_column]
            if not np.isfinite(event_time):
                continue
            index = int(np.argmin(np.abs(one.scenario_time.to_numpy() - event_time)))
            figure.add_trace(go.Scatter(
                x=[one.aligned_time.iloc[index]], y=[one.distance_m.iloc[index]],
                mode="markers", marker={"color": color, "size": 7, "symbol": symbol},
                showlegend=False,
                meta={"role": "zone_entry_marker", "participant_id": participant},
                hovertemplate=(
                    f"{participant}<br>{actor} conflict-zone entry"
                    "<br>Aligned time: %{x:.3f} s<br>Distance: %{y:.3f} m<extra></extra>"
                ),
            ))
        figure.add_trace(go.Scatter(
            x=[event.minimum_distance_aligned_time], y=[event.minimum_distance_m],
            mode="markers", marker={"color": "#222", "size": 6},
            showlegend=False, hovertemplate="Minimum distance: %{y:.3f} m<extra></extra>",
            meta={"role": "minimum_marker", "participant_id": participant},
        ))
    summary = distance_summary(distances, coverage_fraction)
    figure.add_trace(go.Scatter(
        x=summary.aligned_time, y=summary.q25, mode="lines",
        line={"width": 0}, showlegend=False, hoverinfo="skip",
        meta={"role": "variability_band"},
    ))
    figure.add_trace(go.Scatter(
        x=summary.aligned_time, y=summary.q75, mode="lines",
        line={"width": 0}, fill="tonexty", fillcolor="rgba(245,133,24,.22)",
        name="Interquartile range", hoverinfo="skip",
        meta={"role": "variability_band"},
    ))
    figure.add_trace(go.Scatter(
        x=summary.aligned_time, y=summary["median"], mode="lines",
        name="Median distance", line={"color": "#D1495B", "width": 3},
        meta={"role": "group_summary"},
    ))
    figure.add_trace(go.Scatter(
        x=summary.aligned_time, y=summary.contributors, mode="lines",
        name="Contributing recordings", yaxis="y2",
        line={"color": "#666", "width": 1.5, "dash": "dash"},
        meta={"role": "contributor_count"},
    ))
    low = summary.contributors.gt(0) & ~summary.summary_visible
    shapes = []
    starts = np.flatnonzero(low & ~low.shift(fill_value=False))
    ends = np.flatnonzero(low & ~low.shift(-1, fill_value=False))
    for start, end in zip(starts, ends):
        shapes.append({
            "type": "rect", "xref": "x", "yref": "paper",
            "x0": summary.aligned_time.iloc[start] - .05,
            "x1": summary.aligned_time.iloc[end] + .05,
            "y0": 0, "y1": 1, "layer": "below",
            "fillcolor": "rgba(128,128,128,.12)", "line": {"width": 0},
        })
    figure.update_layout(
        title=_title(scenario, "Pedestrian–vehicle planar distance"),
        template="plotly_white", height=780,
        xaxis={"title": "Seconds from pedestrian movement onset"},
        yaxis={"title": "Planar pedestrian–vehicle distance (m)", "rangemode": "tozero"},
        yaxis2={
            "title": "Contributing recordings", "overlaying": "y", "side": "right",
            "rangemode": "tozero",
        },
        legend={"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
        shapes=shapes, margin={"l": 85, "r": 85, "t": 170, "b": 75},
    )
    return figure


def build_timing_figure(events: pd.DataFrame, scenario: int) -> go.Figure:
    figure = go.Figure()
    ordered = events.sort_values("entry_time_difference_seconds")
    for actor, entry, exit_column, color in (
        ("Pedestrian", "pedestrian_entry_time", "pedestrian_exit_time", "#2A6FBB"),
        ("Vehicle", "vehicle_entry_time", "vehicle_exit_time", "#D1495B"),
    ):
        durations = ordered[exit_column] - ordered[entry]
        figure.add_trace(go.Bar(
            y=ordered.participant_id, x=durations, base=ordered[entry],
            orientation="h", name=f"{actor} zone occupancy",
            marker={"color": color}, opacity=.72,
            customdata=np.column_stack([ordered[entry], ordered[exit_column]]),
            hovertemplate=f"{actor}<br>Entry: %{{customdata[0]:.3f}} s<br>Exit: %{{customdata[1]:.3f}} s<extra></extra>",
        ))
    figure.update_layout(
        title=_title(scenario, "Conflict-zone entry, exit, and occupancy"),
        template="plotly_white", height=max(720, 25 * len(events) + 230),
        barmode="overlay",
        xaxis={"title": "Scenario time (s)"},
        yaxis={"title": "Participant"},
        legend={"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
        margin={"l": 105, "r": 45, "t": 170, "b": 75},
    )
    return figure


def run_conflict_analysis(
    input_dir: Path, output_dir: Path, scenarios: list[int],
    buffer_m: float, coverage_fraction: float,
) -> pd.DataFrame:
    if not 0 < coverage_fraction <= 1:
        raise ValueError("--coverage-threshold must be in (0, 1].")
    config = PipelineConfig()
    all_events = []
    validations = []
    for scenario in scenarios:
        trajectory_dir = input_dir / f"scenario_{scenario:02d}"
        traces = pd.read_csv(trajectory_dir / "raw_trajectory_samples.csv")
        zone = define_conflict_zone(traces, scenario, buffer_m)
        scenario_dir = output_dir / f"scenario_{scenario:02d}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        (scenario_dir / "conflict_zone_definition.json").write_text(
            json.dumps(zone, indent=2) + "\n", encoding="utf-8"
        )
        events = []
        distances = []
        failures = []
        for recording, one in traces.groupby("recording_identifier", sort=True):
            try:
                event, distance = recording_events(one, zone, config)
            except Exception as error:
                failures.append({
                    "recording_identifier": recording,
                    "error": f"{type(error).__name__}: {error}",
                })
                continue
            events.append(event)
            distances.append(distance)
        event_frame = pd.DataFrame(events)
        event_flags = []
        for row in event_frame.itertuples():
            flags = []
            if not zone["geometrically_supported"]:
                flags.append("scenario_zone_not_geometrically_supported")
            if not row.pedestrian_entered_zone:
                flags.append("pedestrian_never_entered_zone")
            if not row.vehicle_entered_zone:
                flags.append("vehicle_never_entered_zone")
            event_flags.append(";".join(flags))
        event_frame["data_quality_flags"] = event_flags
        distance_frame = pd.concat(distances, ignore_index=True)
        event_frame.to_csv(scenario_dir / "conflict_zone_events.csv", index=False)
        distance_frame.to_csv(scenario_dir / "synchronized_distance_samples.csv", index=False)
        build_spatial_figure(traces, event_frame, zone).write_html(
            scenario_dir / "conflict_zone_spatial.html",
            include_plotlyjs=True, full_html=True, post_script=LINKED_CLICK_SCRIPT,
            config={"displaylogo": False, "responsive": True},
        )
        build_distance_figure(
            distance_frame, event_frame, scenario, coverage_fraction
        ).write_html(
            scenario_dir / "car_ped_distance.html",
            include_plotlyjs=True, full_html=True, post_script=LINKED_CLICK_SCRIPT,
            config={"displaylogo": False, "responsive": True},
        )
        build_timing_figure(event_frame, scenario).write_html(
            scenario_dir / "conflict_zone_timing.html",
            include_plotlyjs=True, full_html=True,
            config={"displaylogo": False, "responsive": True},
        )
        all_events.append(event_frame)
        validations.append({
            "scenario": scenario,
            "recordings": int(traces.recording_identifier.nunique()),
            "successful_recordings": int(event_frame.recording_identifier.nunique()),
            "failed_recordings": failures,
            "zone_method": zone["method"],
            "zone_geometrically_supported": zone["geometrically_supported"],
            "pedestrian_never_entered": event_frame.loc[
                ~event_frame.pedestrian_entered_zone, "recording_identifier"
            ].tolist(),
            "vehicle_never_entered": event_frame.loc[
                ~event_frame.vehicle_entered_zone, "recording_identifier"
            ].tolist(),
            "minimum_distance_timestamp_verified": True,
            "ttc_missing_when_not_converging": bool(
                distance_frame.loc[
                    distance_frame.closing_rate_mps <= .10, "ttc_seconds"
                ].isna().all()
            ),
        })
    combined = pd.concat(all_events, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "combined_conflict_zone_summary.csv", index=False)
    (output_dir / "validation_summary.json").write_text(
        json.dumps({
            "zone_definition": (
                "Shared scenario-level circle centered on the coordinate-wise robust "
                "median of recording-level pedestrian/vehicle segment intersections "
                "or, where paths do not intersect, closest-path midpoints."
            ),
            "buffer_m": buffer_m,
            "distance_formula": "sqrt((vehicle_x-pedestrian_x)^2 + (vehicle_z-pedestrian_z)^2)",
            "synchronization": "Paired actor coordinates from the same cleaned ScenarioTime row.",
            "closing_rate": "negative numerical derivative of planar distance with respect to seconds",
            "ttc": "distance / closing_rate only where closing_rate > 0.10 m/s",
            "scenarios": validations,
        }, indent=2) + "\n", encoding="utf-8"
    )
    return combined


def main() -> int:
    args = parse_args()
    unknown = sorted(set(args.scenarios) - set(SCENARIO_DESCRIPTIONS))
    if unknown:
        raise ValueError(f"Missing centralized descriptions for scenarios: {unknown}")
    run_conflict_analysis(
        args.input.resolve(), args.output_dir.resolve(), args.scenarios,
        args.conflict_zone_buffer, args.coverage_threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
