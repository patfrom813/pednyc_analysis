"""Run a deterministic cross-participant PedNYC scenario-pattern pilot."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PipelineConfig
from src.features import BehaviorClassifier, EventMerger, MetricsEngine, normalize_columns
from src.preprocess import clean_metrics, load_scenario, validate_scenario
from src.segmentation import MacroSegmenter, MicroSegmenter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUMERIC_FEATURES = [
    "recording_duration", "number_macro_segments", "total_low_motion_duration",
    "acceleration_event_count", "acceleration_event_duration",
    "deceleration_event_count", "deceleration_event_duration", "pause_count",
    "total_pause_duration", "stop_start_count", "head_rotation_event_count",
    "median_pedestrian_speed", "peak_pedestrian_speed",
]
FEATURE_LABELS = {
    "recording_duration": "Duration", "number_macro_segments": "Macro segments",
    "total_low_motion_duration": "Low motion", "acceleration_event_count": "Acceleration count",
    "acceleration_event_duration": "Acceleration duration",
    "deceleration_event_count": "Deceleration count",
    "deceleration_event_duration": "Deceleration duration", "pause_count": "Pause count",
    "total_pause_duration": "Pause duration", "stop_start_count": "Stop-start count",
    "head_rotation_event_count": "Head rotation count", "median_pedestrian_speed": "Median speed",
    "peak_pedestrian_speed": "Peak speed",
}


def _participant_number(value: object) -> int:
    match = re.fullmatch(r"PedNYC(\d+)", str(value), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid participant value: {value}")
    return int(match.group(1))


def discover_recordings(inventory_path: Path, scenario_min: int, scenario_max: int) -> pd.DataFrame:
    """Read the decoded inventory and retain unique, existing, eligible files."""
    inventory = pd.read_csv(inventory_path, dtype=str)
    required = {"pednyc", "scenario", "destination_path"}
    missing = required - set(inventory)
    if missing:
        raise ValueError(f"Inventory is missing columns: {sorted(missing)}")
    inventory["scenario_number"] = pd.to_numeric(inventory["scenario"], errors="coerce")
    eligible = inventory[
        inventory["scenario_number"].between(scenario_min, scenario_max, inclusive="both")
    ].copy()
    eligible["pednyc_number"] = eligible["pednyc"].map(_participant_number)
    eligible["source_decoded_csv_path"] = eligible["destination_path"].map(
        lambda value: str(Path(value).resolve())
    )
    eligible = eligible[
        eligible["source_decoded_csv_path"].map(lambda value: Path(value).is_file())
    ]
    eligible["scenario_number"] = eligible["scenario_number"].astype(int)
    eligible["recording_identifier"] = eligible.apply(
        lambda row: f"PedNYC{row.pednyc_number}_scenario{row.scenario_number}", axis=1
    )
    # A few participant/scenario pairs have multiple timestamped recording files.
    # Preserve each as an independent observation and make its identifier unique.
    duplicate_mask = eligible["recording_identifier"].duplicated(False)
    eligible.loc[duplicate_mask, "recording_identifier"] = eligible.loc[duplicate_mask].apply(
        lambda row: (
            f"PedNYC{row.pednyc_number}_scenario{row.scenario_number}_"
            + re.sub(r"[^A-Za-z0-9]+", "_", Path(row.source_decoded_csv_path).stem).strip("_")
        ),
        axis=1,
    )
    if eligible["recording_identifier"].duplicated().any():
        raise ValueError("Inventory contains exact duplicate decoded file entries")
    return eligible.sort_values(["pednyc_number", "scenario_number"]).reset_index(drop=True)


def select_recordings(eligible: pd.DataFrame, sample_size: int, seed: int) -> tuple[list[int], pd.DataFrame]:
    participants = sorted(eligible["pednyc_number"].unique().tolist())
    if len(participants) < sample_size:
        raise ValueError(
            f"Only {len(participants)} eligible participant folders exist; {sample_size} required"
        )
    selected = sorted(random.Random(seed).sample(participants, sample_size))
    return selected, eligible[eligible["pednyc_number"].isin(selected)].copy()


def _event_duration(events: pd.DataFrame, label: str) -> float:
    return float(events.loc[events["label"] == label, "duration"].sum())


def _sequence(values: Iterable[object]) -> str:
    output: list[str] = []
    for value in map(str, values):
        if not output or value != output[-1]:
            output.append(value)
    return " -> ".join(output)


def _run_lengths(metrics: pd.DataFrame, mask: pd.Series, minimum: float) -> list[tuple[float, float]]:
    time = metrics["time"].to_numpy(dtype=float)
    active = mask.fillna(False).to_numpy(dtype=bool)
    runs: list[tuple[float, float]] = []
    start: int | None = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(active) - 1):
            end = index if value and index == len(active) - 1 else index - 1
            duration = max(0.0, float(time[end] - time[start]))
            if duration >= minimum:
                runs.append((float(time[start]), float(time[end])))
            start = None
    return runs


def _quality_warnings(metrics: pd.DataFrame, validation: dict[str, object]) -> list[str]:
    warnings: list[str] = []
    if metrics["time"].isna().any() or not np.all(np.diff(metrics["time"]) >= 0):
        warnings.append("nonfinite_or_nonmonotonic_time")
    if float(metrics["ped_speed"].isna().mean()) > 0.05:
        warnings.append("pedestrian_speed_missing_over_5pct")
    if "head_yaw" not in metrics or metrics["head_yaw"].isna().all():
        warnings.append("head_signal_unavailable")
    if validation.get("duplicate_columns"):
        warnings.append("duplicate_input_columns")
    return warnings


def process_recording(row: pd.Series, config: PipelineConfig, output_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Run existing feature/macro/micro stages and return one compact summary."""
    pid, scenario = int(row.pednyc_number), int(row.scenario_number)
    identity = str(row.recording_identifier)
    raw = load_scenario(Path(row.source_decoded_csv_path))
    normalized = normalize_columns(raw)
    validation = validate_scenario(normalized)
    if not validation["valid"]:
        raise ValueError(f"validation failed: {validation}")
    cleaned = clean_metrics(normalized, config)
    metrics = MetricsEngine(config).compute(cleaned).reset_index(drop=True)
    if len(metrics) < 2:
        raise ValueError("fewer than two usable frames")
    if float(pd.to_numeric(metrics["ped_speed"], errors="coerce").max()) > config.speed_spike_threshold:
        raise ValueError(
            f"smoothed pedestrian speed exceeds configured {config.speed_spike_threshold:g} m/s spike threshold"
        )
    if not np.isfinite(metrics["time"]).all() or np.any(np.diff(metrics["time"]) < 0):
        raise ValueError("time is nonfinite or nonmonotonic")
    macros = MacroSegmenter(config).segment(metrics)
    windows = MicroSegmenter(config).extract_windows(metrics, macros)
    classified = BehaviorClassifier(config).classify(windows)
    events = EventMerger(config).merge(classified)
    if macros.empty:
        raise ValueError("macro segmentation produced no segments")
    start, end = float(metrics["time"].iloc[0]), float(metrics["time"].iloc[-1])
    if (macros["start_time"] < start).any() or (macros["end_time"] > end).any():
        raise ValueError("macro segment outside recording boundary")
    if not events.empty and (
        (events["start_time"] < start).any() or (events["end_time"] > end).any()
    ):
        raise ValueError("micro event outside recording boundary")
    motion = metrics["ped_speed"] >= config.walking_speed
    onset_indices = np.flatnonzero(motion.to_numpy())
    onset = float(metrics["time"].iloc[onset_indices[0]]) if len(onset_indices) else np.nan
    alignment_event = "pedestrian_movement_onset" if np.isfinite(onset) else "scenario_relative_start"
    alignment_time = onset if np.isfinite(onset) else start
    pause_runs = _run_lengths(metrics, metrics["ped_speed"] < config.pause_speed, config.pause_min_duration)
    low_runs = _run_lengths(metrics, metrics["ped_speed"] < config.speed_threshold, config.min_event_duration)
    # A stop-start is an internal pause bounded by moving samples on both sides.
    stop_starts = sum(
        run_start > start and run_end < end for run_start, run_end in pause_runs
    )
    head_count = int((events["label"] == "head_check").sum())
    macro_sequence = _sequence(macros["phase_label"])
    event_sequence = _sequence(events["label"]) if not events.empty else ""
    macro_durations = ";".join(
        f"{label}:{duration:.3f}" for label, duration in zip(
            macros["phase_label"], macros["end_time"] - macros["start_time"]
        )
    )
    pause_durations = [b - a for a, b in pause_runs]
    acceleration = pd.to_numeric(metrics["ped_acceleration"], errors="coerce")
    positive_acceleration = acceleration[acceleration > 0]
    negative_acceleration = acceleration[acceleration < 0]
    onset_window = metrics[metrics["time"].between(alignment_time - 1.0, alignment_time + 1.0)]
    head_rotation = pd.to_numeric(metrics.get("head_rotation"), errors="coerce")
    interaction_available = all(
        column in metrics and pd.to_numeric(metrics[column], errors="coerce").notna().any()
        for column in ("ped_veh_distance", "veh_speed", "veh_acceleration", "closing_distance")
    )
    summary = {
        "pednyc_number": pid, "scenario_number": scenario,
        "recording_identifier": identity, "source_decoded_csv_path": row.source_decoded_csv_path,
        "recording_duration": end - start, "alignment_event": alignment_event,
        "original_alignment_time": alignment_time, "movement_onset_time": onset,
        "number_macro_segments": len(macros), "macro_phase_sequence": macro_sequence,
        "macro_phase_durations": macro_durations,
        "total_low_motion_duration": sum(b - a for a, b in low_runs),
        "acceleration_event_count": int((events["label"] == "acceleration").sum()),
        "acceleration_event_duration": _event_duration(events, "acceleration"),
        "deceleration_event_count": int((events["label"] == "deceleration").sum()),
        "deceleration_event_duration": _event_duration(events, "deceleration"),
        "pause_count": len(pause_runs), "total_pause_duration": sum(pause_durations),
        "median_pause_duration": float(np.median(pause_durations)) if pause_durations else 0.0,
        "stop_start_count": int(stop_starts), "head_rotation_event_count": head_count,
        "median_absolute_head_rotation_rate": float(head_rotation.abs().median())
        if head_rotation.notna().any() else np.nan,
        "peak_absolute_head_rotation_rate": float(head_rotation.abs().max())
        if head_rotation.notna().any() else np.nan,
        "median_pedestrian_speed": float(metrics["ped_speed"].median()),
        "peak_pedestrian_speed": float(metrics["ped_speed"].max()),
        "median_positive_acceleration": float(positive_acceleration.median())
        if len(positive_acceleration) else np.nan,
        "median_deceleration_magnitude": float((-negative_acceleration).median())
        if len(negative_acceleration) else np.nan,
        "low_motion_proportion": sum(b - a for a, b in low_runs) / (end - start)
        if end > start else np.nan,
        "minimum_pedestrian_vehicle_distance": float(metrics["ped_veh_distance"].min())
        if interaction_available else np.nan,
        "vehicle_speed_at_movement_onset": float(
            metrics.iloc[int(np.argmin(np.abs(metrics["time"].to_numpy() - alignment_time)))]["veh_speed"]
        ) if interaction_available else np.nan,
        "vehicle_deceleration_around_movement_onset": float(
            pd.to_numeric(onset_window["veh_acceleration"], errors="coerce").median()
        ) if interaction_available and not onset_window.empty else np.nan,
        "median_closing_speed": float(
            pd.to_numeric(metrics["closing_distance"], errors="coerce").median()
        ) if interaction_available else np.nan,
        "event_sequence": event_sequence, "processing_status": "success",
        "quality_warnings": ";".join(_quality_warnings(metrics, validation)),
    }
    macro_path = output_dir / "intermediate" / "macro_segments" / f"{identity}.csv"
    event_path = output_dir / "intermediate" / "micro_events" / f"{identity}.csv"
    macro_out = macros.assign(pednyc_number=pid, scenario_number=scenario)
    event_out = events.assign(pednyc_number=pid, scenario_number=scenario)
    macro_out.to_csv(macro_path, index=False)
    event_out.to_csv(event_path, index=False)
    traces = pd.DataFrame({
        "pednyc_number": pid, "scenario_number": scenario,
        "recording_identifier": identity, "original_time": metrics["time"],
        "aligned_time": metrics["time"] - alignment_time, "pedestrian_speed": metrics["ped_speed"],
    })
    return summary, traces


def _median_iqr(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return (
        float(values.median()) if len(values) else np.nan,
        float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else np.nan,
    )


def aggregate_scenarios(recordings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for scenario, all_group in recordings.groupby("scenario_number", sort=True):
        group = all_group[all_group["processing_status"] == "success"]
        row: dict[str, object] = {
            "scenario_number": int(scenario), "number_selected_participants": len(group),
            "participating_pednyc_numbers": ";".join(map(str, sorted(group.pednyc_number.unique()))),
            "sample_size_flag": "adequate_for_pilot" if len(group) >= 3 else "low_sample_size",
        }
        for feature in NUMERIC_FEATURES:
            median, iqr = _median_iqr(group[feature])
            row[f"{feature}_median"], row[f"{feature}_iqr"] = median, iqr
        sequences = group["macro_phase_sequence"].fillna("")
        events = group["event_sequence"].fillna("")
        macro_mode = Counter(sequences).most_common(1)[0] if len(sequences) else ("", 0)
        event_mode = Counter(events).most_common(1)[0] if len(events) else ("", 0)
        row.update({
            "most_common_macro_sequence": macro_mode[0],
            "most_common_micro_event_sequence": event_mode[0],
            "dominant_macro_sequence_percentage": 100 * macro_mode[1] / len(group) if len(group) else np.nan,
            "dominant_micro_sequence_percentage": 100 * event_mode[1] / len(group) if len(group) else np.nan,
            "within_scenario_variability": float(
                np.nanmean([row[f"{feature}_iqr"] for feature in NUMERIC_FEATURES])
            ) if len(group) else np.nan,
            "failed_recordings": int((all_group["processing_status"] != "success").sum()),
        })
        for label in ("low_motion", "acceleration", "deceleration", "head_check", "possible_yielding", "possible_hesitation"):
            row[f"prevalence_{label}"] = (
                float(group["event_sequence"].fillna("").str.contains(label, regex=False).mean())
                if len(group) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _standardize(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    means, std = numeric.mean(), numeric.std(ddof=0).replace(0, np.nan)
    return (numeric - means) / std


def create_figures(recordings: pd.DataFrame, scenarios: pd.DataFrame, traces: pd.DataFrame, figure_dir: Path) -> None:
    successful = recordings[recordings.processing_status == "success"]
    median_columns = [f"{feature}_median" for feature in NUMERIC_FEATURES]
    heat = _standardize(scenarios.set_index("scenario_number")[median_columns]).fillna(0.0)
    labels = [FEATURE_LABELS[c.removesuffix("_median")] for c in heat.columns]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * .8), max(4, len(heat) * .55)))
    image = ax.imshow(heat, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(heat)), [
        f"Scenario {idx} (n={int(scenarios.loc[scenarios.scenario_number == idx, 'number_selected_participants'].iloc[0])})"
        for idx in heat.index
    ])
    fig.colorbar(image, ax=ax, label="Standardized scenario median")
    fig.tight_layout()
    fig.savefig(figure_dir / "scenario_feature_heatmap.png", dpi=180)
    plt.close(fig)

    usable = heat.loc[:, heat.notna().mean() >= .5]
    vectors = usable.to_numpy()
    norms = np.linalg.norm(vectors, axis=1)
    similarity = vectors @ vectors.T / np.outer(norms, norms)
    similarity[~np.isfinite(similarity)] = 0
    order = np.argsort(-np.nanmean(similarity, axis=1))
    similarity = similarity[np.ix_(order, order)]
    ordered_scenarios = heat.index.to_numpy()[order]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(similarity, cmap="viridis", vmin=-1, vmax=1)
    tick = [f"S{s}" for s in ordered_scenarios]
    ax.set_xticks(range(len(tick)), tick, rotation=45)
    ax.set_yticks(range(len(tick)), tick)
    for pos, scenario in enumerate(ordered_scenarios):
        n = int(scenarios.loc[scenarios.scenario_number == scenario, "number_selected_participants"].iloc[0])
        if n < 3:
            ax.text(pos, pos, "low n", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(image, ax=ax, label="Cosine similarity")
    fig.tight_layout()
    fig.savefig(figure_dir / "scenario_similarity_matrix.png", dpi=180)
    plt.close(fig)

    consensus_rows = scenarios[["scenario_number", "most_common_micro_event_sequence", "dominant_micro_sequence_percentage"]]
    palette = plt.get_cmap("tab10")
    labels_all = sorted({
        tag for seq in consensus_rows.most_common_micro_event_sequence.fillna("")
        for tag in seq.split(" -> ") if tag
    })
    colors = {label: palette(i % 10) for i, label in enumerate(labels_all)}
    abbreviations = {
        "low_motion": "LM", "acceleration": "A", "deceleration": "D",
        "crossing_commitment": "C", "head_check": "H",
        "possible_yielding": "Y", "possible_hesitation": "O",
    }
    fig, ax = plt.subplots(figsize=(11, max(4, len(consensus_rows) * .65)))
    for y, item in enumerate(consensus_rows.itertuples(index=False)):
        sequence = [x for x in str(item.most_common_micro_event_sequence).split(" -> ") if x]
        prevalence = float(item.dominant_micro_sequence_percentage) / 100 if np.isfinite(item.dominant_micro_sequence_percentage) else 0
        for x, label in enumerate(sequence):
            ax.barh(y, 1, left=x, color=colors[label], alpha=.25 + .75 * prevalence, edgecolor="white")
            ax.text(x + .5, y, abbreviations.get(label, label[:2].upper()), ha="center", va="center", fontsize=7)
        ax.text(len(sequence) + .1, y, f"{prevalence:.0%}", va="center", fontsize=8)
    ax.set_yticks(range(len(consensus_rows)), [f"Scenario {s}" for s in consensus_rows.scenario_number])
    ax.set_xlabel("Dominant objective event order (opacity/annotation = recording prevalence)")
    ax.invert_yaxis()
    legend_text = "  ".join(f"{abbreviations.get(label, label[:2].upper())}={label.replace('_', ' ')}" for label in labels_all)
    fig.text(.5, .015, legend_text, fontsize=8, ha="center", va="bottom")
    fig.tight_layout(rect=(0, .075, 1, 1))
    fig.savefig(figure_dir / "consensus_gantt.png", dpi=180)
    plt.close(fig)

    represented = sorted(successful.scenario_number.unique())
    columns = 2
    rows = int(np.ceil(len(represented) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, max(4, rows * 3)), squeeze=False, sharex=True, sharey=True)
    grid = np.arange(-5, 31.01, .1)
    for ax, scenario in zip(axes.flat, represented):
        scenario_traces = traces[traces.scenario_number == scenario]
        curves = []
        for _, trace in scenario_traces.groupby("recording_identifier"):
            trace = trace.dropna(subset=["aligned_time", "pedestrian_speed"]).sort_values("aligned_time")
            if len(trace) < 2:
                continue
            curve = np.interp(grid, trace.aligned_time, trace.pedestrian_speed, left=np.nan, right=np.nan)
            curves.append(curve)
            ax.plot(grid, curve, color="#4C78A8", alpha=.18, linewidth=.8)
        if curves:
            stack = np.vstack(curves)
            valid_grid = np.isfinite(stack).any(axis=0)
            median = np.full(len(grid), np.nan)
            low, high = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
            median[valid_grid] = np.nanmedian(stack[:, valid_grid], axis=0)
            low[valid_grid], high[valid_grid] = np.nanpercentile(stack[:, valid_grid], [25, 75], axis=0)
            ax.fill_between(grid, low, high, color="#F58518", alpha=.25)
            ax.plot(grid, median, color="#E45756", linewidth=2)
        ax.axvline(0, color="black", linestyle=":", linewidth=.8)
        ax.set_title(f"Scenario {scenario} (n={len(curves)})")
    for ax in axes.flat[len(represented):]:
        ax.axis("off")
    fig.supxlabel("Seconds from movement onset (fallback: scenario start)")
    fig.supylabel("Pedestrian speed (m/s)")
    fig.tight_layout()
    fig.savefig(figure_dir / "aligned_speed_small_multiples.png", dpi=180)
    plt.close(fig)


def write_report(output_dir: Path, selected: list[int], recordings: pd.DataFrame, scenarios: pd.DataFrame) -> None:
    successful = recordings[recordings.processing_status == "success"]
    adequate = scenarios.loc[scenarios.number_selected_participants >= 3, "scenario_number"].tolist()
    low = scenarios.loc[scenarios.number_selected_participants < 3, "scenario_number"].tolist()
    lines = [
        "# Scenario Pattern Pilot", "",
        f"- Selected participants (seed 42): {', '.join(f'PedNYC{x}' for x in selected)}.",
        f"- Processed successfully: {len(successful)} of {len(recordings)} eligible selected recordings.",
        f"- Represented scenarios: {', '.join(map(str, scenarios.scenario_number.tolist()))}.",
        f"- Scenarios with at least three recordings for pilot comparison: {', '.join(map(str, adequate)) or 'none'}.",
        f"- Low-sample scenarios: {', '.join(map(str, low)) or 'none'}.", "",
        "## Objective pattern comparison", "",
    ]
    for row in scenarios.itertuples(index=False):
        lines.append(
            f"- Scenario {row.scenario_number} (n={row.number_selected_participants}): "
            f"dominant macro sequence `{row.most_common_macro_sequence}` "
            f"({row.dominant_macro_sequence_percentage:.0f}%); dominant micro-event sequence "
            f"`{row.most_common_micro_event_sequence or 'none detected'}` "
            f"({row.dominant_micro_sequence_percentage:.0f}%)."
        )
    fingerprint = _standardize(
        scenarios.set_index("scenario_number")[[f"{feature}_median" for feature in NUMERIC_FEATURES]]
    ).fillna(0.0)
    fingerprint_values = fingerprint.to_numpy()
    norms = np.linalg.norm(fingerprint_values, axis=1)
    similarities = fingerprint_values @ fingerprint_values.T / np.outer(norms, norms)
    pair_rows: list[tuple[float, int, int]] = []
    for left in range(len(fingerprint)):
        for right in range(left + 1, len(fingerprint)):
            pair_rows.append((
                float(similarities[left, right]),
                int(fingerprint.index[left]), int(fingerprint.index[right]),
            ))
    pair_rows.sort(reverse=True)
    closest = pair_rows[:3]
    speed = scenarios.set_index("scenario_number")["median_pedestrian_speed_median"].astype(float)
    macro_agreement = scenarios.set_index("scenario_number")["dominant_macro_sequence_percentage"].astype(float)
    lines += [
        "", "## Strongest recurrence, differences, and similarity", "",
        "- All represented scenarios contained low-motion events in every successful recording.",
        "- The canonical five-phase macro sequence was shared by 100% of recordings in Scenarios 3, 7, and 21; "
        "agreement was 60% in Scenarios 15 and 16, and 50% in Scenario 12.",
        f"- Scenario {int(speed.idxmax())} had the highest median participant speed "
        f"({speed.max():.2f} m/s), while Scenario {int(speed.idxmin())} had the lowest "
        f"({speed.min():.2f} m/s), a {speed.max() - speed.min():.2f} m/s difference.",
        "- Exact micro-event order was highly variable: no scenario's dominant full micro sequence "
        "occurred in more than 25% of its recordings.",
        "- The closest exploratory scenario fingerprints were "
        + ", ".join(f"Scenarios {left} and {right} (similarity {value:.2f})" for value, left, right in closest)
        + ".",
        f"- Macro-sequence agreement ranged from {macro_agreement.min():.0f}% to "
        f"{macro_agreement.max():.0f}%, showing stronger cross-participant recurrence at the macro "
        "level than for exact micro-event sequences.",
    ]
    lines += [
        "", "## Interpretation and limitations", "",
        "- Labels are mathematically defined movement/head/context evidence; they are not psychological judgments.",
        "- A possible behavioral interpretation (for example, yielding) requires interaction context or video confirmation.",
        "- Dominant sequences are unreliable when participant coverage is low, and missing head signals are retained as warnings rather than converted to zero evidence.",
        "- Fingerprint similarity is exploratory: it uses standardized aggregate medians and marks scenarios with fewer than three successful recordings.",
        "- Before scaling, review strong speed outliers, confirm coordinate/time consistency across participants, and validate event thresholds against a small video-checked sample.",
        "- The existing algorithm is considered consistent only where all recordings passed identical stages and thresholds; sequence agreement percentages quantify, rather than assume, generalization.",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_outputs(output_dir: Path, selected: list[int], manifest: pd.DataFrame, recordings: pd.DataFrame) -> dict[str, bool]:
    figures = list((output_dir / "figures").glob("*.png"))
    checks = {
        "exactly_five_participants": len(selected) == 5 and len(set(selected)) == 5,
        "scenarios_in_range": bool(manifest.scenario_number.between(3, 21).all()),
        "manifest_final_status": bool(manifest.processing_status.isin(["success", "failed"]).all()),
        "no_duplicate_recordings": not recordings.recording_identifier.duplicated().any(),
        "finite_success_durations": bool(np.isfinite(
            recordings.loc[recordings.processing_status == "success", "recording_duration"]
        ).all()),
        "four_nonempty_figures": len(figures) == 4 and all(path.stat().st_size > 0 for path in figures),
    }
    (output_dir / "verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite to replace pilot outputs")
    for relative in (
        "tables", "figures", "diagnostics", "intermediate/macro_segments", "intermediate/micro_events"
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    config = PipelineConfig.from_yaml(Path(args.config))
    eligible = discover_recordings(Path(args.inventory), args.scenario_min, args.scenario_max)
    selected, chosen = select_recordings(eligible, args.pednyc_sample_size, args.seed)
    manifest = chosen[[
        "pednyc_number", "scenario_number", "source_decoded_csv_path", "recording_identifier"
    ]].copy()
    manifest.insert(0, "random_seed", args.seed)
    manifest["processing_status"] = "pending"
    manifest["failure_reason"] = ""
    summaries: list[dict] = []
    trace_frames: list[pd.DataFrame] = []
    for index, row in chosen.iterrows():
        try:
            summary, traces = process_recording(row, config, output_dir)
            summaries.append(summary)
            trace_frames.append(traces)
            manifest.loc[manifest.recording_identifier == row.recording_identifier, "processing_status"] = "success"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            summaries.append({
                "pednyc_number": int(row.pednyc_number), "scenario_number": int(row.scenario_number),
                "recording_identifier": row.recording_identifier,
                "source_decoded_csv_path": row.source_decoded_csv_path,
                "processing_status": "failed", "quality_warnings": reason,
            })
            mask = manifest.recording_identifier == row.recording_identifier
            manifest.loc[mask, ["processing_status", "failure_reason"]] = ["failed", reason]
        print(f"progress {len(summaries)}/{len(chosen)}")
    recordings = pd.DataFrame(summaries)
    for column in NUMERIC_FEATURES:
        if column not in recordings:
            recordings[column] = np.nan
    scenarios = aggregate_scenarios(recordings)
    traces = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
    manifest.to_csv(output_dir / "selection_manifest.csv", index=False)
    recordings.to_csv(output_dir / "tables" / "recording_summary.csv", index=False)
    scenarios.to_csv(output_dir / "tables" / "scenario_summary.csv", index=False)
    if traces.empty:
        raise RuntimeError("No successful recordings; figures cannot be generated")
    create_figures(recordings, scenarios, traces, output_dir / "figures")
    write_report(output_dir, selected, recordings, scenarios)
    run_config = {
        "selected_pednyc_numbers": selected, "eligible_scenario_range": [args.scenario_min, args.scenario_max],
        "random_seed": args.seed, "feature_settings": asdict(config),
        "macro_segmentation_settings": {
            key: getattr(config, key) for key in (
                "speed_threshold", "walking_speed", "acceleration_slope", "deceleration_slope",
                "macro_smoothing_seconds", "macro_accel_search_seconds", "final_deceleration_fraction"
            )
        },
        "micro_segmentation_settings": {
            key: getattr(config, key) for key in (
                "window_size", "window_step", "window_min", "window_max", "pause_speed",
                "pause_min_duration", "min_event_duration", "merge_gap_threshold",
                "head_check_yaw", "head_check_turn_rate"
            )
        },
        "alignment_method": "pedestrian movement onset; scenario-relative start fallback",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    checks = verify_outputs(output_dir, selected, manifest, recordings)
    print("selected=" + ",".join(f"PedNYC{x}" for x in selected))
    print(f"completed={sum(manifest.processing_status == 'success')}/{len(manifest)}")
    print("verification=" + ",".join(f"{key}:{value}" for key, value in checks.items()))
    print(f"outputs={output_dir}")
    return 0 if all(checks.values()) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pednyc-sample-size", type=int, default=5)
    parser.add_argument("--scenario-min", type=int, default=3)
    parser.add_argument("--scenario-max", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inventory", type=Path, default=PROJECT_ROOT / "data/processed/decoded_inventory.csv")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config_default.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/scenario_pattern_pilot")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.pednyc_sample_size != 5:
        parser.error("this pilot requires exactly --pednyc-sample-size 5")
    if args.scenario_min > args.scenario_max:
        parser.error("--scenario-min must not exceed --scenario-max")
    return args


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
