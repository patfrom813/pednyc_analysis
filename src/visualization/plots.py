"""Path-independent plots for pipeline outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_speed_profile(df: pd.DataFrame, output_path: Path, title: str | None = None) -> Path:
    """Plot pedestrian and vehicle speed over scenario time."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["time"], df["ped_speed"], label="Pedestrian speed", linewidth=1.8)
    if "veh_speed" in df:
        ax.plot(df["time"], df["veh_speed"], label="Vehicle speed", linewidth=1.2, alpha=0.8)
    ax.set(title=title or "Speed profile", xlabel="Time (s)", ylabel="Speed (m/s)")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save(fig, output_path, plt)


def plot_macro_segments(df: pd.DataFrame, segments: pd.DataFrame, output_path: Path) -> Path:
    """Plot pedestrian speed with macro phase spans."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"low_motion": "#bdbdbd", "acceleration": "#59a14f", "crossing": "#4e79a7", "deceleration": "#f28e2b", "post_motion": "#9c9c9c"}
    for _, row in segments.iterrows():
        ax.axvspan(row["start_time"], row["end_time"], color=colors.get(row["phase_label"], "#dddddd"), alpha=0.25, label=row["phase_label"])
    ax.plot(df["time"], df["ped_speed"], color="#222222", linewidth=1.6)
    ax.set(xlabel="Time (s)", ylabel="Pedestrian speed (m/s)", title="Macro segmentation")
    ax.grid(alpha=0.2)
    return _save(fig, output_path, plt)


def plot_micro_windows(df: pd.DataFrame, windows: pd.DataFrame, output_path: Path) -> Path:
    """Plot micro window boundaries over pedestrian speed."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["time"], df["ped_speed"], color="#222222", linewidth=1.5)
    for value in windows.get("window_start", pd.Series(dtype=float)).drop_duplicates():
        ax.axvline(value, color="#4e79a7", alpha=0.25, linewidth=0.8)
    ax.set(xlabel="Time (s)", ylabel="Pedestrian speed (m/s)", title="Micro windows")
    ax.grid(alpha=0.2)
    return _save(fig, output_path, plt)


def plot_detected_events(df: pd.DataFrame, events: pd.DataFrame, output_path: Path) -> Path:
    """Plot merged detected events over pedestrian speed."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["time"], df["ped_speed"], color="#222222", linewidth=1.5)
    for index, row in events.iterrows():
        ax.axvspan(row["start_time"], row["end_time"], alpha=0.25, color=f"C{index % 10}")
        ax.text(row["start_time"], ax.get_ylim()[1], row["label"], rotation=90, va="top", fontsize=8)
    ax.set(xlabel="Time (s)", ylabel="Pedestrian speed (m/s)", title="Detected events")
    ax.grid(alpha=0.2)
    return _save(fig, output_path, plt)


def plot_validation_comparison(events: pd.DataFrame, annotations: pd.DataFrame, output_path: Path) -> Path:
    """Plot predicted and human event intervals on separate lanes."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(12, 4))
    for _, row in events.iterrows():
        ax.barh(1, row["end_time"] - row["start_time"], left=row["start_time"], height=0.35, alpha=0.7)
    for _, row in annotations.iterrows():
        if pd.notna(row.get("human_label")):
            ax.barh(0, row["end_time"] - row["start_time"], left=row["start_time"], height=0.35, alpha=0.7)
    ax.set(yticks=[0, 1], yticklabels=["Human", "Predicted"], xlabel="Time (s)", title="Validation comparison")
    return _save(fig, output_path, plt)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, output_path: Path, plt) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
