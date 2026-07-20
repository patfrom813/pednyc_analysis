"""Path-independent plots for pipeline outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_GANTT_GROUPS = {
    "Observable Kinematics": ["low_motion", "acceleration", "deceleration", "crossing_commitment"],
    "Inferred Latent Behaviors": ["head_check", "possible_hesitation", "possible_yielding"],
}

_GANTT_COLORS = {
    "low_motion": "#6B83AE",
    "acceleration": "#79A653",
    "deceleration": "#C45E59",
    "crossing_commitment": "#7EAAA5",
    "head_check": "#F0BE68",
    "possible_hesitation": "#DE9331",
    "possible_yielding": "#B97495",
}


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


def plot_behavior_gantt(windows: pd.DataFrame, macro_segments: pd.DataFrame, output_path: Path) -> Path:
    """Plot classified behavior windows in observable and inferred Gantt lanes."""
    plt = _pyplot()
    from matplotlib.patches import Patch

    labels = [label for group in _GANTT_GROUPS.values() for label in group]
    positions = {label: len(labels) - index - 1 for index, label in enumerate(labels)}
    fig, ax = plt.subplots(figsize=(16, 8))
    split = len(_GANTT_GROUPS["Inferred Latent Behaviors"])
    ax.axhspan(split - 0.5, len(labels) - 0.5, color="#F4F6F8", zorder=0)
    ax.axhspan(-0.5, split - 0.5, color="#F8F6F3", zorder=0)

    for label, start, end in _gantt_intervals(windows):
        if label not in positions:
            continue
        ax.broken_barh(
            [(start, max(end - start, 1e-6))],
            (positions[label] - 0.30, 0.60),
            facecolors=_GANTT_COLORS[label],
            edgecolors="white",
            linewidth=0.8,
            zorder=3,
        )

    if not macro_segments.empty:
        top = len(labels) - 0.15
        for index, row in macro_segments.reset_index(drop=True).iterrows():
            start = float(row["start_time"])
            end = float(row["end_time"])
            ax.axvline(start, color="#8C514B", linestyle="--", linewidth=1.0, alpha=0.8, zorder=2)
            short_phase = end - start < 1.2
            ax.text(
                (start + end) / 2.0,
                top - (0.18 if index % 2 else 0.0),
                f"M{index}",
                ha="center",
                va="top",
                rotation=90 if short_phase else 0,
                fontsize=8 if short_phase else 9,
                color="#6F3733",
            )
        ax.axvline(float(macro_segments.iloc[-1]["end_time"]), color="#8C514B", linestyle="--", linewidth=1.0, alpha=0.8, zorder=2)

    ax.axhline(split - 0.5, color="#555555", linewidth=1.1)
    ax.text(0.01, 0.93, "Observable Kinematics", transform=ax.transAxes, weight="bold", fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    ax.text(0.01, 0.34, "Inferred Latent Behaviors", transform=ax.transAxes, weight="bold", fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    ax.set_yticks([positions[label] for label in labels])
    ax.set_yticklabels([label.replace("_", " ") for label in labels])
    ax.set_ylim(-0.5, len(labels) + 0.15)
    ax.set_xlabel("Scenario elapsed time (seconds)")
    ax.set_ylabel("Behavior lane")
    ax.set_title("Observable Kinematics and Inferred Latent Behaviors")
    ax.grid(axis="x", alpha=0.2, zorder=1)
    ax.legend(
        handles=[Patch(facecolor=_GANTT_COLORS[label], label=label.replace("_", " ")) for label in labels],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=4,
        frameon=False,
    )
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


def _gantt_intervals(windows: pd.DataFrame) -> list[tuple[str, float, float]]:
    """Coalesce overlapping windows with the same predicted label."""
    required = {"window_start", "window_end", "predicted_label"}
    if windows.empty or not required.issubset(windows.columns):
        return []
    intervals: list[tuple[str, float, float]] = []
    for label, group in windows.groupby("predicted_label", sort=False):
        ordered = group.sort_values(["window_start", "window_end"])
        current_start: float | None = None
        current_end: float | None = None
        for _, row in ordered.iterrows():
            start, end = float(row["window_start"]), float(row["window_end"])
            if current_start is None:
                current_start, current_end = start, end
            elif start <= float(current_end):
                current_end = max(float(current_end), end)
            else:
                intervals.append((str(label), current_start, float(current_end)))
                current_start, current_end = start, end
        if current_start is not None:
            intervals.append((str(label), current_start, float(current_end)))
    return sorted(intervals, key=lambda item: (item[1], item[0]))


def _save(fig, output_path: Path, plt) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
