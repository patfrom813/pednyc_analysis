from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGE_DIR = ROOT / ".python_packages"
if LOCAL_PACKAGE_DIR.exists():
    sys.path.insert(0, str(LOCAL_PACKAGE_DIR))

DEFAULT_FEATURE_CSV = ROOT / "data" / "processed" / "features_PedNYC1_scenario3_metrics_v1.csv"
DEFAULT_MACRO_CSV = (
    ROOT
    / "outputs"
    / "graphs"
    / "macro_segmentation"
    / "macro_segments_PedNYC1_scenario3_v2.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "graphs" / "micro_segmentation"

TIME_COL = "ScenarioTime"
DT_COL = "dt"
GAME_TIME_COL = "GameTime"
FRAME_COL = "Frame Number"

PED_SPEED_COLS = ["ped_speed_xz_smooth", "ped_speed_xz"]
PED_ACCEL_COLS = ["ped_accel_xz_smooth", "ped_accel_xz"]
HEAD_TURN_COLS = ["head_turn_rate"]
HEAD_BODY_DIFF_COLS = ["head_body_yaw_diff"]
PED_X_COLS = ["ped_x"]
PED_Z_COLS = ["ped_z"]
DISTANCE_COLS = ["car_ped_distance_xz_smooth", "car_ped_distance_xz"]
CLOSING_COLS = ["distance_closing_smooth", "distance_closing"]
DISTANCE_CHANGE_COLS = ["distance_change"]
CAR_SPEED_COLS = ["car_speed_xz_smooth", "car_speed_xz"]
CAR_ACCEL_COLS = ["car_accel_xz_smooth", "car_accel_xz"]
AVATAR_GAP_COLS = ["avatar_vr_gap_xz"]

# Change-point detection settings. The percentile is computed inside each
# macro segment, so short/slow and long/active macro intervals get their own
# local sensitivity. These are descriptive segmentation settings, not behavior
# classification thresholds.
CHANGE_WEIGHTS = {
    "speed_delta": 1.0,
    "accel_delta": 1.5,
    "head_turn_delta": 1.0,
}
CHANGE_SCORE_PERCENTILE = 75
CHANGE_SCORE_SMOOTH_SECONDS = 0.20
MIN_BOUNDARY_GAP_SECONDS = 0.30
MIN_MICRO_SEGMENT_SECONDS = 0.40

# Descriptive event groups. These are not final behavior labels; they are
# shorthand for measured properties so manual review can interpret the segment.
GROUP_DEFINITIONS = {
    "near_stationary": "Mean VR-derived pedestrian speed is very low.",
    "speed_increasing": "Pedestrian speed rises across the micro-segment.",
    "speed_decreasing": "Pedestrian speed falls across the micro-segment.",
    "head_active": "Head turn rate or head/body yaw difference is elevated.",
    "speed_steady": "Pedestrian speed is relatively stable.",
    "mixed_motion": "No single measured pattern dominates.",
}
GROUP_COLORS = {
    "near_stationary": "#4E79A7",
    "speed_increasing": "#59A14F",
    "speed_decreasing": "#E15759",
    "head_active": "#EDC948",
    "speed_steady": "#76B7B2",
    "mixed_motion": "#9C755F",
}
NEAR_STATIONARY_SPEED = 0.15
SPEED_CHANGE_MIN = 0.20
SPEED_SLOPE_MIN = 0.08
STEADY_SPEED_STD_MAX = 0.12
HEAD_ACTIVE_TURN_RATE = 60.0
HEAD_ACTIVE_YAW_DIFF = 20.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create descriptive micro-segments inside existing macro segments. "
            "Segments are found by change points, then described with measured features."
        )
    )
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--macro-csv", type=Path, default=DEFAULT_MACRO_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


def clean_columns(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def pick_col(df, candidates, required=False, purpose="column"):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Missing required {purpose}. Tried: {candidates}")
    return None


def numeric_series(df, col, default=np.nan):
    if col is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def is_seconds_like(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return False
    duration = float(np.nanmax(values) - np.nanmin(values))
    return 0 < duration <= 60 and abs(float(np.nanmin(values))) <= 5


def reconstruct_elapsed_time(df):
    if TIME_COL in df.columns:
        scenario_time = numeric_series(df, TIME_COL).to_numpy(dtype=float)
        if is_seconds_like(scenario_time):
            return scenario_time - scenario_time[0], "ScenarioTime_seconds"

    if DT_COL in df.columns:
        dt = numeric_series(df, DT_COL).to_numpy(dtype=float)
        good_dt = dt[np.isfinite(dt) & (dt > 0) & (dt < 1.0)]
        if len(good_dt):
            median_dt = float(np.nanmedian(good_dt))
            dt_clean = dt.copy()
            bad = ~np.isfinite(dt_clean) | (dt_clean <= 0) | (dt_clean > 1.0)
            dt_clean[bad] = median_dt
            elapsed = np.zeros(len(df), dtype=float)
            elapsed[1:] = np.cumsum(dt_clean[1:])
            return elapsed, "cumulative_dt_seconds"

    if GAME_TIME_COL in df.columns:
        game_time = numeric_series(df, GAME_TIME_COL).to_numpy(dtype=float)
        if np.any(np.isfinite(game_time)):
            return game_time - game_time[np.isfinite(game_time)][0], "GameTime_minus_start"

    if FRAME_COL in df.columns and DT_COL in df.columns:
        frames = numeric_series(df, FRAME_COL).to_numpy(dtype=float)
        dt = numeric_series(df, DT_COL).to_numpy(dtype=float)
        median_dt = float(np.nanmedian(dt[np.isfinite(dt) & (dt > 0)]))
        return (frames - frames[np.isfinite(frames)][0]) * median_dt, "Frame_Number_times_dt"

    raise ValueError("Could not reconstruct elapsed seconds from ScenarioTime, dt, GameTime, or frame number.")


def seconds_to_frames(time_sec, seconds):
    if len(time_sec) < 2:
        return 1
    dt = np.nanmedian(np.diff(time_sec))
    if not np.isfinite(dt) or dt <= 0:
        return 1
    return max(1, int(round(seconds / dt)))


def rolling_mean(values, window):
    window = max(1, int(window))
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def safe_stat(values, func):
    arr = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(arr)):
        return np.nan
    return float(func(arr))


def linear_slope(time_sec, values):
    t = np.asarray(time_sec, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = t[mask] - t[mask][0]
    if np.nanmax(x) <= np.nanmin(x):
        return np.nan
    return float(np.polyfit(x, y[mask], 1)[0])


def normalized_delta(values):
    arr = np.asarray(values, dtype=float)
    filled = pd.Series(arr).interpolate(limit_direction="both").to_numpy(dtype=float)
    delta = np.abs(np.diff(filled, prepend=filled[0]))
    scale = float(np.nanstd(filled))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return delta / scale


def local_peak_indices(values, threshold):
    values = np.asarray(values, dtype=float)
    peaks = []
    if len(values) < 3:
        return peaks
    for i in range(1, len(values) - 1):
        if values[i] >= threshold and values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            peaks.append(i)
    return peaks


def merge_close_boundaries(boundaries, time_sec, scores):
    if not boundaries:
        return []
    boundaries = sorted(set(int(b) for b in boundaries))
    merged = [boundaries[0]]
    for boundary in boundaries[1:]:
        prev = merged[-1]
        if time_sec[boundary] - time_sec[prev] < MIN_BOUNDARY_GAP_SECONDS:
            if scores[boundary] > scores[prev]:
                merged[-1] = boundary
        else:
            merged.append(boundary)
    return merged


def macro_row_to_indices(macro_row, df):
    if {"start_idx", "end_idx"}.issubset(macro_row.index):
        return max(0, int(macro_row["start_idx"])), min(len(df) - 1, int(macro_row["end_idx"]))

    start_col = "time_start_sec" if "time_start_sec" in macro_row.index else "ScenarioTime_start"
    end_col = "time_end_sec" if "time_end_sec" in macro_row.index else "ScenarioTime_end"
    start_t = float(macro_row[start_col])
    end_t = float(macro_row[end_col])
    elapsed = df["elapsed_time_sec"].to_numpy(dtype=float)
    idx = np.where((elapsed >= start_t) & (elapsed <= end_t))[0]
    if len(idx) == 0:
        raise ValueError(f"No feature rows found for macro segment {macro_row.get('segment_id', 'unknown')}")
    return int(idx[0]), int(idx[-1])


def compute_change_score(seg):
    time_sec = seg["elapsed_time_sec"].to_numpy(dtype=float)
    speed_col = pick_col(seg, PED_SPEED_COLS, required=True, purpose="pedestrian speed")
    accel_col = pick_col(seg, PED_ACCEL_COLS)
    head_col = pick_col(seg, HEAD_TURN_COLS)

    speed_score = normalized_delta(numeric_series(seg, speed_col))
    accel_score = normalized_delta(numeric_series(seg, accel_col, default=0.0))
    head_score = normalized_delta(numeric_series(seg, head_col, default=0.0))

    score = (
        CHANGE_WEIGHTS["speed_delta"] * speed_score
        + CHANGE_WEIGHTS["accel_delta"] * accel_score
        + CHANGE_WEIGHTS["head_turn_delta"] * head_score
    )
    window = seconds_to_frames(time_sec, CHANGE_SCORE_SMOOTH_SECONDS)
    return rolling_mean(score, window)


def find_boundaries(seg):
    time_sec = seg["elapsed_time_sec"].to_numpy(dtype=float)
    if len(seg) < 3:
        return [0, len(seg) - 1], np.zeros(len(seg), dtype=float)

    score = compute_change_score(seg)
    threshold = float(np.nanpercentile(score, CHANGE_SCORE_PERCENTILE))
    candidates = local_peak_indices(score, threshold)
    candidates = merge_close_boundaries(candidates, time_sec, score)
    boundaries = [0] + [c for c in candidates if 0 < c < len(seg) - 1] + [len(seg) - 1]
    return sorted(set(boundaries)), score


def displacement_stats(seg):
    x_col = pick_col(seg, PED_X_COLS)
    z_col = pick_col(seg, PED_Z_COLS)
    if x_col is None or z_col is None:
        return np.nan, np.nan, np.nan
    x = numeric_series(seg, x_col).to_numpy(dtype=float)
    z = numeric_series(seg, z_col).to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(z)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan
    x = x[mask]
    z = z[mask]
    displacement = float(np.sqrt((x[-1] - x[0]) ** 2 + (z[-1] - z[0]) ** 2))
    path_length = float(np.nansum(np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2)))
    progress_ratio = displacement / path_length if path_length > 1e-9 else np.nan
    return displacement, path_length, progress_ratio


def classify_event_group(features):
    if features["speed_mean"] < NEAR_STATIONARY_SPEED:
        return "near_stationary"
    if (
        features["head_turn_rate_max_abs"] >= HEAD_ACTIVE_TURN_RATE
        or features["head_body_yaw_diff_max_abs"] >= HEAD_ACTIVE_YAW_DIFF
    ):
        return "head_active"
    if features["speed_delta"] >= SPEED_CHANGE_MIN or features["speed_slope"] >= SPEED_SLOPE_MIN:
        return "speed_increasing"
    if features["speed_delta"] <= -SPEED_CHANGE_MIN or features["speed_slope"] <= -SPEED_SLOPE_MIN:
        return "speed_decreasing"
    if features["speed_std"] <= STEADY_SPEED_STD_MAX:
        return "speed_steady"
    return "mixed_motion"


def describe_segment(features):
    parts = [
        f"speed {features['speed_start']:.2f}->{features['speed_end']:.2f} m/s",
        f"mean {features['speed_mean']:.2f} m/s",
    ]
    if np.isfinite(features["distance_start"]) and np.isfinite(features["distance_end"]):
        parts.append(f"distance {features['distance_start']:.2f}->{features['distance_end']:.2f} m")
    if np.isfinite(features["head_turn_rate_max_abs"]):
        parts.append(f"head max {features['head_turn_rate_max_abs']:.1f} deg/s")
    if np.isfinite(features["progress_ratio"]):
        parts.append(f"progress ratio {features['progress_ratio']:.2f}")
    return "; ".join(parts)


def extract_features(global_df, macro_row, micro_id, local_start, local_end, local_seg, change_score):
    seg = local_seg.iloc[local_start:local_end + 1].copy()
    time_sec = seg["elapsed_time_sec"].to_numpy(dtype=float)
    if len(seg) < 2:
        return None
    duration = float(time_sec[-1] - time_sec[0])
    if duration < MIN_MICRO_SEGMENT_SECONDS:
        return None

    speed = numeric_series(seg, pick_col(seg, PED_SPEED_COLS, required=True)).to_numpy(dtype=float)
    accel = numeric_series(seg, pick_col(seg, PED_ACCEL_COLS)).to_numpy(dtype=float)
    head_turn = numeric_series(seg, pick_col(seg, HEAD_TURN_COLS), default=0.0).to_numpy(dtype=float)
    head_body = numeric_series(seg, pick_col(seg, HEAD_BODY_DIFF_COLS), default=0.0).to_numpy(dtype=float)
    distance = numeric_series(seg, pick_col(seg, DISTANCE_COLS)).to_numpy(dtype=float)
    closing = numeric_series(seg, pick_col(seg, CLOSING_COLS), default=np.nan).to_numpy(dtype=float)
    distance_change = numeric_series(seg, pick_col(seg, DISTANCE_CHANGE_COLS), default=np.nan).to_numpy(dtype=float)
    car_speed = numeric_series(seg, pick_col(seg, CAR_SPEED_COLS), default=np.nan).to_numpy(dtype=float)
    car_accel = numeric_series(seg, pick_col(seg, CAR_ACCEL_COLS), default=np.nan).to_numpy(dtype=float)
    avatar_gap = numeric_series(seg, pick_col(seg, AVATAR_GAP_COLS), default=np.nan).to_numpy(dtype=float)
    displacement, path_length, progress_ratio = displacement_stats(seg)

    features = {
        "micro_segment_id": micro_id,
        "macro_segment_id": int(macro_row["segment_id"]),
        "macro_label": str(macro_row.get("macro_label", "")),
        "start_idx": int(seg.index[0]),
        "end_idx": int(seg.index[-1]),
        "start_time_sec": float(time_sec[0]),
        "end_time_sec": float(time_sec[-1]),
        "duration_sec": duration,
        "start_frame": seg[FRAME_COL].iloc[0] if FRAME_COL in seg.columns else np.nan,
        "end_frame": seg[FRAME_COL].iloc[-1] if FRAME_COL in seg.columns else np.nan,
        "ScenarioTime_start_raw": seg[TIME_COL].iloc[0] if TIME_COL in seg.columns else np.nan,
        "ScenarioTime_end_raw": seg[TIME_COL].iloc[-1] if TIME_COL in seg.columns else np.nan,
        "change_score_mean": safe_stat(change_score[local_start:local_end + 1], np.nanmean),
        "change_score_max": safe_stat(change_score[local_start:local_end + 1], np.nanmax),
        "speed_start": safe_stat(speed[:1], np.nanmean),
        "speed_end": safe_stat(speed[-1:], np.nanmean),
        "speed_mean": safe_stat(speed, np.nanmean),
        "speed_std": safe_stat(speed, np.nanstd),
        "speed_min": safe_stat(speed, np.nanmin),
        "speed_max": safe_stat(speed, np.nanmax),
        "speed_slope": linear_slope(time_sec, speed),
        "accel_start": safe_stat(accel[:1], np.nanmean),
        "accel_end": safe_stat(accel[-1:], np.nanmean),
        "accel_mean": safe_stat(accel, np.nanmean),
        "accel_std": safe_stat(accel, np.nanstd),
        "accel_min": safe_stat(accel, np.nanmin),
        "accel_max": safe_stat(accel, np.nanmax),
        "head_turn_rate_mean": safe_stat(np.abs(head_turn), np.nanmean),
        "head_turn_rate_max_abs": safe_stat(np.abs(head_turn), np.nanmax),
        "head_body_yaw_diff_mean_abs": safe_stat(np.abs(head_body), np.nanmean),
        "head_body_yaw_diff_max_abs": safe_stat(np.abs(head_body), np.nanmax),
        "distance_start": safe_stat(distance[:1], np.nanmean),
        "distance_end": safe_stat(distance[-1:], np.nanmean),
        "distance_mean": safe_stat(distance, np.nanmean),
        "distance_min": safe_stat(distance, np.nanmin),
        "distance_slope": linear_slope(time_sec, distance),
        "distance_closing_mean": safe_stat(closing, np.nanmean),
        "distance_change_mean": safe_stat(distance_change, np.nanmean),
        "car_speed_mean": safe_stat(car_speed, np.nanmean),
        "car_accel_mean": safe_stat(car_accel, np.nanmean),
        "avatar_vr_gap_max": safe_stat(avatar_gap, np.nanmax),
        "displacement_xz": displacement,
        "path_length_xz": path_length,
        "progress_ratio": progress_ratio,
    }
    features["speed_delta"] = features["speed_end"] - features["speed_start"]
    features["distance_delta"] = features["distance_end"] - features["distance_start"]
    features["event_group"] = classify_event_group(features)
    features["event_group_definition"] = GROUP_DEFINITIONS[features["event_group"]]
    features["plain_description"] = describe_segment(features)
    features["signals_used_for_boundaries"] = "ped_speed_xz_smooth|ped_accel_xz_smooth|head_turn_rate"
    return features


def build_micro_segments(features_df, macro_df):
    rows = []
    frame_df = features_df.copy()
    frame_df["micro_segment_id"] = -1
    frame_df["micro_event_group"] = "unassigned"
    frame_df["macro_segment_id"] = -1
    frame_df["change_score"] = np.nan

    next_micro_id = 0
    boundary_records = []

    for _, macro_row in macro_df.iterrows():
        start_idx, end_idx = macro_row_to_indices(macro_row, frame_df)
        macro_seg = frame_df.iloc[start_idx:end_idx + 1].copy()
        boundaries, score = find_boundaries(macro_seg)
        frame_df.loc[macro_seg.index, "change_score"] = score
        frame_df.loc[start_idx:end_idx, "macro_segment_id"] = int(macro_row["segment_id"])

        for boundary in boundaries:
            boundary_records.append({
                "macro_segment_id": int(macro_row["segment_id"]),
                "boundary_idx": int(macro_seg.index[boundary]),
                "boundary_time_sec": float(macro_seg["elapsed_time_sec"].iloc[boundary]),
                "change_score": float(score[boundary]) if len(score) else np.nan,
            })

        for left, right in zip(boundaries[:-1], boundaries[1:]):
            features = extract_features(frame_df, macro_row, next_micro_id, left, right, macro_seg, score)
            if features is None:
                continue
            rows.append(features)
            frame_df.loc[features["start_idx"]:features["end_idx"], "micro_segment_id"] = next_micro_id
            frame_df.loc[features["start_idx"]:features["end_idx"], "micro_event_group"] = features["event_group"]
            next_micro_id += 1

    return pd.DataFrame(rows), frame_df, pd.DataFrame(boundary_records)


def summarize_segments(segments_df, macro_df):
    if segments_df.empty:
        return pd.DataFrame()
    summary = (
        segments_df
        .groupby(["macro_segment_id", "macro_label", "event_group"], dropna=False)
        .agg(
            count=("micro_segment_id", "count"),
            total_duration_sec=("duration_sec", "sum"),
            mean_duration_sec=("duration_sec", "mean"),
            first_start_time_sec=("start_time_sec", "min"),
            last_end_time_sec=("end_time_sec", "max"),
            speed_mean=("speed_mean", "mean"),
            distance_min=("distance_min", "min"),
        )
        .reset_index()
    )
    duration_map = macro_df.set_index("segment_id")["duration_sec"].to_dict()
    summary["macro_duration_sec"] = summary["macro_segment_id"].map(duration_map)
    summary["percent_of_macro_segment"] = np.where(
        summary["macro_duration_sec"] > 0,
        100.0 * summary["total_duration_sec"] / summary["macro_duration_sec"],
        np.nan,
    )
    return summary.sort_values(["macro_segment_id", "first_start_time_sec", "event_group"])


def load_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        return plt, Patch
    except ImportError:
        print("Skipping plots: matplotlib is not installed.")
        return None, None


def shade_macro_segments(ax, macro_df, label_top=False):
    for i, row in macro_df.iterrows():
        color = "#F7F7F7" if i % 2 == 0 else "#ECECEC"
        ax.axvspan(row["time_start_sec"], row["time_end_sec"], color=color, alpha=0.45, zorder=0)
        ax.axvline(row["time_start_sec"], color="#777777", linewidth=0.8, alpha=0.8, zorder=1)
        if label_top:
            ax.text(
                (row["time_start_sec"] + row["time_end_sec"]) / 2,
                0.98,
                f"macro {int(row['segment_id'])}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#555555",
            )
    if len(macro_df):
        ax.axvline(macro_df["time_end_sec"].iloc[-1], color="#777777", linewidth=0.8, alpha=0.8, zorder=1)


def draw_boundaries(ax, boundary_df):
    for t in boundary_df["boundary_time_sec"].dropna().unique():
        ax.axvline(float(t), color="#AA3377", linewidth=0.7, alpha=0.45, zorder=2)


def plot_standard_three_panel(frame_df, macro_df, boundary_df, output_path):
    plt, _ = load_matplotlib()
    if plt is None:
        return False
    time = frame_df["elapsed_time_sec"].to_numpy(dtype=float)
    panels = [
        ("Pedestrian speed", pick_col(frame_df, PED_SPEED_COLS), "m/s"),
        ("Pedestrian acceleration", pick_col(frame_df, PED_ACCEL_COLS), "m/s^2"),
        ("Head turn rate", pick_col(frame_df, HEAD_TURN_COLS), "deg/s"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    for ax, (title, col, ylabel) in zip(axes, panels):
        shade_macro_segments(ax, macro_df)
        draw_boundaries(ax, boundary_df)
        if col is not None:
            ax.plot(time, numeric_series(frame_df, col), color="#222222", linewidth=1.3, label=col)
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Scenario elapsed time, seconds")
    fig.suptitle("Micro-Segmentation Change Boundaries: Core Signals", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_stacked_comparison(frame_df, macro_df, segments_df, boundary_df, output_path):
    plt, Patch = load_matplotlib()
    if plt is None:
        return False
    time = frame_df["elapsed_time_sec"].to_numpy(dtype=float)
    panels = [
        ("Speed", pick_col(frame_df, PED_SPEED_COLS), "m/s"),
        ("Acceleration", pick_col(frame_df, PED_ACCEL_COLS), "m/s^2"),
        ("Head turn", pick_col(frame_df, HEAD_TURN_COLS), "deg/s"),
        ("Distance to car", pick_col(frame_df, DISTANCE_COLS), "m"),
        ("Distance closing/change", pick_col(frame_df, CLOSING_COLS) or pick_col(frame_df, DISTANCE_CHANGE_COLS), "value"),
        ("Car speed/accel", pick_col(frame_df, CAR_SPEED_COLS) or pick_col(frame_df, CAR_ACCEL_COLS), "value"),
    ]
    fig, axes = plt.subplots(7, 1, figsize=(18, 14), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1, 1, 1, 1.4]})
    for ax, (title, col, ylabel) in zip(axes[:6], panels):
        shade_macro_segments(ax, macro_df)
        draw_boundaries(ax, boundary_df)
        if col is not None:
            ax.plot(time, numeric_series(frame_df, col), color="#222222", linewidth=1.1, label=col)
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "signal unavailable", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    ax = axes[-1]
    shade_macro_segments(ax, macro_df, label_top=True)
    group_order = [group for group in GROUP_DEFINITIONS if group in set(segments_df["event_group"])]
    y_lookup = {group: i for i, group in enumerate(group_order)}
    for _, row in segments_df.iterrows():
        ax.barh(
            y_lookup[row["event_group"]],
            row["duration_sec"],
            left=row["start_time_sec"],
            height=0.6,
            color=GROUP_COLORS[row["event_group"]],
            edgecolor="white",
            linewidth=0.8,
        )
    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(group_order, fontsize=8)
    ax.set_title("Descriptive micro-segment groups", loc="left", fontsize=10)
    ax.set_xlabel("Scenario elapsed time, seconds")
    ax.grid(True, axis="x", alpha=0.25)
    handles = [Patch(facecolor=GROUP_COLORS[group], label=f"{group}: {GROUP_DEFINITIONS[group]}") for group in group_order]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=8, frameon=False)
    fig.suptitle("Stacked Signal Comparison With Macro and Micro Segments", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.78, 0.97])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_gantt(segments_df, macro_df, output_path):
    plt, Patch = load_matplotlib()
    if plt is None:
        return False
    group_order = [group for group in GROUP_DEFINITIONS if group in set(segments_df["event_group"])]
    fig, ax = plt.subplots(figsize=(18, max(6, 0.6 * len(group_order) + 3)))
    shade_macro_segments(ax, macro_df, label_top=True)
    y_lookup = {group: i for i, group in enumerate(group_order)}
    for _, row in segments_df.iterrows():
        ax.barh(
            y_lookup[row["event_group"]],
            row["duration_sec"],
            left=row["start_time_sec"],
            height=0.62,
            color=GROUP_COLORS[row["event_group"]],
            edgecolor="white",
            linewidth=0.8,
        )
    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(group_order)
    ax.set_xlabel("Scenario elapsed time, seconds")
    ax.set_ylabel("Descriptive event group")
    ax.set_title("Descriptive Micro-Segments by Event Group")
    ax.grid(True, axis="x", alpha=0.25)
    handles = [Patch(facecolor=GROUP_COLORS[group], label=f"{group}: {GROUP_DEFINITIONS[group]}") for group in group_order]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    args = parse_args()
    if not args.feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {args.feature_csv}")
    if not args.macro_csv.exists():
        raise FileNotFoundError(f"Macro segment CSV not found: {args.macro_csv}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    features = clean_columns(pd.read_csv(args.feature_csv))
    macro_df = clean_columns(pd.read_csv(args.macro_csv))
    pick_col(features, PED_SPEED_COLS, required=True, purpose="pedestrian speed")
    features = features.reset_index(drop=True)
    features["elapsed_time_sec"], time_source = reconstruct_elapsed_time(features)

    if not is_seconds_like(features["elapsed_time_sec"].to_numpy(dtype=float)):
        duration = features["elapsed_time_sec"].iloc[-1] - features["elapsed_time_sec"].iloc[0]
        raise ValueError(f"Elapsed time is not in expected seconds range after reconstruction: {duration}")

    segments_df, frame_df, boundary_df = build_micro_segments(features, macro_df)
    summary_df = summarize_segments(segments_df, macro_df)
    definitions_df = pd.DataFrame(
        [{"event_group": key, "definition": value, "color": GROUP_COLORS[key]} for key, value in GROUP_DEFINITIONS.items()]
    )

    segments_csv = output_dir / "micro_segments_descriptive_PedNYC1_scenario3_v3.csv"
    frame_csv = output_dir / "features_with_descriptive_micro_segments_PedNYC1_scenario3_v3.csv"
    summary_csv = output_dir / "micro_segment_summary_by_macro_PedNYC1_scenario3_v3.csv"
    boundary_csv = output_dir / "micro_change_boundaries_PedNYC1_scenario3_v3.csv"
    definitions_csv = output_dir / "micro_event_group_definitions_v3.csv"
    standard_png = output_dir / "micro_standard_3panel_PedNYC1_scenario3_v3.png"
    stacked_png = output_dir / "micro_stacked_feature_comparison_PedNYC1_scenario3_v3.png"
    gantt_png = output_dir / "micro_gantt_descriptive_groups_PedNYC1_scenario3_v3.png"

    segments_df.to_csv(segments_csv, index=False)
    frame_df.to_csv(frame_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    boundary_df.to_csv(boundary_csv, index=False)
    definitions_df.to_csv(definitions_csv, index=False)

    wrote_standard = wrote_stacked = wrote_gantt = False
    if not args.no_plot:
        wrote_standard = plot_standard_three_panel(frame_df, macro_df, boundary_df, standard_png)
        wrote_stacked = plot_stacked_comparison(frame_df, macro_df, segments_df, boundary_df, stacked_png)
        wrote_gantt = plot_gantt(segments_df, macro_df, gantt_png)

    print("\nDone.")
    print(f"Time source used: {time_source}")
    print(f"Scenario elapsed duration: {frame_df['elapsed_time_sec'].iloc[-1]:.3f} seconds")
    print(f"Saved descriptive micro-segments CSV: {segments_csv}")
    print(f"Saved frame-level CSV: {frame_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved boundary CSV: {boundary_csv}")
    print(f"Saved event group definitions CSV: {definitions_csv}")
    if wrote_standard:
        print(f"Saved standard 3-panel plot: {standard_png}")
    if wrote_stacked:
        print(f"Saved stacked comparison plot: {stacked_png}")
    if wrote_gantt:
        print(f"Saved Gantt plot: {gantt_png}")
    print("\nMicro-segments by descriptive group:")
    print(segments_df["event_group"].value_counts().to_string() if not segments_df.empty else "none")


if __name__ == "__main__":
    main()
