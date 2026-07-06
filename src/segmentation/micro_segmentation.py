from pathlib import Path
import argparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

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
FRAME_COL = "Frame Number"

PED_SPEED_COLS = ["ped_speed_xz_smooth", "ped_speed_xz"]
PED_ACCEL_COLS = ["ped_accel_xz_smooth", "ped_accel_xz"]
DISTANCE_COLS = ["car_ped_distance_xz_smooth", "car_ped_distance_xz"]
CLOSING_COLS = ["distance_closing_smooth", "distance_closing"]
CAR_ACCEL_COLS = ["car_accel_xz_smooth", "car_accel_xz"]
HEAD_TURN_COLS = ["head_turn_rate"]
HEAD_BODY_DIFF_COLS = ["head_body_yaw_diff"]
AVATAR_GAP_COLS = ["avatar_vr_gap_xz"]

STOP_SPEED = 0.15
WALK_SPEED = 0.30
CROSSING_SPEED = 0.60
ACCEL_THRESHOLD = 0.25
STRONG_ACCEL_THRESHOLD = 0.35
DECEL_THRESHOLD = -0.25
SLOPE_DEADBAND = 0.08
CAR_CLOSE_DISTANCE = 10.0
CAR_VERY_CLOSE_DISTANCE = 7.5
CLOSING_THRESHOLD = 0.10
CAR_BRAKE_ACCEL = -0.25
HEAD_TURN_RATE_THRESHOLD = 60.0
HEAD_BODY_DIFF_THRESHOLD = 20.0
NOISE_GAP_MIN_ABSOLUTE = 1.0
NOISE_GAP_QUANTILE = 0.95

MIN_EVENT_SECONDS = 0.35
MIN_LOW_MOTION_SECONDS = 0.50
MIN_CROSSING_SECONDS = 0.75
MIN_HEAD_CHECK_SECONDS = 0.25
MERGE_GAP_SECONDS = 0.25
TREND_WINDOW_SECONDS = 0.35
HESITATION_REBOUND_SECONDS = 1.50
HESITATION_REBOUND_SPEED = 0.15
HESITATION_DROP_SPEED = 0.20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect repeated micro-behavior events inside existing macro segments."
    )
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--macro-csv", type=Path, default=DEFAULT_MACRO_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the micro-event timeline plot.",
    )
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


def build_elapsed_time_seconds(df):
    if DT_COL in df.columns:
        dt = pd.to_numeric(df[DT_COL], errors="coerce").to_numpy(dtype=float)
        good_dt = dt[np.isfinite(dt) & (dt > 0) & (dt < 1.0)]
        median_dt = float(np.nanmedian(good_dt)) if len(good_dt) else 1 / 30

        dt_clean = dt.copy()
        bad = ~np.isfinite(dt_clean) | (dt_clean <= 0) | (dt_clean > 1.0)
        dt_clean[bad] = median_dt

        elapsed = np.zeros(len(df), dtype=float)
        elapsed[1:] = np.cumsum(dt_clean[1:])
        return elapsed

    if TIME_COL not in df.columns:
        raise ValueError(f"Missing required time column: {TIME_COL}")

    raw_time = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)
    return raw_time - raw_time[0]


def seconds_to_frames(time_sec, seconds):
    if len(time_sec) < 2:
        return 1
    dt = np.nanmedian(np.diff(time_sec))
    if not np.isfinite(dt) or dt <= 0:
        return 1
    return max(1, int(round(seconds / dt)))


def odd_window_from_seconds(time_sec, seconds):
    window = max(3, seconds_to_frames(time_sec, seconds))
    return window + 1 if window % 2 == 0 else window


def smooth_signal(values, time_sec, seconds=TREND_WINDOW_SECONDS):
    window = odd_window_from_seconds(time_sec, seconds)
    return (
        pd.Series(values)
        .interpolate(limit_direction="both")
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def gradient(values, time_sec):
    values = np.asarray(values, dtype=float)
    time_sec = np.asarray(time_sec, dtype=float)
    if len(values) < 3 or np.nanmax(time_sec) <= np.nanmin(time_sec):
        return np.full(len(values), np.nan)
    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=float)
    return np.gradient(filled, time_sec)


def true_runs(mask):
    runs = []
    start = None
    for i, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def run_duration(time_sec, start, end):
    if end <= start:
        return 0.0
    return float(time_sec[end] - time_sec[start])


def duration_mask_runs(mask, time_sec, min_seconds):
    return [
        (start, end)
        for start, end in true_runs(mask)
        if run_duration(time_sec, start, end) >= min_seconds
    ]


def safe_stat(values, func):
    arr = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(arr)):
        return np.nan
    return float(func(arr))


def fraction(values, mask_func):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return float(np.mean(mask_func(arr)))


def displacement_stats(seg):
    if "ped_x" not in seg.columns or "ped_z" not in seg.columns:
        return np.nan, np.nan, np.nan

    x = pd.to_numeric(seg["ped_x"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(seg["ped_z"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(z)
    if valid.sum() < 2:
        return np.nan, np.nan, np.nan

    x = x[valid]
    z = z[valid]
    displacement = float(np.sqrt((x[-1] - x[0]) ** 2 + (z[-1] - z[0]) ** 2))
    path = float(np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(z) ** 2)))
    progress_ratio = displacement / path if path > 1e-9 else np.nan
    return displacement, path, progress_ratio


def evidence_for_window(signals, local_start, local_end):
    sl = slice(local_start, local_end + 1)
    flags = []

    min_distance = safe_stat(signals["distance"][sl], np.nanmin)
    max_closing = safe_stat(signals["closing"][sl], np.nanmax)
    min_car_accel = safe_stat(signals["car_accel"][sl], np.nanmin)
    max_head_turn = safe_stat(np.abs(signals["head_turn"][sl]), np.nanmax)
    max_head_body = safe_stat(np.abs(signals["head_body"][sl]), np.nanmax)
    max_avatar_gap = safe_stat(signals["avatar_gap"][sl], np.nanmax)

    if np.isfinite(min_distance) and min_distance <= CAR_CLOSE_DISTANCE:
        flags.append("car_close")
    if np.isfinite(min_distance) and min_distance <= CAR_VERY_CLOSE_DISTANCE:
        flags.append("car_very_close")
    if np.isfinite(max_closing) and max_closing >= CLOSING_THRESHOLD:
        flags.append("car_ped_closing")
    if np.isfinite(min_car_accel) and min_car_accel <= CAR_BRAKE_ACCEL:
        flags.append("car_braking")
    if np.isfinite(max_head_turn) and max_head_turn >= HEAD_TURN_RATE_THRESHOLD:
        flags.append("head_turning")
    if np.isfinite(max_head_body) and max_head_body >= HEAD_BODY_DIFF_THRESHOLD:
        flags.append("head_body_offset")
    avatar_gap_threshold = signals.get("avatar_gap_threshold", np.inf)
    if np.isfinite(max_avatar_gap) and max_avatar_gap >= avatar_gap_threshold:
        flags.append("avatar_vr_gap_high")

    return flags


def confidence_from_flags(base, flags):
    score = base + 0.07 * min(4, len(flags))
    if "avatar_vr_gap_high" in flags:
        score -= 0.10
    return float(np.clip(score, 0.35, 0.95))


def add_event(events, macro_row, seg, signals, local_start, local_end, label, base_confidence, reason):
    local_start = int(max(0, local_start))
    local_end = int(min(len(seg) - 1, local_end))
    if local_end <= local_start:
        return

    time_sec = signals["time_sec"]
    duration = run_duration(time_sec, local_start, local_end)
    if duration <= 0:
        return

    flags = evidence_for_window(signals, local_start, local_end)
    confidence = confidence_from_flags(base_confidence, flags)
    window = seg.iloc[local_start:local_end + 1]
    speed = signals["speed"][local_start:local_end + 1]
    accel = signals["accel"][local_start:local_end + 1]
    distance = signals["distance"][local_start:local_end + 1]
    closing = signals["closing"][local_start:local_end + 1]
    car_accel = signals["car_accel"][local_start:local_end + 1]
    head_turn = np.abs(signals["head_turn"][local_start:local_end + 1])
    head_body = np.abs(signals["head_body"][local_start:local_end + 1])
    displacement, path, progress_ratio = displacement_stats(window)

    events.append({
        "macro_segment_id": int(macro_row["segment_id"]),
        "macro_label": str(macro_row["macro_label"]),
        "event_label": label,
        "start_idx": int(window.index[0]),
        "end_idx": int(window.index[-1]),
        "start_frame": window[FRAME_COL].iloc[0] if FRAME_COL in window.columns else np.nan,
        "end_frame": window[FRAME_COL].iloc[-1] if FRAME_COL in window.columns else np.nan,
        "ScenarioTime_start": window[TIME_COL].iloc[0] if TIME_COL in window.columns else np.nan,
        "ScenarioTime_end": window[TIME_COL].iloc[-1] if TIME_COL in window.columns else np.nan,
        "time_start_sec": float(time_sec[local_start]),
        "time_end_sec": float(time_sec[local_end]),
        "duration_sec": duration,
        "confidence": confidence,
        "evidence_flags": "|".join(flags) if flags else "none",
        "rule_reason": reason,
        "mean_ped_speed": safe_stat(speed, np.nanmean),
        "min_ped_speed": safe_stat(speed, np.nanmin),
        "max_ped_speed": safe_stat(speed, np.nanmax),
        "mean_ped_accel": safe_stat(accel, np.nanmean),
        "min_ped_accel": safe_stat(accel, np.nanmin),
        "max_ped_accel": safe_stat(accel, np.nanmax),
        "min_car_ped_distance": safe_stat(distance, np.nanmin),
        "mean_car_ped_distance": safe_stat(distance, np.nanmean),
        "max_distance_closing": safe_stat(closing, np.nanmax),
        "mean_distance_closing": safe_stat(closing, np.nanmean),
        "min_car_accel": safe_stat(car_accel, np.nanmin),
        "max_abs_head_turn_rate": safe_stat(head_turn, np.nanmax),
        "max_abs_head_body_yaw_diff": safe_stat(head_body, np.nanmax),
        "ped_displacement_xz": displacement,
        "ped_path_length_xz": path,
        "ped_progress_ratio": progress_ratio,
    })


def merge_event_runs(events):
    if not events:
        return []

    merged = []
    for event in sorted(events, key=lambda r: (r["macro_segment_id"], r["event_label"], r["start_idx"])):
        if not merged:
            merged.append(event)
            continue

        prev = merged[-1]
        same_group = (
            prev["macro_segment_id"] == event["macro_segment_id"]
            and prev["event_label"] == event["event_label"]
            and event["time_start_sec"] - prev["time_end_sec"] <= MERGE_GAP_SECONDS
        )

        if not same_group:
            merged.append(event)
            continue

        prev["end_idx"] = max(prev["end_idx"], event["end_idx"])
        prev["end_frame"] = event["end_frame"]
        prev["ScenarioTime_end"] = event["ScenarioTime_end"]
        prev["time_end_sec"] = event["time_end_sec"]
        prev["duration_sec"] = prev["time_end_sec"] - prev["time_start_sec"]
        prev["confidence"] = max(prev["confidence"], event["confidence"])
        prev_flags = set(str(prev["evidence_flags"]).split("|")) - {"none"}
        next_flags = set(str(event["evidence_flags"]).split("|")) - {"none"}
        flags = sorted(prev_flags | next_flags)
        prev["evidence_flags"] = "|".join(flags) if flags else "none"
        prev["rule_reason"] = f"{prev['rule_reason']}; {event['rule_reason']}"

    return sorted(merged, key=lambda r: (r["macro_segment_id"], r["start_idx"], r["event_label"]))


def detect_hesitation_events(events, macro_row, seg, signals):
    time_sec = signals["time_sec"]
    speed = signals["speed"]
    accel = signals["accel"]
    decel_runs = duration_mask_runs(accel <= DECEL_THRESHOLD, time_sec, MIN_EVENT_SECONDS)

    for start, end in decel_runs:
        before = max(0, start - seconds_to_frames(time_sec, 0.75))
        after = min(len(speed) - 1, end + seconds_to_frames(time_sec, HESITATION_REBOUND_SECONDS))

        prev_speed = safe_stat(speed[before:start + 1], np.nanmax)
        event_low = safe_stat(speed[start:end + 1], np.nanmin)
        future_speed = safe_stat(speed[end:after + 1], np.nanmax)

        drop = prev_speed - event_low if np.isfinite(prev_speed) and np.isfinite(event_low) else np.nan
        rebound = future_speed - event_low if np.isfinite(future_speed) and np.isfinite(event_low) else np.nan

        flags = evidence_for_window(signals, start, min(after, len(speed) - 1))
        supported = bool(set(flags) & {"car_close", "car_ped_closing", "head_turning", "head_body_offset"})

        if (
            np.isfinite(drop)
            and np.isfinite(rebound)
            and drop >= HESITATION_DROP_SPEED
            and rebound >= HESITATION_REBOUND_SPEED
            and supported
        ):
            add_event(
                events,
                macro_row,
                seg,
                signals,
                start,
                min(after, len(speed) - 1),
                "hesitation",
                0.70,
                "deceleration with later speed rebound and interaction/head evidence",
            )


def detect_micro_events_for_macro(macro_row, df):
    start_idx = int(macro_row["start_idx"])
    end_idx = int(macro_row["end_idx"])
    start_idx = max(0, start_idx)
    end_idx = min(len(df) - 1, end_idx)

    seg = df.iloc[start_idx:end_idx + 1].copy()
    time_sec = seg["elapsed_time"].to_numpy(dtype=float)

    ped_speed_col = pick_col(seg, PED_SPEED_COLS, required=True, purpose="pedestrian speed column")
    ped_accel_col = pick_col(seg, PED_ACCEL_COLS)
    distance_col = pick_col(seg, DISTANCE_COLS)
    closing_col = pick_col(seg, CLOSING_COLS)
    car_accel_col = pick_col(seg, CAR_ACCEL_COLS)
    head_turn_col = pick_col(seg, HEAD_TURN_COLS)
    head_body_col = pick_col(seg, HEAD_BODY_DIFF_COLS)
    avatar_gap_col = pick_col(seg, AVATAR_GAP_COLS)

    speed_raw = numeric_series(seg, ped_speed_col).to_numpy(dtype=float)
    speed = smooth_signal(speed_raw, time_sec)
    accel = numeric_series(seg, ped_accel_col).to_numpy(dtype=float) if ped_accel_col else gradient(speed, time_sec)
    accel = smooth_signal(accel, time_sec, seconds=0.25)
    slope = gradient(speed, time_sec)

    avatar_gap = numeric_series(seg, avatar_gap_col).to_numpy(dtype=float)
    finite_gap = avatar_gap[np.isfinite(avatar_gap)]
    if len(finite_gap):
        avatar_gap_threshold = max(
            NOISE_GAP_MIN_ABSOLUTE,
            float(np.nanquantile(finite_gap, NOISE_GAP_QUANTILE)),
        )
    else:
        avatar_gap_threshold = np.inf

    signals = {
        "time_sec": time_sec,
        "speed": speed,
        "accel": accel,
        "slope": slope,
        "distance": numeric_series(seg, distance_col).to_numpy(dtype=float),
        "closing": numeric_series(seg, closing_col, default=0.0).to_numpy(dtype=float),
        "car_accel": numeric_series(seg, car_accel_col).to_numpy(dtype=float),
        "head_turn": numeric_series(seg, head_turn_col, default=0.0).to_numpy(dtype=float),
        "head_body": numeric_series(seg, head_body_col, default=0.0).to_numpy(dtype=float),
        "avatar_gap": avatar_gap,
        "avatar_gap_threshold": avatar_gap_threshold,
    }

    events = []

    for start, end in duration_mask_runs(speed < STOP_SPEED, time_sec, MIN_LOW_MOTION_SECONDS):
        label = "stopped_low_motion"
        if safe_stat(signals["distance"][start:end + 1], np.nanmin) <= CAR_CLOSE_DISTANCE:
            label = "stopped_low_motion_near_car"
        add_event(events, macro_row, seg, signals, start, end, label, 0.78, "speed below stop threshold")

    for start, end in duration_mask_runs((speed >= WALK_SPEED) & (accel >= ACCEL_THRESHOLD), time_sec, MIN_EVENT_SECONDS):
        start_speed = safe_stat(speed[start:start + 2], np.nanmean)
        future_end = min(len(speed) - 1, end + seconds_to_frames(time_sec, 1.0))
        future_speed = safe_stat(speed[start:future_end + 1], np.nanmax)
        if np.isfinite(start_speed) and np.isfinite(future_speed) and future_speed - start_speed >= 0.20:
            label = "acceleration_commitment" if start_speed < CROSSING_SPEED else "acceleration_burst"
            add_event(events, macro_row, seg, signals, start, future_end, label, 0.72, "positive acceleration with speed gain")

    for start, end in duration_mask_runs(accel <= DECEL_THRESHOLD, time_sec, MIN_EVENT_SECONDS):
        label = "deceleration"
        if (
            safe_stat(signals["distance"][start:end + 1], np.nanmin) <= CAR_CLOSE_DISTANCE
            or safe_stat(signals["closing"][start:end + 1], np.nanmax) >= CLOSING_THRESHOLD
        ):
            label = "yielding_slowing"
        add_event(events, macro_row, seg, signals, start, end, label, 0.68, "negative pedestrian acceleration")

    for start, end in duration_mask_runs(
        (speed >= CROSSING_SPEED) & (np.abs(slope) <= max(0.20, SLOPE_DEADBAND)),
        time_sec,
        MIN_CROSSING_SECONDS,
    ):
        add_event(events, macro_row, seg, signals, start, end, "crossing_movement", 0.70, "sustained walking/crossing speed")

    for start, end in duration_mask_runs(signals["distance"] <= CAR_CLOSE_DISTANCE, time_sec, MIN_EVENT_SECONDS):
        add_event(events, macro_row, seg, signals, start, end, "close_interaction", 0.82, "car-pedestrian distance below threshold")

    head_mask = (
        (np.abs(signals["head_turn"]) >= HEAD_TURN_RATE_THRESHOLD)
        | (np.abs(signals["head_body"]) >= HEAD_BODY_DIFF_THRESHOLD)
    )
    for start, end in duration_mask_runs(head_mask, time_sec, MIN_HEAD_CHECK_SECONDS):
        add_event(events, macro_row, seg, signals, start, end, "head_direction_check", 0.64, "head turn or head/body yaw difference")

    noise_mask = signals["avatar_gap"] >= signals["avatar_gap_threshold"]
    for start, end in duration_mask_runs(noise_mask, time_sec, MIN_EVENT_SECONDS):
        add_event(events, macro_row, seg, signals, start, end, "possible_noise_artifact", 0.58, "avatar and VR pedestrian position gap high")

    detect_hesitation_events(events, macro_row, seg, signals)

    return events


def assign_frame_labels(df, events):
    frame_df = df.copy()
    frame_df["micro_event_ids"] = ""
    frame_df["active_micro_events"] = ""
    frame_df["primary_micro_label"] = "none"
    frame_df["micro_event_count"] = 0

    priority = {
        "hesitation": 90,
        "yielding_slowing": 80,
        "stopped_low_motion_near_car": 75,
        "stopped_low_motion": 70,
        "acceleration_commitment": 65,
        "deceleration": 60,
        "close_interaction": 55,
        "head_direction_check": 50,
        "crossing_movement": 45,
        "acceleration_burst": 40,
        "possible_noise_artifact": 35,
    }

    active_labels = [[] for _ in range(len(frame_df))]
    active_ids = [[] for _ in range(len(frame_df))]

    for _, event in events.iterrows():
        event_id = str(event["micro_event_id"])
        label = str(event["event_label"])
        start = int(event["start_idx"])
        end = int(event["end_idx"])
        for idx in range(max(0, start), min(len(frame_df) - 1, end) + 1):
            active_labels[idx].append(label)
            active_ids[idx].append(event_id)

    primary = []
    for labels in active_labels:
        if not labels:
            primary.append("none")
            continue
        primary.append(sorted(labels, key=lambda label: priority.get(label, 0), reverse=True)[0])

    frame_df["micro_event_ids"] = ["|".join(ids) for ids in active_ids]
    frame_df["active_micro_events"] = ["|".join(labels) for labels in active_labels]
    frame_df["primary_micro_label"] = primary
    frame_df["micro_event_count"] = [len(labels) for labels in active_labels]
    return frame_df


def build_summary(events):
    if events.empty:
        return pd.DataFrame(columns=[
            "macro_segment_id",
            "macro_label",
            "event_label",
            "event_count",
            "total_duration_sec",
            "mean_duration_sec",
            "mean_confidence",
        ])

    return (
        events
        .groupby(["macro_segment_id", "macro_label", "event_label"], dropna=False)
        .agg(
            event_count=("micro_event_id", "count"),
            total_duration_sec=("duration_sec", "sum"),
            mean_duration_sec=("duration_sec", "mean"),
            mean_confidence=("confidence", "mean"),
            first_time_sec=("time_start_sec", "min"),
            last_time_sec=("time_end_sec", "max"),
            min_car_ped_distance=("min_car_ped_distance", "min"),
            max_distance_closing=("max_distance_closing", "max"),
            mean_ped_speed=("mean_ped_speed", "mean"),
        )
        .reset_index()
        .sort_values(["macro_segment_id", "first_time_sec", "event_label"])
    )


def plot_timeline(frame_df, macro_df, events, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Skipping plot: matplotlib is not installed in this Python environment.")
        return False

    if events.empty:
        return False

    label_order = list(dict.fromkeys(events["event_label"].tolist()))
    label_y = {label: i for i, label in enumerate(label_order)}

    fig, ax = plt.subplots(figsize=(18, max(6, 0.35 * len(label_order) + 4)))
    time = frame_df["elapsed_time"].to_numpy(dtype=float)

    speed_col = pick_col(frame_df, PED_SPEED_COLS)
    distance_col = pick_col(frame_df, DISTANCE_COLS)
    speed = pd.to_numeric(frame_df[speed_col], errors="coerce").to_numpy(dtype=float)

    ax.plot(time, speed, color="black", linewidth=1.8, label=speed_col)
    ax.set_xlabel("Scenario elapsed time, seconds")
    ax.set_ylabel("Pedestrian speed XZ smooth")

    y_scale = max(np.nanmax(speed) if np.any(np.isfinite(speed)) else 1.0, 1.0)
    event_base = y_scale * 1.10
    event_step = y_scale * 0.08

    for _, row in macro_df.iterrows():
        ax.axvspan(row["time_start_sec"], row["time_end_sec"], color="#eeeeee", alpha=0.25)
        ax.axvline(row["time_start_sec"], color="#999999", linewidth=0.8, alpha=0.6)

    for _, event in events.iterrows():
        y = event_base + event_step * label_y[event["event_label"]]
        ax.hlines(
            y,
            event["time_start_sec"],
            event["time_end_sec"],
            linewidth=5,
            label=event["event_label"] if event["event_label"] not in ax.get_legend_handles_labels()[1] else None,
        )

    ax.set_ylim(0, event_base + event_step * (len(label_order) + 1))
    ax.grid(True, alpha=0.25)

    if distance_col is not None:
        ax2 = ax.twinx()
        distance = pd.to_numeric(frame_df[distance_col], errors="coerce").to_numpy(dtype=float)
        ax2.plot(time, distance, color="#7777cc", linewidth=1.0, alpha=0.55, label=distance_col)
        ax2.set_ylabel("Car-pedestrian distance")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=8)
    ax.set_title("Pedestrian Micro-Events Within Macro Segments")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return True


def main():
    args = parse_args()

    if not args.feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {args.feature_csv}")
    if not args.macro_csv.exists():
        raise FileNotFoundError(f"Macro segment CSV not found: {args.macro_csv}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_df = clean_columns(pd.read_csv(args.feature_csv))
    macro_df = clean_columns(pd.read_csv(args.macro_csv))

    required_macro_cols = ["segment_id", "macro_label", "start_idx", "end_idx", "time_start_sec", "time_end_sec"]
    missing_macro = [col for col in required_macro_cols if col not in macro_df.columns]
    if missing_macro:
        raise ValueError(f"Macro segment CSV missing required columns: {missing_macro}")

    pick_col(feature_df, PED_SPEED_COLS, required=True, purpose="VR-derived pedestrian speed column")

    feature_df = feature_df.sort_values(TIME_COL).reset_index(drop=True)
    feature_df["elapsed_time"] = build_elapsed_time_seconds(feature_df)

    all_events = []
    for _, macro_row in macro_df.iterrows():
        all_events.extend(detect_micro_events_for_macro(macro_row, feature_df))

    all_events = merge_event_runs(all_events)
    events_df = pd.DataFrame(all_events)
    if not events_df.empty:
        events_df.insert(0, "micro_event_id", np.arange(len(events_df), dtype=int))
        events_df["event_order"] = events_df.groupby("macro_segment_id").cumcount() + 1
        events_df["n_frames"] = events_df["end_idx"] - events_df["start_idx"] + 1

    frame_df = assign_frame_labels(feature_df, events_df)
    summary_df = build_summary(events_df)

    events_csv = output_dir / "micro_events_PedNYC1_scenario3_v2.csv"
    frame_csv = output_dir / "features_with_micro_labels_PedNYC1_scenario3_v2.csv"
    summary_csv = output_dir / "micro_event_summary_by_macro_PedNYC1_scenario3_v2.csv"
    plot_path = output_dir / "micro_events_timeline_PedNYC1_scenario3_v2.png"

    events_df.to_csv(events_csv, index=False)
    frame_df.to_csv(frame_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    wrote_plot = False
    if not args.no_plot:
        wrote_plot = plot_timeline(frame_df, macro_df, events_df, plot_path)

    print("\nDone.")
    print(f"Feature CSV: {args.feature_csv}")
    print(f"Macro segment CSV: {args.macro_csv}")
    print(f"Saved micro-events CSV: {events_csv}")
    print(f"Saved frame-level labels CSV: {frame_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    if wrote_plot:
        print(f"Saved timeline plot: {plot_path}")

    if events_df.empty:
        print("\nNo micro-events detected with the current thresholds.")
    else:
        print("\nDetected micro-events by label:")
        print(events_df["event_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
