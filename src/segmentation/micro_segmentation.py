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
FRAME_COL = "Frame Number"

PED_SPEED_COLS = ["ped_speed_xz_smooth", "ped_speed_xz"]
PED_ACCEL_COLS = ["ped_accel_xz_smooth", "ped_accel_xz"]
DISTANCE_COLS = ["car_ped_distance_xz_smooth", "car_ped_distance_xz"]
CLOSING_COLS = ["distance_closing_smooth", "distance_closing"]
DISTANCE_CHANGE_COLS = ["distance_change"]
CAR_SPEED_COLS = ["car_speed_xz_smooth", "car_speed_xz"]
CAR_ACCEL_COLS = ["car_accel_xz_smooth", "car_accel_xz"]
HEAD_YAW_COLS = ["head_yaw"]
HEAD_TURN_COLS = ["head_turn_rate"]
HEAD_BODY_DIFF_COLS = ["head_body_yaw_diff"]
AVATAR_GAP_COLS = ["avatar_vr_gap_xz"]
STOPPED_FLAG_COLS = ["ped_stopped"]
WALKING_FLAG_COLS = ["ped_walking"]
CLOSE_INTERACTION_FLAG_COLS = ["close_interaction"]

# Tunable thresholds. These are intentionally conservative and near the values
# already used elsewhere in this repo. Macro segments are only parent intervals;
# all rules below may produce repeated and overlapping micro-events inside one
# macro interval.
STOP_SPEED = 0.15
WALK_SPEED = 0.30
CROSSING_SPEED = 0.60
ACCEL_THRESHOLD = 0.25
DECEL_THRESHOLD = -0.25
SLOPE_STABLE_LIMIT = 0.20
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

EVENT_METADATA = {
    "stopped_low_motion": {
        "family": "movement",
        "kind": "direct_observation",
        "color": "#4E79A7",
        "description": "VR-derived pedestrian speed remains below the stop threshold.",
    },
    "stopped_low_motion_near_car": {
        "family": "interaction",
        "kind": "direct_observation",
        "color": "#2F5D8C",
        "description": "Low pedestrian motion occurs while the car is close.",
    },
    "acceleration_commitment": {
        "family": "movement",
        "kind": "inferred",
        "color": "#59A14F",
        "description": "Pedestrian speed rises from walking threshold with positive acceleration.",
    },
    "acceleration_burst": {
        "family": "movement",
        "kind": "direct_observation",
        "color": "#8CD17D",
        "description": "Pedestrian is already moving and accelerates further.",
    },
    "deceleration": {
        "family": "movement",
        "kind": "direct_observation",
        "color": "#F28E2B",
        "description": "Pedestrian acceleration is negative for a sustained interval.",
    },
    "yielding_slowing": {
        "family": "interaction",
        "kind": "inferred",
        "color": "#E15759",
        "description": "Pedestrian decelerates with close/closing car evidence.",
    },
    "crossing_movement": {
        "family": "movement",
        "kind": "direct_observation",
        "color": "#76B7B2",
        "description": "Sustained walking/crossing-speed motion from VR-derived speed.",
    },
    "close_interaction": {
        "family": "interaction",
        "kind": "direct_observation",
        "color": "#B07AA1",
        "description": "Car-pedestrian distance is below the close-interaction threshold.",
    },
    "head_direction_check": {
        "family": "attention",
        "kind": "direct_observation",
        "color": "#EDC948",
        "description": "Head turn rate or head/body yaw difference exceeds threshold.",
    },
    "hesitation": {
        "family": "interaction",
        "kind": "inferred",
        "color": "#FF9DA7",
        "description": "Speed drop, later rebound, and interaction/head evidence all occur.",
    },
    "possible_noise_artifact": {
        "family": "quality",
        "kind": "audit",
        "color": "#9C755F",
        "description": "Avatar/VR gap is unusually high for this macro segment.",
    },
}

PRIMARY_LABEL_PRIORITY = {
    # Explicit primary-label rule: interpretive interaction labels outrank
    # simple motion labels; quality/audit labels never outrank behavior labels.
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect repeated micro-behavior events inside existing macro segments."
    )
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--macro-csv", type=Path, default=DEFAULT_MACRO_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true", help="Skip writing PNG plots.")
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


def mode_or_blank(series):
    values = series.dropna().astype(str)
    if values.empty:
        return ""
    return values.mode().iloc[0]


def evidence_for_window(signals, local_start, local_end):
    sl = slice(local_start, local_end + 1)
    flags = []

    min_distance = safe_stat(signals["distance"][sl], np.nanmin)
    max_closing = safe_stat(signals["closing"][sl], np.nanmax)
    mean_distance_change = safe_stat(signals["distance_change"][sl], np.nanmean)
    min_car_accel = safe_stat(signals["car_accel"][sl], np.nanmin)
    max_head_turn = safe_stat(np.abs(signals["head_turn"][sl]), np.nanmax)
    max_head_body = safe_stat(np.abs(signals["head_body"][sl]), np.nanmax)
    max_avatar_gap = safe_stat(signals["avatar_gap"][sl], np.nanmax)
    stopped_fraction = fraction(signals["stopped_flag"][sl], lambda x: x > 0)
    walking_fraction = fraction(signals["walking_flag"][sl], lambda x: x > 0)
    close_flag_fraction = fraction(signals["close_interaction_flag"][sl], lambda x: x > 0)

    if np.isfinite(min_distance) and min_distance <= CAR_CLOSE_DISTANCE:
        flags.append("car_close")
    if np.isfinite(min_distance) and min_distance <= CAR_VERY_CLOSE_DISTANCE:
        flags.append("car_very_close")
    if np.isfinite(max_closing) and max_closing >= CLOSING_THRESHOLD:
        flags.append("car_ped_closing")
    if np.isfinite(mean_distance_change) and mean_distance_change < 0:
        flags.append("distance_decreasing")
    if np.isfinite(min_car_accel) and min_car_accel <= CAR_BRAKE_ACCEL:
        flags.append("car_braking")
    if np.isfinite(max_head_turn) and max_head_turn >= HEAD_TURN_RATE_THRESHOLD:
        flags.append("head_turning")
    if np.isfinite(max_head_body) and max_head_body >= HEAD_BODY_DIFF_THRESHOLD:
        flags.append("head_body_offset")
    if np.isfinite(stopped_fraction) and stopped_fraction >= 0.50:
        flags.append("stopped_flag")
    if np.isfinite(walking_fraction) and walking_fraction >= 0.50:
        flags.append("walking_flag")
    if np.isfinite(close_flag_fraction) and close_flag_fraction >= 0.50:
        flags.append("close_interaction_flag")

    avatar_gap_threshold = signals.get("avatar_gap_threshold", np.inf)
    if np.isfinite(max_avatar_gap) and max_avatar_gap >= avatar_gap_threshold:
        flags.append("avatar_vr_gap_high")

    return flags


def split_evidence(label, flags, reason):
    primary_by_label = {
        "stopped_low_motion": "ped_speed_below_stop_threshold",
        "stopped_low_motion_near_car": "ped_speed_below_stop_threshold+car_close",
        "acceleration_commitment": "ped_accel_positive+speed_gain",
        "acceleration_burst": "ped_accel_positive+speed_gain",
        "deceleration": "ped_accel_negative",
        "yielding_slowing": "ped_accel_negative+car_close_or_closing",
        "crossing_movement": "sustained_vr_ped_speed",
        "close_interaction": "car_ped_distance_below_threshold",
        "head_direction_check": "head_turn_or_head_body_yaw",
        "hesitation": "speed_drop+rebound+interaction_or_head_evidence",
        "possible_noise_artifact": "avatar_vr_gap_upper_tail",
    }
    primary = primary_by_label.get(label, reason)
    supporting = [flag for flag in flags if flag not in set(primary.split("+"))]
    return primary, "|".join(supporting) if supporting else "none"


def confidence_from_flags(base, flags):
    score = base + 0.06 * min(4, len(flags))
    if "avatar_vr_gap_high" in flags:
        score -= 0.08
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
    primary_evidence, supporting_evidence = split_evidence(label, flags, reason)
    meta = EVENT_METADATA.get(label, {})
    window = seg.iloc[local_start:local_end + 1]
    speed = signals["speed"][local_start:local_end + 1]
    accel = signals["accel"][local_start:local_end + 1]
    distance = signals["distance"][local_start:local_end + 1]
    distance_change = signals["distance_change"][local_start:local_end + 1]
    closing = signals["closing"][local_start:local_end + 1]
    car_speed = signals["car_speed"][local_start:local_end + 1]
    car_accel = signals["car_accel"][local_start:local_end + 1]
    head_yaw = signals["head_yaw"][local_start:local_end + 1]
    head_turn = np.abs(signals["head_turn"][local_start:local_end + 1])
    head_body = np.abs(signals["head_body"][local_start:local_end + 1])
    avatar_gap = signals["avatar_gap"][local_start:local_end + 1]
    displacement, path, progress_ratio = displacement_stats(window)

    head_yaw_change = np.nan
    valid_head_yaw = head_yaw[np.isfinite(head_yaw)]
    if len(valid_head_yaw) >= 2:
        head_yaw_change = float(((valid_head_yaw[-1] - valid_head_yaw[0] + 180) % 360) - 180)

    signal_cols = [
        signals["source_columns"].get(name)
        for name in [
            "ped_speed",
            "ped_accel",
            "distance",
            "closing",
            "distance_change",
            "car_speed",
            "car_accel",
            "head_yaw",
            "head_turn",
            "head_body",
            "avatar_gap",
            "stopped_flag",
            "walking_flag",
            "close_interaction_flag",
        ]
        if signals["source_columns"].get(name)
    ]

    events.append({
        "macro_segment_id": int(macro_row["segment_id"]),
        "macro_label": str(macro_row["macro_label"]),
        "event_label": label,
        "event_family": meta.get("family", "unknown"),
        "label_kind": meta.get("kind", "unknown"),
        "start_idx": int(window.index[0]),
        "end_idx": int(window.index[-1]),
        "start_frame": window[FRAME_COL].iloc[0] if FRAME_COL in window.columns else np.nan,
        "end_frame": window[FRAME_COL].iloc[-1] if FRAME_COL in window.columns else np.nan,
        "ScenarioTime_start": window[TIME_COL].iloc[0] if TIME_COL in window.columns else np.nan,
        "ScenarioTime_end": window[TIME_COL].iloc[-1] if TIME_COL in window.columns else np.nan,
        "start_time_sec": float(time_sec[local_start]),
        "end_time_sec": float(time_sec[local_end]),
        "duration_sec": duration,
        "confidence": confidence,
        "primary_evidence": primary_evidence,
        "supporting_evidence": supporting_evidence,
        "evidence_flags": "|".join(flags) if flags else "none",
        "signals_used": "|".join(signal_cols) if signal_cols else "none",
        "notes": f"{reason}. {meta.get('description', '')}".strip(),
        "ped_speed_mean": safe_stat(speed, np.nanmean),
        "ped_speed_min": safe_stat(speed, np.nanmin),
        "ped_speed_max": safe_stat(speed, np.nanmax),
        "ped_accel_mean": safe_stat(accel, np.nanmean),
        "ped_accel_min": safe_stat(accel, np.nanmin),
        "ped_accel_max": safe_stat(accel, np.nanmax),
        "distance_min": safe_stat(distance, np.nanmin),
        "distance_mean": safe_stat(distance, np.nanmean),
        "distance_change_mean": safe_stat(distance_change, np.nanmean),
        "distance_closing_mean": safe_stat(closing, np.nanmean),
        "distance_closing_max": safe_stat(closing, np.nanmax),
        "car_speed_mean": safe_stat(car_speed, np.nanmean),
        "car_accel_mean": safe_stat(car_accel, np.nanmean),
        "car_accel_min": safe_stat(car_accel, np.nanmin),
        "head_yaw_change": head_yaw_change,
        "head_turn_rate_max_abs": safe_stat(head_turn, np.nanmax),
        "head_body_yaw_diff_max_abs": safe_stat(head_body, np.nanmax),
        "avatar_vr_gap_max": safe_stat(avatar_gap, np.nanmax),
        "ped_displacement_xz": displacement,
        "ped_path_length_xz": path,
        "ped_progress_ratio": progress_ratio,
        "movement_state_mode": mode_or_blank(window["movement_state_v3"]) if "movement_state_v3" in window.columns else "",
        "head_state_mode": mode_or_blank(window["head_state_v3"]) if "head_state_v3" in window.columns else "",
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
            and event["start_time_sec"] - prev["end_time_sec"] <= MERGE_GAP_SECONDS
        )
        if not same_group:
            merged.append(event)
            continue

        prev["end_idx"] = max(prev["end_idx"], event["end_idx"])
        prev["end_frame"] = event["end_frame"]
        prev["ScenarioTime_end"] = event["ScenarioTime_end"]
        prev["end_time_sec"] = event["end_time_sec"]
        prev["duration_sec"] = prev["end_time_sec"] - prev["start_time_sec"]
        prev["confidence"] = max(prev["confidence"], event["confidence"])
        prev_flags = set(str(prev["evidence_flags"]).split("|")) - {"none"}
        next_flags = set(str(event["evidence_flags"]).split("|")) - {"none"}
        flags = sorted(prev_flags | next_flags)
        prev["evidence_flags"] = "|".join(flags) if flags else "none"
        prev["supporting_evidence"] = prev["evidence_flags"]
        prev["notes"] = f"{prev['notes']}; merged adjacent event: {event['notes']}"

    return sorted(merged, key=lambda r: (r["macro_segment_id"], r["start_idx"], r["event_label"]))


def detect_hesitation_events(events, macro_row, seg, signals):
    # Hesitation is intentionally interpretive: require a speed drop, a later
    # rebound, and at least one independent interaction/head cue.
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
        flags = evidence_for_window(signals, start, after)
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
                after,
                "hesitation",
                0.70,
                "speed drops, later rebounds, and interaction/head evidence is present",
            )


def detect_micro_events_for_macro(macro_row, df):
    start_idx = max(0, int(macro_row["start_idx"]))
    end_idx = min(len(df) - 1, int(macro_row["end_idx"]))
    seg = df.iloc[start_idx:end_idx + 1].copy()
    time_sec = seg["elapsed_time_sec"].to_numpy(dtype=float)

    ped_speed_col = pick_col(seg, PED_SPEED_COLS, required=True, purpose="VR-derived pedestrian speed column")
    ped_accel_col = pick_col(seg, PED_ACCEL_COLS)
    distance_col = pick_col(seg, DISTANCE_COLS)
    closing_col = pick_col(seg, CLOSING_COLS)
    distance_change_col = pick_col(seg, DISTANCE_CHANGE_COLS)
    car_speed_col = pick_col(seg, CAR_SPEED_COLS)
    car_accel_col = pick_col(seg, CAR_ACCEL_COLS)
    head_yaw_col = pick_col(seg, HEAD_YAW_COLS)
    head_turn_col = pick_col(seg, HEAD_TURN_COLS)
    head_body_col = pick_col(seg, HEAD_BODY_DIFF_COLS)
    avatar_gap_col = pick_col(seg, AVATAR_GAP_COLS)
    stopped_flag_col = pick_col(seg, STOPPED_FLAG_COLS)
    walking_flag_col = pick_col(seg, WALKING_FLAG_COLS)
    close_interaction_flag_col = pick_col(seg, CLOSE_INTERACTION_FLAG_COLS)

    speed_raw = numeric_series(seg, ped_speed_col).to_numpy(dtype=float)
    speed = smooth_signal(speed_raw, time_sec)
    accel = numeric_series(seg, ped_accel_col).to_numpy(dtype=float) if ped_accel_col else gradient(speed, time_sec)
    accel = smooth_signal(accel, time_sec, seconds=0.25)
    slope = gradient(speed, time_sec)
    distance = numeric_series(seg, distance_col).to_numpy(dtype=float)
    closing = numeric_series(seg, closing_col, default=0.0).to_numpy(dtype=float)
    distance_change = numeric_series(seg, distance_change_col).to_numpy(dtype=float)
    car_speed = numeric_series(seg, car_speed_col).to_numpy(dtype=float)
    car_accel = numeric_series(seg, car_accel_col).to_numpy(dtype=float)
    head_yaw = numeric_series(seg, head_yaw_col).to_numpy(dtype=float)
    head_turn = numeric_series(seg, head_turn_col, default=0.0).to_numpy(dtype=float)
    head_body = numeric_series(seg, head_body_col, default=0.0).to_numpy(dtype=float)
    avatar_gap = numeric_series(seg, avatar_gap_col).to_numpy(dtype=float)
    stopped_flag = numeric_series(seg, stopped_flag_col, default=0.0).to_numpy(dtype=float)
    walking_flag = numeric_series(seg, walking_flag_col, default=0.0).to_numpy(dtype=float)
    close_interaction_flag = numeric_series(seg, close_interaction_flag_col, default=0.0).to_numpy(dtype=float)

    finite_gap = avatar_gap[np.isfinite(avatar_gap)]
    avatar_gap_threshold = (
        max(NOISE_GAP_MIN_ABSOLUTE, float(np.nanquantile(finite_gap, NOISE_GAP_QUANTILE)))
        if len(finite_gap)
        else np.inf
    )

    signals = {
        "time_sec": time_sec,
        "speed": speed,
        "accel": accel,
        "slope": slope,
        "distance": distance,
        "closing": closing,
        "distance_change": distance_change,
        "car_speed": car_speed,
        "car_accel": car_accel,
        "head_yaw": head_yaw,
        "head_turn": head_turn,
        "head_body": head_body,
        "avatar_gap": avatar_gap,
        "avatar_gap_threshold": avatar_gap_threshold,
        "stopped_flag": stopped_flag,
        "walking_flag": walking_flag,
        "close_interaction_flag": close_interaction_flag,
        "source_columns": {
            "ped_speed": ped_speed_col,
            "ped_accel": ped_accel_col,
            "distance": distance_col,
            "closing": closing_col,
            "distance_change": distance_change_col,
            "car_speed": car_speed_col,
            "car_accel": car_accel_col,
            "head_yaw": head_yaw_col,
            "head_turn": head_turn_col,
            "head_body": head_body_col,
            "avatar_gap": avatar_gap_col,
            "stopped_flag": stopped_flag_col,
            "walking_flag": walking_flag_col,
            "close_interaction_flag": close_interaction_flag_col,
        },
    }

    events = []

    # Direct: low VR-derived pedestrian speed, strengthened by stopped flag.
    low_motion_mask = (speed < STOP_SPEED) | (stopped_flag > 0)
    for start, end in duration_mask_runs(low_motion_mask, time_sec, MIN_LOW_MOTION_SECONDS):
        label = "stopped_low_motion"
        if safe_stat(distance[start:end + 1], np.nanmin) <= CAR_CLOSE_DISTANCE:
            label = "stopped_low_motion_near_car"
        add_event(events, macro_row, seg, signals, start, end, label, 0.78, "low speed or stopped flag persists")

    # Direct/inferred: positive acceleration with a measurable speed gain.
    accel_mask = ((speed >= WALK_SPEED) | (walking_flag > 0)) & (accel >= ACCEL_THRESHOLD)
    for start, end in duration_mask_runs(accel_mask, time_sec, MIN_EVENT_SECONDS):
        start_speed = safe_stat(speed[start:start + 2], np.nanmean)
        future_end = min(len(speed) - 1, end + seconds_to_frames(time_sec, 1.0))
        future_speed = safe_stat(speed[start:future_end + 1], np.nanmax)
        if np.isfinite(start_speed) and np.isfinite(future_speed) and future_speed - start_speed >= 0.20:
            label = "acceleration_commitment" if start_speed < CROSSING_SPEED else "acceleration_burst"
            add_event(events, macro_row, seg, signals, start, future_end, label, 0.72, "positive acceleration with later speed gain")

    # Direct/inferred: negative acceleration; only call it yielding when car cues support it.
    for start, end in duration_mask_runs(accel <= DECEL_THRESHOLD, time_sec, MIN_EVENT_SECONDS):
        label = "deceleration"
        interaction_supported = (
            safe_stat(distance[start:end + 1], np.nanmin) <= CAR_CLOSE_DISTANCE
            or safe_stat(closing[start:end + 1], np.nanmax) >= CLOSING_THRESHOLD
            or fraction(close_interaction_flag[start:end + 1], lambda x: x > 0) >= 0.5
        )
        if interaction_supported:
            label = "yielding_slowing"
        add_event(events, macro_row, seg, signals, start, end, label, 0.68, "negative pedestrian acceleration")

    # Direct: sustained moving/crossing speed with stable local slope.
    crossing_mask = ((speed >= CROSSING_SPEED) | (walking_flag > 0)) & (np.abs(slope) <= SLOPE_STABLE_LIMIT)
    for start, end in duration_mask_runs(crossing_mask, time_sec, MIN_CROSSING_SECONDS):
        add_event(events, macro_row, seg, signals, start, end, "crossing_movement", 0.70, "sustained walking-speed motion")

    # Direct: close car-pedestrian distance or an existing close-interaction flag.
    close_mask = (distance <= CAR_CLOSE_DISTANCE) | (close_interaction_flag > 0)
    for start, end in duration_mask_runs(close_mask, time_sec, MIN_EVENT_SECONDS):
        add_event(events, macro_row, seg, signals, start, end, "close_interaction", 0.82, "car-pedestrian distance or close flag indicates interaction")

    # Direct: head orientation cues. These are attention events, not movement source.
    head_mask = (np.abs(head_turn) >= HEAD_TURN_RATE_THRESHOLD) | (np.abs(head_body) >= HEAD_BODY_DIFF_THRESHOLD)
    for start, end in duration_mask_runs(head_mask, time_sec, MIN_HEAD_CHECK_SECONDS):
        add_event(events, macro_row, seg, signals, start, end, "head_direction_check", 0.64, "head turn or head/body yaw difference")

    # Audit-only: Avatar/VR gap is not used for movement labels.
    noise_mask = avatar_gap >= avatar_gap_threshold
    for start, end in duration_mask_runs(noise_mask, time_sec, MIN_EVENT_SECONDS):
        add_event(events, macro_row, seg, signals, start, end, "possible_noise_artifact", 0.58, "avatar/VR gap is unusually high")

    detect_hesitation_events(events, macro_row, seg, signals)
    return events


def add_macro_ids_to_frames(df, macro_df):
    frame_df = df.copy()
    frame_df["macro_segment_id"] = -1
    frame_df["macro_label"] = "unassigned"
    for _, row in macro_df.iterrows():
        start = max(0, int(row["start_idx"]))
        end = min(len(frame_df) - 1, int(row["end_idx"]))
        frame_df.loc[start:end, "macro_segment_id"] = int(row["segment_id"])
        frame_df.loc[start:end, "macro_label"] = str(row["macro_label"])
    return frame_df


def assign_frame_labels(df, events):
    frame_df = df.copy()
    frame_df["micro_event_ids"] = ""
    frame_df["active_micro_events"] = ""
    frame_df["primary_micro_label"] = "none"
    frame_df["micro_event_count"] = 0

    for label in EVENT_METADATA:
        frame_df[f"is_{label}"] = False

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
        bool_col = f"is_{label}"
        if bool_col in frame_df.columns:
            frame_df.loc[max(0, start):min(len(frame_df) - 1, end), bool_col] = True

    primary = []
    for labels in active_labels:
        if not labels:
            primary.append("none")
            continue
        primary.append(sorted(labels, key=lambda label: PRIMARY_LABEL_PRIORITY.get(label, 0), reverse=True)[0])

    frame_df["micro_event_ids"] = ["|".join(ids) for ids in active_ids]
    frame_df["active_micro_events"] = ["|".join(labels) for labels in active_labels]
    frame_df["primary_micro_label"] = primary
    frame_df["micro_event_count"] = [len(labels) for labels in active_labels]

    preferred = [
        TIME_COL,
        "elapsed_time_sec",
        "macro_segment_id",
        "macro_label",
        "micro_event_ids",
        "active_micro_events",
        "primary_micro_label",
        "micro_event_count",
    ]
    bool_cols = [f"is_{label}" for label in EVENT_METADATA if f"is_{label}" in frame_df.columns]
    signal_cols = [
        col for col in [
            FRAME_COL,
            "ped_speed_xz_smooth",
            "ped_accel_xz_smooth",
            "car_ped_distance_xz_smooth",
            "distance_closing_smooth",
            "distance_change",
            "car_speed_xz_smooth",
            "car_accel_xz_smooth",
            "ped_stopped",
            "ped_walking",
            "close_interaction",
            "head_yaw",
            "head_turn_rate",
            "head_body_yaw_diff",
            "avatar_vr_gap_xz",
        ]
        if col in frame_df.columns
    ]
    other_cols = [col for col in frame_df.columns if col not in set(preferred + bool_cols + signal_cols)]
    return frame_df[[col for col in preferred if col in frame_df.columns] + bool_cols + signal_cols + other_cols]


def build_summary(events, macro_df):
    columns = [
        "macro_segment_id",
        "macro_label",
        "event_label",
        "count",
        "total_duration_sec",
        "mean_duration_sec",
        "first_start_time_sec",
        "last_end_time_sec",
        "percent_of_macro_segment",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        events
        .groupby(["macro_segment_id", "macro_label", "event_label"], dropna=False)
        .agg(
            count=("micro_event_id", "count"),
            total_duration_sec=("duration_sec", "sum"),
            mean_duration_sec=("duration_sec", "mean"),
            mean_confidence=("confidence", "mean"),
            first_start_time_sec=("start_time_sec", "min"),
            last_end_time_sec=("end_time_sec", "max"),
            distance_min=("distance_min", "min"),
            distance_closing_max=("distance_closing_max", "max"),
            ped_speed_mean=("ped_speed_mean", "mean"),
        )
        .reset_index()
    )
    macro_duration = macro_df.set_index("segment_id")["duration_sec"].to_dict()
    summary["macro_duration_sec"] = summary["macro_segment_id"].map(macro_duration)
    summary["percent_of_macro_segment"] = np.where(
        summary["macro_duration_sec"] > 0,
        100.0 * summary["total_duration_sec"] / summary["macro_duration_sec"],
        np.nan,
    )
    return summary.sort_values(["macro_segment_id", "first_start_time_sec", "event_label"])


def load_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        return plt, Patch
    except ImportError:
        print("Skipping plots: matplotlib is not installed in this Python environment.")
        return None, None


def event_color(label):
    return EVENT_METADATA.get(label, {}).get("color", "#BAB0AC")


def shade_macro_segments(ax, macro_df, label_top=False):
    for i, row in macro_df.iterrows():
        color = "#F5F5F5" if i % 2 == 0 else "#EDEDED"
        ax.axvspan(row["time_start_sec"], row["time_end_sec"], color=color, alpha=0.45, zorder=0)
        ax.axvline(row["time_start_sec"], color="#888888", linewidth=0.8, alpha=0.75, zorder=1)
        if label_top:
            ax.text(
                (row["time_start_sec"] + row["time_end_sec"]) / 2,
                1.02,
                f"macro {int(row['segment_id'])}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
    if len(macro_df):
        ax.axvline(macro_df["time_end_sec"].iloc[-1], color="#888888", linewidth=0.8, alpha=0.75, zorder=1)


def plot_event_tracks(ax, events):
    labels = list(EVENT_METADATA.keys())
    present = [label for label in labels if label in set(events["event_label"])]
    y_lookup = {label: i for i, label in enumerate(present)}
    for _, event in events.iterrows():
        y = y_lookup[event["event_label"]]
        ax.barh(
            y,
            event["duration_sec"],
            left=event["start_time_sec"],
            height=0.62,
            color=event_color(event["event_label"]),
            edgecolor="white",
            linewidth=0.8,
        )
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=8)
    ax.set_ylim(-0.7, len(present) - 0.3 if present else 1)
    ax.grid(True, axis="x", alpha=0.25)
    return present


def plot_diagnostic_panels(frame_df, macro_df, events, output_path):
    plt, Patch = load_matplotlib()
    if plt is None or events.empty:
        return False

    time = frame_df["elapsed_time_sec"].to_numpy(dtype=float)
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(18, 14),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.1, 1.1, 1.1, 1.1, 1.4]},
    )

    panels = [
        ("Pedestrian speed", pick_col(frame_df, PED_SPEED_COLS), "m/s"),
        ("Pedestrian acceleration", pick_col(frame_df, PED_ACCEL_COLS), "m/s^2"),
        ("Car-pedestrian distance", pick_col(frame_df, DISTANCE_COLS), "m"),
        ("Distance closing/change", pick_col(frame_df, CLOSING_COLS) or pick_col(frame_df, DISTANCE_CHANGE_COLS), "m/frame"),
        ("Car speed/acceleration", pick_col(frame_df, CAR_SPEED_COLS) or pick_col(frame_df, CAR_ACCEL_COLS), "value"),
    ]

    for ax, (title, col, ylabel) in zip(axes[:5], panels):
        shade_macro_segments(ax, macro_df)
        if col is not None:
            ax.plot(time, pd.to_numeric(frame_df[col], errors="coerce"), color="#222222", linewidth=1.3, label=col)
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "signal unavailable", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    shade_macro_segments(axes[5], macro_df, label_top=True)
    present = plot_event_tracks(axes[5], events)
    axes[5].set_title("Micro-event timeline", loc="left", fontsize=10)
    axes[5].set_xlabel("Scenario elapsed time, seconds")

    legend_handles = [Patch(facecolor=event_color(label), label=label) for label in present]
    if legend_handles:
        fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=8, frameon=False)

    fig.suptitle("Pedestrian Micro-Segmentation Diagnostic Panels", fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.86, 0.97])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_gantt(events, macro_df, output_path):
    plt, Patch = load_matplotlib()
    if plt is None or events.empty:
        return False

    labels_present = [label for label in EVENT_METADATA if label in set(events["event_label"])]
    fig_height = max(6, 0.55 * len(labels_present) + 3)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    shade_macro_segments(ax, macro_df, label_top=True)
    present = plot_event_tracks(ax, events)
    ax.set_title("Pedestrian Micro-Events by Event Type")
    ax.set_xlabel("Scenario elapsed time, seconds")
    ax.set_ylabel("Event label")

    legend_handles = [Patch(facecolor=event_color(label), label=label) for label in present]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)

    fig.tight_layout(rect=[0, 0, 0.82, 1])
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

    feature_df = clean_columns(pd.read_csv(args.feature_csv))
    macro_df = clean_columns(pd.read_csv(args.macro_csv))
    required_macro_cols = ["segment_id", "macro_label", "start_idx", "end_idx", "time_start_sec", "time_end_sec", "duration_sec"]
    missing_macro = [col for col in required_macro_cols if col not in macro_df.columns]
    if missing_macro:
        raise ValueError(f"Macro segment CSV missing required columns: {missing_macro}")

    pick_col(feature_df, PED_SPEED_COLS, required=True, purpose="VR-derived pedestrian speed column")
    feature_df = feature_df.sort_values(TIME_COL).reset_index(drop=True)
    feature_df["elapsed_time_sec"] = build_elapsed_time_seconds(feature_df)
    feature_df = add_macro_ids_to_frames(feature_df, macro_df)

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
    summary_df = build_summary(events_df, macro_df)

    events_csv = output_dir / "micro_events_PedNYC1_scenario3_v2.csv"
    frame_csv = output_dir / "features_with_micro_labels_PedNYC1_scenario3_v2.csv"
    summary_csv = output_dir / "micro_event_summary_by_macro_PedNYC1_scenario3_v2.csv"
    diagnostic_png = output_dir / "micro_diagnostic_panels_PedNYC1_scenario3_v2.png"
    gantt_png = output_dir / "micro_events_gantt_PedNYC1_scenario3_v2.png"

    events_df.to_csv(events_csv, index=False)
    frame_df.to_csv(frame_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    wrote_diagnostic = False
    wrote_gantt = False
    if not args.no_plot:
        wrote_diagnostic = plot_diagnostic_panels(frame_df, macro_df, events_df, diagnostic_png)
        wrote_gantt = plot_gantt(events_df, macro_df, gantt_png)

    print("\nDone.")
    print(f"Feature CSV: {args.feature_csv}")
    print(f"Macro segment CSV: {args.macro_csv}")
    print(f"Saved micro-events CSV: {events_csv}")
    print(f"Saved frame-level labels CSV: {frame_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    if wrote_diagnostic:
        print(f"Saved diagnostic plot: {diagnostic_png}")
    if wrote_gantt:
        print(f"Saved Gantt plot: {gantt_png}")

    if events_df.empty:
        print("\nNo micro-events detected with the current thresholds.")
    else:
        print("\nDetected micro-events by label:")
        print(events_df["event_label"].value_counts().to_string())
        overlap_frames = int((frame_df["micro_event_count"] > 1).sum())
        print(f"\nFrames with overlapping micro-events: {overlap_frames}")


if __name__ == "__main__":
    main()
