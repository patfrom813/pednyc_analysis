"""
infer_ped_behavior_v3_multilayer.py

BURE/Grokwalks PedNYC pedestrian behavior inference, version 3.

Main idea:
Instead of forcing each frame into ONE simple behavior label, this script creates
separate behavior layers:

    movement_state_v3
    head_state_v3
    hand_state_v3
    hesitation_score_v3
    yield_score_v3
    higher_level_inference_v3
    ped_behavior_phrase_v3

The readable phrase column is meant for FFmpeg overlays. The separate layer
columns are better for later analysis, plotting, or machine learning.

Important pandas fix:
This script does NOT use pd.to_numeric(..., errors="ignore").
It only converts math columns with errors="coerce".
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# ============================================================
# 1. DEFAULT PATHS
# ============================================================

DEFAULT_INPUT_CSV = Path(
    r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis\feature_outputs\features_PedNYC1_scenario3_smoothed_accel_with_head_arms.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis\4frame_view"
)


# ============================================================
# 2. SETTINGS / THRESHOLDS
# ============================================================

# Movement thresholds, in meters/second and meters/second^2.
STOP_SPEED = 0.15
NEAR_STOP_SPEED = 0.30
WALK_SPEED = 0.30

STARTING_ACCEL = 0.20
SPEEDING_ACCEL = 0.25
STEADY_ACCEL_ABS = 0.18
SLOWING_ACCEL = -0.30
STOPPING_ACCEL = -0.20
STEPPING_BACK_SPEED = -0.18

# Head thresholds, in degrees and degrees/second.
HEAD_LOOK_ANGLE_DEG = 20.0
HEAD_TURN_RATE_DEG_S = 35.0
SCANNING_WINDOW_SECONDS = 1.20

# Hand thresholds. These are first-pass values and should be calibrated visually.
HAND_ACTIVITY_SPEED = 0.35
HAND_RAISED_REL_HEIGHT = -0.15

# Optional vehicle interaction threshold. Used only if distance columns exist.
CLOSE_DISTANCE_M = 10.0

# Smoothing / segment cleanup.
SMOOTH_WINDOW_SECONDS = 0.50
MIN_SEGMENT_DURATION = 0.45

# IMPORTANT:
# Unity left/right sign may be reversed depending on coordinate convention.
# If the video overlay shows left/right backward, flip this to False.
LEFT_IS_POSITIVE = True


# ============================================================
# 3. HELPERS
# ============================================================


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def require_any_column(df: pd.DataFrame, options: list[str], meaning: str) -> str:
    """Return the first available column from options, or raise a clear error."""
    for col in options:
        if col in df.columns:
            return col

    raise ValueError(
        f"Missing required column for {meaning}. Need one of:\n"
        + "\n".join(options)
        + "\n\nAvailable columns:\n"
        + "\n".join(df.columns)
    )


def safe_to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convert only math columns to numeric using errors='coerce'.
    This avoids the pandas errors='ignore' problem and avoids damaging label columns.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def rolling_smooth(series: pd.Series, window_rows: int) -> pd.Series:
    """Median then mean smoothing. Good for reducing small spikes."""
    return (
        series
        .rolling(window=window_rows, center=True, min_periods=1)
        .median()
        .rolling(window=window_rows, center=True, min_periods=1)
        .mean()
    )


def angular_diff_deg(a, b):
    """
    Shortest signed difference between angles in degrees.
    Handles wrap-around: 359 to 1 degrees is 2 degrees, not -358.
    """
    return (a - b + 180) % 360 - 180


def direction_from_sign(value: float) -> str:
    """Convert positive/negative angular sign into left/right."""
    if not np.isfinite(value):
        return "unknown"

    if LEFT_IS_POSITIVE:
        return "left" if value > 0 else "right"
    return "right" if value > 0 else "left"


def make_odd_window(seconds: float, median_dt: float, minimum: int = 3) -> int:
    if not np.isfinite(median_dt) or median_dt <= 0:
        return minimum
    rows = max(minimum, int(round(seconds / median_dt)))
    if rows % 2 == 0:
        rows += 1
    return rows


def segment_duration_seconds(times: np.ndarray, start: int, end_exclusive: int, fallback_dt: float) -> float:
    if end_exclusive <= start:
        return 0.0
    if end_exclusive - 1 < len(times):
        return float(times[end_exclusive - 1] - times[start] + fallback_dt)
    return float(times[-1] - times[start] + fallback_dt)


def create_segments(labels, times, label_col_name: str, fallback_dt: float) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=object)
    times = np.asarray(times, dtype=float)

    if len(labels) == 0:
        return pd.DataFrame()

    starts = [0]
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            starts.append(i)

    rows = []
    for seg_num, start_idx in enumerate(starts, start=1):
        end_idx = starts[seg_num] if seg_num < len(starts) else len(labels)
        duration = segment_duration_seconds(times, start_idx, end_idx, fallback_dt)

        rows.append({
            "segment_id": seg_num,
            "start_idx": int(start_idx),
            "end_idx_exclusive": int(end_idx),
            "start_time": float(times[start_idx]),
            "end_time": float(times[end_idx - 1]),
            "duration_seconds": duration,
            "frame_count": int(end_idx - start_idx),
            label_col_name: labels[start_idx],
        })

    return pd.DataFrame(rows)


def smooth_short_segments(labels, times, min_duration, fallback_dt):
    """
    Remove very short one-off label flips by replacing them with the neighboring label.
    This is applied to the component layers, not only the final phrase.
    """
    labels = np.asarray(labels, dtype=object).copy()

    for _ in range(5):
        segments = create_segments(labels, times, "label", fallback_dt)
        changed = False

        for _, seg in segments.iterrows():
            if seg["duration_seconds"] >= min_duration:
                continue

            start = int(seg["start_idx"])
            end = int(seg["end_idx_exclusive"])

            prev_label = labels[start - 1] if start > 0 else None
            next_label = labels[end] if end < len(labels) else None

            if prev_label is not None and next_label is not None:
                replacement = prev_label if prev_label == next_label else prev_label
            elif prev_label is not None:
                replacement = prev_label
            elif next_label is not None:
                replacement = next_label
            else:
                continue

            labels[start:end] = replacement
            changed = True

        if not changed:
            break

    return labels


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalized_0_100(x):
    return np.clip(x, 0, 100).astype(int)


# ============================================================
# 4. FEATURE PREP
# ============================================================


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = clean_column_names(df)

    # Required basics.
    time_col = require_any_column(df, ["ScenarioTime", "GameTime"], "time")
    dt_col = require_any_column(df, ["dt"], "frame delta time")
    speed_col = require_any_column(df, ["ped_speed_xz_smooth", "ped_speed_xz"], "pedestrian speed")

    # Acceleration can be loaded or derived from speed.
    accel_col = first_existing(df, ["ped_accel_xz", "ped_accel_xz_raw_unsmoothed"])

    # Useful optional columns.
    optional_numeric = [
        "ped_x", "ped_z", "ped_yaw",
        "head_yaw", "head_body_yaw_diff", "head_body_yaw_diff_abs",
        "head_turn_rate_deg_s", "head_turn_rate_abs_deg_s", "head_turn_rate_abs_smooth",
        "left_hand_speed_3d", "left_hand_speed_3d_smooth",
        "right_hand_speed_3d", "right_hand_speed_3d_smooth",
        "hand_activity_speed",
        "left_hand_height_relative_to_head", "right_hand_height_relative_to_head",
        "car_ped_distance_xz", "close_interaction",
    ]

    numeric_cols = [time_col, dt_col, speed_col]
    if accel_col is not None:
        numeric_cols.append(accel_col)
    numeric_cols.extend(optional_numeric)
    df = safe_to_numeric(df, numeric_cols)

    df = df.sort_values(time_col).reset_index(drop=True)
    df["ScenarioTime_behavior_v3"] = df[time_col]

    median_dt = float(df[dt_col].median())
    if not np.isfinite(median_dt) or median_dt <= 0:
        median_dt = 1 / 30

    smooth_rows = make_odd_window(SMOOTH_WINDOW_SECONDS, median_dt)
    scan_rows = make_odd_window(SCANNING_WINDOW_SECONDS, median_dt)

    df["ped_speed_behavior_smooth_v3"] = rolling_smooth(df[speed_col].astype(float), smooth_rows)

    if accel_col is not None:
        df["ped_accel_behavior_smooth_v3"] = rolling_smooth(df[accel_col].astype(float), smooth_rows)
    else:
        speed = df["ped_speed_behavior_smooth_v3"].to_numpy(dtype=float)
        dt = df[dt_col].to_numpy(dtype=float)
        safe_dt = np.where(dt > 1e-6, dt, np.nan)
        accel = np.full(len(df), np.nan)
        accel[1:] = np.diff(speed) / safe_dt[1:]
        accel[0] = accel[1] if len(accel) > 1 else np.nan
        df["ped_accel_behavior_smooth_v3"] = rolling_smooth(pd.Series(accel), smooth_rows)

    # Signed progress/back-step estimate. This is only an approximation.
    # It estimates the main walking direction from start to end, then checks
    # if the pedestrian briefly moves opposite that direction.
    if "ped_x" in df.columns and "ped_z" in df.columns:
        x = df["ped_x"].to_numpy(dtype=float)
        z = df["ped_z"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(z)
        signed_speed = np.full(len(df), np.nan)

        if valid.sum() >= 3:
            first = np.where(valid)[0][0]
            last = np.where(valid)[0][-1]
            overall_vec = np.array([x[last] - x[first], z[last] - z[first]], dtype=float)
            norm = np.linalg.norm(overall_vec)
            if norm > 1e-6:
                direction = overall_vec / norm
                dx = np.full(len(df), np.nan)
                dz = np.full(len(df), np.nan)
                dx[1:] = np.diff(x)
                dz[1:] = np.diff(z)
                dt = df[dt_col].to_numpy(dtype=float)
                safe_dt = np.where(dt > 1e-6, dt, np.nan)
                signed_step = dx * direction[0] + dz * direction[1]
                signed_speed = signed_step / safe_dt

        df["ped_signed_progress_speed_v3"] = rolling_smooth(pd.Series(signed_speed), smooth_rows)
    else:
        df["ped_signed_progress_speed_v3"] = np.nan

    # Head-body angle.
    if "head_body_yaw_diff" not in df.columns:
        if "head_yaw" in df.columns and "ped_yaw" in df.columns:
            df["head_body_yaw_diff"] = angular_diff_deg(
                df["head_yaw"].astype(float),
                df["ped_yaw"].astype(float),
            )
        else:
            df["head_body_yaw_diff"] = np.nan

    df["head_body_yaw_diff_abs"] = np.abs(df["head_body_yaw_diff"].astype(float))

    # Head turn rate.
    if "head_turn_rate_deg_s" not in df.columns:
        if "head_yaw" in df.columns:
            head_yaw = df["head_yaw"].to_numpy(dtype=float)
            dt = df[dt_col].to_numpy(dtype=float)
            safe_dt = np.where(dt > 1e-6, dt, np.nan)
            rate = np.full(len(df), np.nan)
            rate[1:] = angular_diff_deg(head_yaw[1:], head_yaw[:-1]) / safe_dt[1:]
            rate[0] = rate[1] if len(rate) > 1 else np.nan
            df["head_turn_rate_deg_s"] = rate
        else:
            df["head_turn_rate_deg_s"] = np.nan

    df["head_turn_rate_smooth_signed_v3"] = rolling_smooth(
        df["head_turn_rate_deg_s"].astype(float), smooth_rows
    )
    df["head_turn_rate_abs_smooth_v3"] = np.abs(df["head_turn_rate_smooth_signed_v3"])

    # Hand activity.
    left_speed_col = first_existing(df, ["left_hand_speed_3d_smooth", "left_hand_speed_3d"])
    right_speed_col = first_existing(df, ["right_hand_speed_3d_smooth", "right_hand_speed_3d"])

    if left_speed_col is not None:
        df["left_hand_speed_behavior_smooth_v3"] = rolling_smooth(df[left_speed_col].astype(float), smooth_rows)
    else:
        df["left_hand_speed_behavior_smooth_v3"] = np.nan

    if right_speed_col is not None:
        df["right_hand_speed_behavior_smooth_v3"] = rolling_smooth(df[right_speed_col].astype(float), smooth_rows)
    else:
        df["right_hand_speed_behavior_smooth_v3"] = np.nan

    if "hand_activity_speed" in df.columns:
        df["hand_activity_speed_behavior_smooth_v3"] = rolling_smooth(
            df["hand_activity_speed"].astype(float), smooth_rows
        )
    elif left_speed_col is not None or right_speed_col is not None:
        df["hand_activity_speed_behavior_smooth_v3"] = np.nanmean(
            np.vstack([
                df["left_hand_speed_behavior_smooth_v3"].to_numpy(dtype=float),
                df["right_hand_speed_behavior_smooth_v3"].to_numpy(dtype=float),
            ]),
            axis=0,
        )
    else:
        df["hand_activity_speed_behavior_smooth_v3"] = np.nan

    meta = {
        "time_col": time_col,
        "dt_col": dt_col,
        "speed_col": speed_col,
        "accel_col": accel_col,
        "median_dt": median_dt,
        "smooth_rows": smooth_rows,
        "scan_rows": scan_rows,
    }
    return df, meta


# ============================================================
# 5. MULTI-LAYER INFERENCE
# ============================================================


def infer_movement_state(df: pd.DataFrame, times: np.ndarray, fallback_dt: float) -> np.ndarray:
    speed = df["ped_speed_behavior_smooth_v3"].to_numpy(dtype=float)
    accel = df["ped_accel_behavior_smooth_v3"].to_numpy(dtype=float)
    signed_progress = df["ped_signed_progress_speed_v3"].to_numpy(dtype=float)

    movement = np.full(len(df), "unknown_movement", dtype=object)

    movement[speed < STOP_SPEED] = "stopped"
    movement[(speed >= STOP_SPEED) & (speed < NEAR_STOP_SPEED)] = "near_stopped"
    movement[speed >= WALK_SPEED] = "walking"

    movement[(speed >= WALK_SPEED) & (np.abs(accel) <= STEADY_ACCEL_ABS)] = "steady_walking"
    movement[(speed >= WALK_SPEED) & (accel >= SPEEDING_ACCEL)] = "speeding_up"
    movement[(speed >= WALK_SPEED) & (accel <= SLOWING_ACCEL)] = "slowing_down"

    movement[(speed < WALK_SPEED) & (accel >= STARTING_ACCEL)] = "starting_to_walk"
    movement[(speed < WALK_SPEED) & (accel <= STOPPING_ACCEL)] = "slowing_to_stop"

    # Only call it stepping_back when the person is moving opposite their overall path.
    movement[(signed_progress <= STEPPING_BACK_SPEED) & (speed >= STOP_SPEED)] = "stepping_back"

    movement = smooth_short_segments(movement, times, MIN_SEGMENT_DURATION, fallback_dt)
    return movement


def infer_head_state(df: pd.DataFrame, times: np.ndarray, fallback_dt: float, scan_rows: int) -> np.ndarray:
    head_diff = df["head_body_yaw_diff"].to_numpy(dtype=float)
    head_diff_abs = df["head_body_yaw_diff_abs"].to_numpy(dtype=float)
    head_rate = df["head_turn_rate_smooth_signed_v3"].to_numpy(dtype=float)
    head_rate_abs = df["head_turn_rate_abs_smooth_v3"].to_numpy(dtype=float)

    head = np.full(len(df), "head_forward", dtype=object)

    no_head_data = np.all(~np.isfinite(head_diff_abs)) and np.all(~np.isfinite(head_rate_abs))
    if no_head_data:
        return np.full(len(df), "head_unavailable", dtype=object)

    # Base looking / turning labels.
    for i in range(len(df)):
        is_looking_side = np.isfinite(head_diff_abs[i]) and head_diff_abs[i] >= HEAD_LOOK_ANGLE_DEG
        is_turning = np.isfinite(head_rate_abs[i]) and head_rate_abs[i] >= HEAD_TURN_RATE_DEG_S

        if is_turning:
            side = direction_from_sign(head_rate[i])
            head[i] = f"turning_head_{side}"
        elif is_looking_side:
            side = direction_from_sign(head_diff[i])
            head[i] = f"looking_{side}"

    # Scanning = left and right looks/turns happen within a short time window.
    side_series = np.full(len(df), "center", dtype=object)
    for i in range(len(df)):
        if np.isfinite(head_diff_abs[i]) and head_diff_abs[i] >= HEAD_LOOK_ANGLE_DEG:
            side_series[i] = direction_from_sign(head_diff[i])

    half = max(1, scan_rows // 2)
    scanning = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        lo = max(0, i - half)
        hi = min(len(df), i + half + 1)
        window = side_series[lo:hi]
        has_left = np.any(window == "left")
        has_right = np.any(window == "right")
        enough_motion = np.nanmax(head_rate_abs[lo:hi]) >= HEAD_TURN_RATE_DEG_S if np.any(np.isfinite(head_rate_abs[lo:hi])) else False
        if has_left and has_right and enough_motion:
            scanning[i] = True

    head[scanning] = "scanning"
    head = smooth_short_segments(head, times, MIN_SEGMENT_DURATION, fallback_dt)
    return head


def infer_hand_state(df: pd.DataFrame, times: np.ndarray, fallback_dt: float) -> np.ndarray:
    left_speed = df["left_hand_speed_behavior_smooth_v3"].to_numpy(dtype=float)
    right_speed = df["right_hand_speed_behavior_smooth_v3"].to_numpy(dtype=float)
    hand_activity = df["hand_activity_speed_behavior_smooth_v3"].to_numpy(dtype=float)

    left_height = df["left_hand_height_relative_to_head"].to_numpy(dtype=float) if "left_hand_height_relative_to_head" in df.columns else np.full(len(df), np.nan)
    right_height = df["right_hand_height_relative_to_head"].to_numpy(dtype=float) if "right_hand_height_relative_to_head" in df.columns else np.full(len(df), np.nan)

    no_hand_data = (
        np.all(~np.isfinite(left_speed))
        and np.all(~np.isfinite(right_speed))
        and np.all(~np.isfinite(hand_activity))
        and np.all(~np.isfinite(left_height))
        and np.all(~np.isfinite(right_height))
    )
    if no_hand_data:
        return np.full(len(df), "hands_unavailable", dtype=object)

    hand = np.full(len(df), "hands_quiet", dtype=object)

    for i in range(len(df)):
        left_active = np.isfinite(left_speed[i]) and left_speed[i] >= HAND_ACTIVITY_SPEED
        right_active = np.isfinite(right_speed[i]) and right_speed[i] >= HAND_ACTIVITY_SPEED
        any_active = (
            left_active
            or right_active
            or (np.isfinite(hand_activity[i]) and hand_activity[i] >= HAND_ACTIVITY_SPEED)
        )

        left_raised = np.isfinite(left_height[i]) and left_height[i] >= HAND_RAISED_REL_HEIGHT
        right_raised = np.isfinite(right_height[i]) and right_height[i] >= HAND_RAISED_REL_HEIGHT
        any_raised = left_raised or right_raised

        if any_raised and any_active:
            hand[i] = "possible_signal"
        elif any_raised:
            hand[i] = "hand_raised"
        elif left_active and right_active:
            hand[i] = "both_hands_motion"
        elif left_active:
            hand[i] = "left_hand_motion"
        elif right_active:
            hand[i] = "right_hand_motion"
        elif any_active:
            hand[i] = "both_hands_motion"

    hand = smooth_short_segments(hand, times, MIN_SEGMENT_DURATION, fallback_dt)
    return hand


def infer_scores_and_high_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    movement = df["movement_state_v3"].astype(str).to_numpy()
    head = df["head_state_v3"].astype(str).to_numpy()
    hand = df["hand_state_v3"].astype(str).to_numpy()
    speed = df["ped_speed_behavior_smooth_v3"].to_numpy(dtype=float)
    accel = df["ped_accel_behavior_smooth_v3"].to_numpy(dtype=float)

    head_active = ~np.isin(head, ["head_forward", "head_unavailable"])
    hand_active = ~np.isin(hand, ["hands_quiet", "hands_unavailable"])

    near_stop_like = np.isin(movement, ["stopped", "near_stopped", "slowing_to_stop", "stepping_back"])
    slowing_like = np.isin(movement, ["slowing_down", "slowing_to_stop", "stepping_back"])
    walking_like = np.isin(movement, ["walking", "steady_walking", "speeding_up", "starting_to_walk"])
    decelerating = np.isfinite(accel) & (accel <= STOPPING_ACCEL)

    # Optional close interaction feature.
    close_interaction = np.zeros(len(df), dtype=bool)
    if "close_interaction" in df.columns:
        close_interaction = df["close_interaction"].fillna(0).astype(float).to_numpy() > 0
    elif "car_ped_distance_xz" in df.columns:
        dist = df["car_ped_distance_xz"].to_numpy(dtype=float)
        close_interaction = np.isfinite(dist) & (dist <= CLOSE_DISTANCE_M)

    hesitation_raw = (
        near_stop_like.astype(int) * 35
        + slowing_like.astype(int) * 20
        + decelerating.astype(int) * 20
        + head_active.astype(int) * 20
        + hand_active.astype(int) * 5
    )
    df["hesitation_score_v3"] = normalized_0_100(hesitation_raw)

    yield_raw = (
        near_stop_like.astype(int) * 35
        + decelerating.astype(int) * 15
        + head_active.astype(int) * 25
        + close_interaction.astype(int) * 15
        + np.isin(hand, ["hand_raised", "possible_signal"]).astype(int) * 10
    )
    df["yield_score_v3"] = normalized_0_100(yield_raw)

    higher = np.full(len(df), "uncertain_crossing", dtype=object)

    for i in range(len(df)):
        if df.loc[i, "yield_score_v3"] >= 70 and near_stop_like[i]:
            higher[i] = "yield_candidate"
        elif df.loc[i, "hesitation_score_v3"] >= 60:
            higher[i] = "hesitating"
        elif near_stop_like[i] and not walking_like[i]:
            higher[i] = "waiting"
        elif head_active[i]:
            higher[i] = "checking_vehicle"
        elif walking_like[i] and not decelerating[i] and not head_active[i]:
            higher[i] = "committed_crossing"
        else:
            higher[i] = "uncertain_crossing"

    df["higher_level_inference_v3"] = higher
    df["checking_vehicle_v3"] = (higher == "checking_vehicle").astype(int)
    df["hesitating_v3"] = (higher == "hesitating").astype(int)
    df["yield_candidate_v3"] = (higher == "yield_candidate").astype(int)
    df["committed_crossing_v3"] = (higher == "committed_crossing").astype(int)

    return df


def build_behavior_phrase(row: pd.Series) -> str:
    """Readable multi-layer phrase for FFmpeg overlay."""
    parts = [str(row["movement_state_v3"])]

    head = str(row["head_state_v3"])
    hand = str(row["hand_state_v3"])
    inference = str(row["higher_level_inference_v3"])

    if head not in ["head_forward", "head_unavailable"]:
        parts.append(head)

    if hand not in ["hands_quiet", "hands_unavailable"]:
        parts.append(hand)

    # Keep overlay useful but not too noisy.
    if inference == "hesitating":
        parts.append("possible_hesitation")
    elif inference in ["yield_candidate", "checking_vehicle", "waiting", "committed_crossing"]:
        parts.append(inference)

    return " + ".join(parts)


# ============================================================
# 6. SEGMENT OUTPUT
# ============================================================


def build_segment_summary(df: pd.DataFrame, times: np.ndarray, fallback_dt: float) -> pd.DataFrame:
    phrase_segments = create_segments(
        df["ped_behavior_phrase_v3"].to_numpy(dtype=object),
        times,
        "ped_behavior_phrase_v3",
        fallback_dt,
    )

    df["ped_behavior_v3_segment_id"] = -1
    rows = []

    for _, seg in phrase_segments.iterrows():
        start = int(seg["start_idx"])
        end = int(seg["end_idx_exclusive"])
        chunk = df.iloc[start:end]
        segment_id = int(seg["segment_id"])
        df.loc[start:end - 1, "ped_behavior_v3_segment_id"] = segment_id

        def mode_or_unknown(col: str) -> str:
            if col not in chunk.columns or chunk[col].dropna().empty:
                return "unknown"
            return str(chunk[col].mode(dropna=True).iloc[0])

        rows.append({
            "segment_id": segment_id,
            "start_idx": start,
            "end_idx_exclusive": end,
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "duration_seconds": seg["duration_seconds"],
            "frame_count": int(end - start),
            "movement_state_v3": mode_or_unknown("movement_state_v3"),
            "head_state_v3": mode_or_unknown("head_state_v3"),
            "hand_state_v3": mode_or_unknown("hand_state_v3"),
            "higher_level_inference_v3": mode_or_unknown("higher_level_inference_v3"),
            "ped_behavior_phrase_v3": seg["ped_behavior_phrase_v3"],
            "mean_speed": chunk["ped_speed_behavior_smooth_v3"].mean(),
            "mean_accel": chunk["ped_accel_behavior_smooth_v3"].mean(),
            "mean_head_body_yaw_diff_abs": chunk["head_body_yaw_diff_abs"].mean(),
            "max_head_turn_rate_abs": chunk["head_turn_rate_abs_smooth_v3"].max(),
            "mean_hand_activity_speed": chunk["hand_activity_speed_behavior_smooth_v3"].mean(),
            "mean_hesitation_score_v3": chunk["hesitation_score_v3"].mean(),
            "mean_yield_score_v3": chunk["yield_score_v3"].mean(),
            "hesitating_frames": int(chunk["hesitating_v3"].sum()),
            "yield_candidate_frames": int(chunk["yield_candidate_v3"].sum()),
            "committed_crossing_frames": int(chunk["committed_crossing_v3"].sum()),
        })

    return df, pd.DataFrame(rows)


# ============================================================
# 7. MAIN
# ============================================================


def run(input_csv: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    output_labeled_csv = output_dir / f"{input_csv.stem}_ped_behavior_v3_labeled.csv"
    output_segments_csv = output_dir / f"{input_csv.stem}_ped_behavior_v3_segments.csv"

    print("Loading:")
    print(input_csv)

    df = pd.read_csv(input_csv)
    df, meta = prepare_features(df)

    times = df[meta["time_col"]].to_numpy(dtype=float)
    fallback_dt = float(meta["median_dt"])

    print(f"Rows: {len(df)}")
    print(f"Median dt: {fallback_dt:.4f}")
    print(f"Smoothing window: {meta['smooth_rows']} rows")
    print(f"Scanning window: {meta['scan_rows']} rows")

    df["movement_state_v3"] = infer_movement_state(df, times, fallback_dt)
    df["head_state_v3"] = infer_head_state(df, times, fallback_dt, meta["scan_rows"])
    df["hand_state_v3"] = infer_hand_state(df, times, fallback_dt)

    df = infer_scores_and_high_level(df)
    df["ped_behavior_phrase_v3"] = df.apply(build_behavior_phrase, axis=1)

    # Stabilize the overlay phrase too. The separate component columns remain
    # available for analysis, while the phrase is cleaned for human viewing.
    df["ped_behavior_phrase_v3"] = smooth_short_segments(
        df["ped_behavior_phrase_v3"].to_numpy(dtype=object),
        times,
        MIN_SEGMENT_DURATION,
        fallback_dt,
    )

    df, segments_out = build_segment_summary(df, times, fallback_dt)

    df.to_csv(output_labeled_csv, index=False)
    segments_out.to_csv(output_segments_csv, index=False)

    print("\nSaved frame-level labeled CSV:")
    print(output_labeled_csv)

    print("\nSaved segment-level CSV:")
    print(output_segments_csv)

    print("\nBehavior segments preview:")
    preview_cols = [
        "segment_id", "start_time", "end_time", "duration_seconds",
        "movement_state_v3", "head_state_v3", "hand_state_v3",
        "higher_level_inference_v3", "ped_behavior_phrase_v3",
        "mean_speed", "mean_accel", "mean_hesitation_score_v3", "mean_yield_score_v3",
    ]
    if len(segments_out) > 0:
        print(segments_out[preview_cols].to_string(index=False))
    else:
        print("No segments created.")

    print("\nDone.")
    return output_labeled_csv, output_segments_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer multi-layer pedestrian behavior labels for PedNYC/Grokwalks CSV files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to the feature CSV. Defaults to the PedNYC1 scenario 3 head/arms feature CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where v3 labeled and segment CSVs will be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output_dir)
