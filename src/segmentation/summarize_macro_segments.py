from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(r"C:\Users\patl5\OneDrive\Desktop\BURE\pednyc_analysis")

FEATURE_CSV = ROOT / "data" / "processed" / "features_PedNYC1_scenario3_metrics_v1.csv"

MACRO_SEGMENTS_CSV = (
    ROOT
    / "outputs"
    / "graphs"
    / "macro_segmentation"
    / "macro_segments_PedNYC1_scenario3_v2.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "graphs" / "macro_segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = OUTPUT_DIR / "macro_segments_multisignal_summary.csv"
FRAME_LEVEL_CSV = OUTPUT_DIR / "features_with_macro_segments.csv"


# ============================================================
# Main column names
# ============================================================

TIME_COL = "ScenarioTime"
DT_COL = "dt"
FRAME_COL = "Frame Number"

PED_SPEED_COLS = ["ped_speed_xz_smooth", "ped_speed_xz"]
PED_ACCEL_COLS = ["ped_accel_xz_smooth", "ped_accel_xz"]

CAR_SPEED_COLS = ["car_speed_xz_smooth", "car_speed_xz"]
CAR_ACCEL_COLS = ["car_accel_xz_smooth", "car_accel_xz"]

DISTANCE_COLS = ["car_ped_distance_xz_smooth", "car_ped_distance_xz"]
CLOSING_COLS = ["distance_closing_smooth", "distance_closing"]

HEAD_TURN_RATE_COLS = ["head_turn_rate"]
HEAD_BODY_DIFF_COLS = ["head_body_yaw_diff"]

PED_X_COLS = ["ped_x"]
PED_Z_COLS = ["ped_z"]


# ============================================================
# Tunable inference thresholds
# ============================================================

STOP_SPEED = 0.15
WALK_SPEED = 0.30

CAR_CLOSE_METERS = 10.0
CLOSING_FRACTION_MIN = 0.25

CAR_BRAKE_ACCEL = -0.25
CAR_BRAKING_FRACTION_MIN = 0.25

HEAD_TURN_RATE_THRESHOLD = 60.0       # deg/sec
HEAD_BODY_DIFF_THRESHOLD = 20.0       # degrees
HEAD_TURN_FRACTION_MIN = 0.15

PED_POSITIVE_SLOPE = 0.05
PED_NEGATIVE_SLOPE = -0.05

MIN_CLEAN_DISPLACEMENT = 0.50
CLEAN_PROGRESS_RATIO = 0.75

MIN_HESITATION_SIGN_CHANGES = 2
SPEED_UNSTABLE_STD = 0.15


# ============================================================
# Helper functions
# ============================================================

def pick_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def to_numeric_array(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.any(np.isfinite(x)) else np.nan


def safe_max(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmax(x)) if np.any(np.isfinite(x)) else np.nan


def safe_min(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmin(x)) if np.any(np.isfinite(x)) else np.nan


def safe_std(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanstd(x)) if np.any(np.isfinite(x)) else np.nan


def safe_first(x):
    x = np.asarray(x, dtype=float)
    valid = x[np.isfinite(x)]
    return float(valid[0]) if len(valid) else np.nan


def safe_last(x):
    x = np.asarray(x, dtype=float)
    valid = x[np.isfinite(x)]
    return float(valid[-1]) if len(valid) else np.nan


def linear_slope(time_sec, values):
    time_sec = np.asarray(time_sec, dtype=float)
    values = np.asarray(values, dtype=float)

    mask = np.isfinite(time_sec) & np.isfinite(values)

    if mask.sum() < 2:
        return np.nan

    x = time_sec[mask] - time_sec[mask][0]
    y = values[mask]

    if np.nanmax(x) - np.nanmin(x) <= 0:
        return np.nan

    return float(np.polyfit(x, y, 1)[0])


def fraction_where(values, condition_func):
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(condition_func(values[mask])))


def count_sign_changes(values, eps=0.03):
    """
    Counts meaningful sign changes while ignoring tiny values near zero.
    Useful for detecting unstable accel/decel wiggles.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 3:
        return 0

    signs = np.zeros(len(values), dtype=int)
    signs[values > eps] = 1
    signs[values < -eps] = -1

    signs = signs[signs != 0]

    if len(signs) < 2:
        return 0

    return int(np.sum(signs[1:] != signs[:-1]))


def build_elapsed_time_seconds(df):
    """
    Rebuild elapsed time from dt.
    This should match your macro segmentation script.
    """
    if DT_COL in df.columns:
        dt = pd.to_numeric(df[DT_COL], errors="coerce").to_numpy(dtype=float)

        good_dt = dt[np.isfinite(dt) & (dt > 0) & (dt < 1.0)]

        if len(good_dt) > 0:
            median_dt = float(np.nanmedian(good_dt))
        else:
            median_dt = 1 / 30

        dt_clean = dt.copy()
        bad = ~np.isfinite(dt_clean) | (dt_clean <= 0) | (dt_clean > 1.0)
        dt_clean[bad] = median_dt

        time_sec = np.zeros(len(df), dtype=float)
        time_sec[1:] = np.cumsum(dt_clean[1:])

        return time_sec

    raw_time = pd.to_numeric(df[TIME_COL], errors="coerce").to_numpy(dtype=float)
    return raw_time - raw_time[0]


def compute_gradient_from_time(values, time_sec):
    values = np.asarray(values, dtype=float)
    time_sec = np.asarray(time_sec, dtype=float)

    if len(values) < 3:
        return np.full(len(values), np.nan)

    good = np.isfinite(values) & np.isfinite(time_sec)

    if good.sum() < 3:
        return np.full(len(values), np.nan)

    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    return np.gradient(filled, time_sec)


def movement_progress(seg):
    ped_x_col = pick_col(seg, PED_X_COLS)
    ped_z_col = pick_col(seg, PED_Z_COLS)

    if ped_x_col is None or ped_z_col is None:
        return {
            "ped_displacement_xz": np.nan,
            "ped_path_length_xz": np.nan,
            "ped_progress_ratio": np.nan,
        }

    x = to_numeric_array(seg[ped_x_col])
    z = to_numeric_array(seg[ped_z_col])

    mask = np.isfinite(x) & np.isfinite(z)

    if mask.sum() < 2:
        return {
            "ped_displacement_xz": np.nan,
            "ped_path_length_xz": np.nan,
            "ped_progress_ratio": np.nan,
        }

    x = x[mask]
    z = z[mask]

    displacement = float(np.sqrt((x[-1] - x[0]) ** 2 + (z[-1] - z[0]) ** 2))

    dx = np.diff(x)
    dz = np.diff(z)
    path_length = float(np.nansum(np.sqrt(dx ** 2 + dz ** 2)))

    if path_length > 1e-9:
        progress_ratio = float(displacement / path_length)
    else:
        progress_ratio = np.nan

    return {
        "ped_displacement_xz": displacement,
        "ped_path_length_xz": path_length,
        "ped_progress_ratio": progress_ratio,
    }


# ============================================================
# Segment summary
# ============================================================

def summarize_one_segment(df, segment_row):
    start_idx = int(segment_row["start_idx"])
    end_idx = int(segment_row["end_idx"])

    start_idx = max(0, start_idx)
    end_idx = min(len(df) - 1, end_idx)

    seg = df.iloc[start_idx:end_idx + 1].copy()

    seg_time = to_numeric_array(seg["elapsed_time"])

    ped_speed_col = pick_col(seg, PED_SPEED_COLS)
    ped_accel_col = pick_col(seg, PED_ACCEL_COLS)

    car_speed_col = pick_col(seg, CAR_SPEED_COLS)
    car_accel_col = pick_col(seg, CAR_ACCEL_COLS)

    distance_col = pick_col(seg, DISTANCE_COLS)
    closing_col = pick_col(seg, CLOSING_COLS)

    head_turn_col = pick_col(seg, HEAD_TURN_RATE_COLS)
    head_body_col = pick_col(seg, HEAD_BODY_DIFF_COLS)

    # ------------------------------------------------------------
    # Pedestrian speed
    # ------------------------------------------------------------

    if ped_speed_col is not None:
        ped_speed = to_numeric_array(seg[ped_speed_col])
    else:
        ped_speed = np.full(len(seg), np.nan)

    ped_speed_start = safe_first(ped_speed)
    ped_speed_end = safe_last(ped_speed)
    ped_speed_delta = ped_speed_end - ped_speed_start

    ped_speed_slope = linear_slope(seg_time, ped_speed)

    if ped_accel_col is not None:
        ped_accel = to_numeric_array(seg[ped_accel_col])
    else:
        ped_accel = compute_gradient_from_time(ped_speed, seg_time)

    ped_accel_sign_changes = count_sign_changes(ped_accel)

    # ------------------------------------------------------------
    # Movement progress
    # ------------------------------------------------------------

    progress_stats = movement_progress(seg)

    # ------------------------------------------------------------
    # Distance and closing
    # ------------------------------------------------------------

    if distance_col is not None:
        distance = to_numeric_array(seg[distance_col])
    else:
        distance = np.full(len(seg), np.nan)

    distance_start = safe_first(distance)
    distance_end = safe_last(distance)
    distance_delta = distance_end - distance_start

    if closing_col is not None:
        closing = to_numeric_array(seg[closing_col])
    else:
        # Fallback: if distance is decreasing, interaction is closing.
        closing = -compute_gradient_from_time(distance, seg_time)

    # ------------------------------------------------------------
    # Car movement
    # ------------------------------------------------------------

    if car_speed_col is not None:
        car_speed = to_numeric_array(seg[car_speed_col])
    else:
        car_speed = np.full(len(seg), np.nan)

    car_speed_start = safe_first(car_speed)
    car_speed_end = safe_last(car_speed)
    car_speed_delta = car_speed_end - car_speed_start
    car_speed_slope = linear_slope(seg_time, car_speed)

    if car_accel_col is not None:
        car_accel = to_numeric_array(seg[car_accel_col])
    else:
        car_accel = compute_gradient_from_time(car_speed, seg_time)

    # ------------------------------------------------------------
    # Head features
    # ------------------------------------------------------------

    if head_turn_col is not None:
        head_turn_rate = np.abs(to_numeric_array(seg[head_turn_col]))
    else:
        head_turn_rate = np.full(len(seg), np.nan)

    if head_body_col is not None:
        head_body_diff = np.abs(to_numeric_array(seg[head_body_col]))
    else:
        head_body_diff = np.full(len(seg), np.nan)

    # ------------------------------------------------------------
    # Basic row information
    # ------------------------------------------------------------

    out = {
        "segment_id": int(segment_row["segment_id"]),
        "macro_label": segment_row["macro_label"],
        "start_idx": start_idx,
        "end_idx": end_idx,
        "n_frames": int(len(seg)),
        "time_start_sec": float(seg_time[0]) if len(seg_time) else np.nan,
        "time_end_sec": float(seg_time[-1]) if len(seg_time) else np.nan,
        "duration_sec": float(seg_time[-1] - seg_time[0]) if len(seg_time) >= 2 else np.nan,
    }

    if FRAME_COL in seg.columns:
        out["start_frame"] = seg[FRAME_COL].iloc[0]
        out["end_frame"] = seg[FRAME_COL].iloc[-1]
    else:
        out["start_frame"] = np.nan
        out["end_frame"] = np.nan

    if TIME_COL in seg.columns:
        out["ScenarioTime_start_raw"] = seg[TIME_COL].iloc[0]
        out["ScenarioTime_end_raw"] = seg[TIME_COL].iloc[-1]

    # ------------------------------------------------------------
    # Add pedestrian stats
    # ------------------------------------------------------------

    out.update({
        "ped_speed_start": ped_speed_start,
        "ped_speed_end": ped_speed_end,
        "ped_speed_mean": safe_mean(ped_speed),
        "ped_speed_median": float(np.nanmedian(ped_speed)) if np.any(np.isfinite(ped_speed)) else np.nan,
        "ped_speed_max": safe_max(ped_speed),
        "ped_speed_min": safe_min(ped_speed),
        "ped_speed_std": safe_std(ped_speed),
        "ped_speed_range": safe_max(ped_speed) - safe_min(ped_speed),
        "ped_speed_delta": ped_speed_delta,
        "ped_speed_slope": ped_speed_slope,
        "ped_accel_mean": safe_mean(ped_accel),
        "ped_accel_max": safe_max(ped_accel),
        "ped_accel_min": safe_min(ped_accel),
        "ped_accel_sign_changes": ped_accel_sign_changes,
        "fraction_near_stopped": fraction_where(ped_speed, lambda x: x < STOP_SPEED),
        "fraction_walking": fraction_where(ped_speed, lambda x: x >= WALK_SPEED),
    })

    out.update(progress_stats)

    # ------------------------------------------------------------
    # Add distance stats
    # ------------------------------------------------------------

    out.update({
        "distance_start": distance_start,
        "distance_end": distance_end,
        "distance_delta": distance_delta,
        "min_car_ped_distance": safe_min(distance),
        "mean_car_ped_distance": safe_mean(distance),
        "mean_distance_closing": safe_mean(closing),
        "max_distance_closing": safe_max(closing),
        "fraction_time_closing": fraction_where(closing, lambda x: x > 0),
    })

    # ------------------------------------------------------------
    # Add car stats
    # ------------------------------------------------------------

    out.update({
        "car_speed_start": car_speed_start,
        "car_speed_end": car_speed_end,
        "car_speed_mean": safe_mean(car_speed),
        "car_speed_max": safe_max(car_speed),
        "car_speed_min": safe_min(car_speed),
        "car_speed_delta": car_speed_delta,
        "car_speed_slope": car_speed_slope,
        "car_accel_mean": safe_mean(car_accel),
        "car_accel_min": safe_min(car_accel),
        "fraction_car_braking": fraction_where(car_accel, lambda x: x < CAR_BRAKE_ACCEL),
    })

    # ------------------------------------------------------------
    # Add head stats
    # ------------------------------------------------------------

    out.update({
        "max_abs_head_turn_rate": safe_max(head_turn_rate),
        "mean_abs_head_turn_rate": safe_mean(head_turn_rate),
        "fraction_high_head_turn_rate": fraction_where(
            head_turn_rate,
            lambda x: x >= HEAD_TURN_RATE_THRESHOLD
        ),
        "max_abs_head_body_yaw_diff": safe_max(head_body_diff),
        "mean_abs_head_body_yaw_diff": safe_mean(head_body_diff),
        "fraction_head_body_yaw_diff_high": fraction_where(
            head_body_diff,
            lambda x: x >= HEAD_BODY_DIFF_THRESHOLD
        ),
    })

    return out


# ============================================================
# Evidence flags and interpretation
# ============================================================

def add_evidence_flags(summary_df):
    summary_df = summary_df.copy()

    summary_df["car_close"] = summary_df["min_car_ped_distance"] <= CAR_CLOSE_METERS

    summary_df["car_closing"] = (
        summary_df["fraction_time_closing"] >= CLOSING_FRACTION_MIN
    )

    summary_df["car_braking"] = (
        summary_df["fraction_car_braking"] >= CAR_BRAKING_FRACTION_MIN
    )

    summary_df["ped_speed_unstable"] = (
        (summary_df["ped_accel_sign_changes"] >= MIN_HESITATION_SIGN_CHANGES)
        & (summary_df["ped_speed_std"] >= SPEED_UNSTABLE_STD)
    )

    summary_df["clean_progress"] = (
        (summary_df["ped_progress_ratio"] >= CLEAN_PROGRESS_RATIO)
        & (summary_df["ped_displacement_xz"] >= MIN_CLEAN_DISPLACEMENT)
    )

    summary_df["head_checking"] = (
        (summary_df["max_abs_head_turn_rate"] >= HEAD_TURN_RATE_THRESHOLD)
        | (summary_df["max_abs_head_body_yaw_diff"] >= HEAD_BODY_DIFF_THRESHOLD)
        | (summary_df["fraction_high_head_turn_rate"] >= HEAD_TURN_FRACTION_MIN)
        | (summary_df["fraction_head_body_yaw_diff_high"] >= HEAD_TURN_FRACTION_MIN)
    )

    summary_df["ped_accelerating"] = summary_df["ped_speed_slope"] >= PED_POSITIVE_SLOPE
    summary_df["ped_decelerating"] = summary_df["ped_speed_slope"] <= PED_NEGATIVE_SLOPE

    summary_df["low_motion"] = (
        (summary_df["ped_speed_mean"] < STOP_SPEED)
        | (summary_df["fraction_near_stopped"] >= 0.60)
    )

    summary_df["evidence_flags"] = summary_df.apply(make_evidence_string, axis=1)

    return summary_df


def make_evidence_string(row):
    flags = []

    flag_cols = [
        "car_close",
        "car_closing",
        "car_braking",
        "ped_speed_unstable",
        "clean_progress",
        "head_checking",
        "ped_accelerating",
        "ped_decelerating",
        "low_motion",
    ]

    for col in flag_cols:
        if bool(row.get(col, False)):
            flags.append(col)

    return "|".join(flags) if flags else "none"


def add_interpretations(summary_df):
    summary_df = summary_df.copy()

    interpretations = []

    for i, row in summary_df.iterrows():
        label = str(row.get("macro_label", ""))

        car_close = bool(row.get("car_close", False))
        car_closing = bool(row.get("car_closing", False))
        car_braking = bool(row.get("car_braking", False))
        unstable = bool(row.get("ped_speed_unstable", False))
        clean_progress = bool(row.get("clean_progress", False))
        head_checking = bool(row.get("head_checking", False))
        ped_accelerating = bool(row.get("ped_accelerating", False))
        ped_decelerating = bool(row.get("ped_decelerating", False))
        low_motion = bool(row.get("low_motion", False))

        prev_car_braking = False
        if i > 0:
            prev_car_braking = bool(summary_df.loc[i - 1, "car_braking"])

        # --------------------------------------------------------
        # Priority rules
        # --------------------------------------------------------

        if low_motion and not car_close:
            interpretation = "waiting_or_post_interaction_low_motion"

        elif low_motion and car_close and head_checking:
            interpretation = "waiting_near_car_with_possible_checking"

        elif unstable and (car_close or car_closing or head_checking):
            interpretation = "possible_hesitation_or_checking"

        elif ped_decelerating and car_close and car_closing and not car_braking:
            interpretation = "possible_yielding_or_slowing_for_car"

        elif ped_accelerating and (prev_car_braking or car_braking) and car_close:
            interpretation = "possible_response_to_car_yielding"

        elif clean_progress and car_close and car_closing:
            interpretation = "crossing_interaction"

        elif clean_progress and not car_close:
            interpretation = "main_movement_no_close_car_interaction"

        elif ped_accelerating:
            interpretation = "committing_or_accelerating"

        elif ped_decelerating:
            interpretation = "decelerating_or_exiting_interaction"

        elif head_checking and car_close:
            interpretation = "possible_checking_near_car"

        else:
            interpretation = f"macro_phase_{label}"

        interpretations.append(interpretation)

    summary_df["possible_interpretation"] = interpretations

    return summary_df


# ============================================================
# Frame-level assignment
# ============================================================

def add_segment_labels_to_frames(df, summary_df):
    df = df.copy()

    df["segment_id"] = -1
    df["macro_label"] = "unassigned"
    df["possible_interpretation"] = "unassigned"

    for _, row in summary_df.iterrows():
        s = int(row["start_idx"])
        e = int(row["end_idx"])

        df.loc[s:e, "segment_id"] = int(row["segment_id"])
        df.loc[s:e, "macro_label"] = row["macro_label"]
        df.loc[s:e, "possible_interpretation"] = row["possible_interpretation"]

    return df


# ============================================================
# Main
# ============================================================

def main():
    print("\nLoading feature CSV...")
    df = pd.read_csv(FEATURE_CSV)

    # Clean column names.
    df.columns = df.columns.astype(str).str.strip()

    if TIME_COL not in df.columns:
        raise ValueError(f"Missing required column: {TIME_COL}")

    ped_speed_col = pick_col(df, PED_SPEED_COLS)

    if ped_speed_col is None:
        raise ValueError(
            f"Missing pedestrian speed column. Tried: {PED_SPEED_COLS}"
        )

    # Match the preprocessing done by the segmentation script.
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    df["elapsed_time"] = build_elapsed_time_seconds(df)

    valid = (
        np.isfinite(to_numeric_array(df[ped_speed_col]))
        & np.isfinite(to_numeric_array(df["elapsed_time"]))
    )

    df = df.loc[valid].reset_index(drop=True)

    print("Loading macro segment CSV...")
    segments_df = pd.read_csv(MACRO_SEGMENTS_CSV)
    segments_df.columns = segments_df.columns.astype(str).str.strip()

    required_segment_cols = ["segment_id", "macro_label", "start_idx", "end_idx"]
    missing = [c for c in required_segment_cols if c not in segments_df.columns]

    if missing:
        raise ValueError(
            f"Macro segment CSV is missing required columns: {missing}"
        )

    print("Computing segment-level summary statistics...")

    summary_rows = []

    for _, segment_row in segments_df.iterrows():
        summary = summarize_one_segment(df, segment_row)
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)

    print("Adding evidence flags...")
    summary_df = add_evidence_flags(summary_df)

    print("Adding rule-based behavioral interpretations...")
    summary_df = add_interpretations(summary_df)

    print("Creating frame-level labeled CSV...")
    frame_level_df = add_segment_labels_to_frames(df, summary_df)

    summary_df.to_csv(SUMMARY_CSV, index=False)
    frame_level_df.to_csv(FRAME_LEVEL_CSV, index=False)

    print("\nDone.")
    print(f"Saved segment summary to:\n  {SUMMARY_CSV}")
    print(f"Saved frame-level labeled data to:\n  {FRAME_LEVEL_CSV}")

    print("\nSegment interpretations:")
    display_cols = [
        "segment_id",
        "macro_label",
        "time_start_sec",
        "time_end_sec",
        "duration_sec",
        "ped_speed_mean",
        "ped_speed_slope",
        "min_car_ped_distance",
        "fraction_time_closing",
        "fraction_car_braking",
        "max_abs_head_turn_rate",
        "max_abs_head_body_yaw_diff",
        "evidence_flags",
        "possible_interpretation",
    ]

    existing_cols = [c for c in display_cols if c in summary_df.columns]

    print(summary_df[existing_cols].to_string(index=False))


if __name__ == "__main__":
    main()