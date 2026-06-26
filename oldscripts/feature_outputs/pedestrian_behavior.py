from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

INPUT_CSV = Path(
    r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis\feature_outputs\features_PedNYC1_scenario3_smoothed_accel.csv"
)

OUTPUT_LABELED_CSV = INPUT_CSV.with_name(
    INPUT_CSV.stem + "_ped_behavior_labeled.csv"
)

OUTPUT_SEGMENTS_CSV = INPUT_CSV.with_name(
    INPUT_CSV.stem + "_ped_behavior_segments.csv"
)


# ============================================================
# 2. SETTINGS / THRESHOLDS
# ============================================================

# These are first-pass thresholds.
# You can tune them after watching the FFmpeg verification video.

STOP_SPEED = 0.15          # m/s, below this = stopped
WALK_SPEED = 0.30          # m/s, above this = walking
STARTING_ACCEL = 0.20      # m/s^2, positive acceleration
STOPPING_ACCEL = -0.20     # m/s^2, negative acceleration
SLOWING_ACCEL = -0.35      # m/s^2, strong negative acceleration

MIN_SEGMENT_DURATION = 0.45     # seconds; removes tiny noisy label flips
HESITATION_MAX_DURATION = 1.50  # seconds; short stop/slowing between walking = hesitation

SMOOTH_WINDOW_SECONDS = 0.50    # rolling smoothing window for behavior logic


# Old empty/weak labels to ignore.
# We do NOT use these to infer behavior.
IGNORE_LABEL_COLUMNS = [
    "ped_stopped",
    "ped_walking",
    "car_slowing",
    "close_interaction",
    "ped_possible_noise",
    "car_possible_noise",
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes weird extra spaces from CSV column names.
    Example: 'dt                ' becomes 'dt'
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
            + "\n\nAvailable columns are:\n"
            + "\n".join(df.columns)
        )


def get_median_dt_seconds(time_values: np.ndarray) -> float:
    diffs = np.diff(time_values)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        raise ValueError("Could not compute dt from ScenarioTime.")

    return float(np.median(diffs))


def rolling_smooth(series: pd.Series, window_rows: int) -> pd.Series:
    """
    Median then mean smoothing.
    Median removes spikes.
    Mean makes it less jumpy.
    """
    return (
        series
        .rolling(window=window_rows, center=True, min_periods=1)
        .median()
        .rolling(window=window_rows, center=True, min_periods=1)
        .mean()
    )


def create_segments(labels: np.ndarray, times: np.ndarray) -> pd.DataFrame:
    """
    Converts frame-level labels into segment-level rows.
    """
    labels = np.asarray(labels)
    times = np.asarray(times)

    if len(labels) == 0:
        return pd.DataFrame()

    starts = [0]
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            starts.append(i)

    rows = []
    for seg_num, start_idx in enumerate(starts, start=1):
        end_idx_exclusive = starts[seg_num] if seg_num < len(starts) else len(labels)

        start_time = float(times[start_idx])
        end_time = float(times[end_idx_exclusive - 1])
        duration = end_time - start_time

        rows.append({
            "segment_id": seg_num,
            "start_idx": start_idx,
            "end_idx_exclusive": end_idx_exclusive,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration,
            "ped_behavior_inferred": labels[start_idx],
        })

    return pd.DataFrame(rows)


def smooth_short_segments(labels: np.ndarray, times: np.ndarray, min_duration: float) -> np.ndarray:
    """
    Removes tiny label flips.

    Example:
    walking, walking, transition, walking, walking

    If transition is very short, replace it with walking.
    """
    labels = labels.copy()

    for _ in range(5):
        segments = create_segments(labels, times)
        if segments.empty:
            break

        changed = False

        for _, seg in segments.iterrows():
            duration = seg["duration_seconds"]

            if duration >= min_duration:
                continue

            start = int(seg["start_idx"])
            end = int(seg["end_idx_exclusive"])

            prev_label = labels[start - 1] if start > 0 else None
            next_label = labels[end] if end < len(labels) else None

            if prev_label is not None and next_label is not None:
                if prev_label == next_label:
                    replacement = prev_label
                else:
                    # Prefer the previous label if neighbors disagree.
                    replacement = prev_label
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


def apply_hesitation_logic(labels: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Marks short stop/slowing/stopping segments between movement as hesitation.

    Example:
    walking -> stopping -> walking

    becomes:
    walking -> hesitating -> walking
    """
    labels = labels.copy()
    segments = create_segments(labels, times)

    movement_labels = {"walking", "starting"}
    possible_hesitation_labels = {"stopped", "stopping", "slowing", "transition"}

    for i in range(1, len(segments) - 1):
        prev_seg = segments.iloc[i - 1]
        curr_seg = segments.iloc[i]
        next_seg = segments.iloc[i + 1]

        prev_label = prev_seg["ped_behavior_inferred"]
        curr_label = curr_seg["ped_behavior_inferred"]
        next_label = next_seg["ped_behavior_inferred"]

        curr_duration = curr_seg["duration_seconds"]

        if (
            curr_label in possible_hesitation_labels
            and prev_label in movement_labels
            and next_label in movement_labels
            and curr_duration <= HESITATION_MAX_DURATION
        ):
            start = int(curr_seg["start_idx"])
            end = int(curr_seg["end_idx_exclusive"])
            labels[start:end] = "hesitating"

    return labels


def assign_confidence(label: str) -> str:
    """
    Confidence is not statistical.
    It is just a practical confidence level for interpreting the rule labels.
    """
    high = {"stopped", "walking"}
    medium = {"starting", "stopping", "slowing", "stepping_back"}
    low = {"transition", "hesitating", "unknown"}

    if label in high:
        return "high"
    if label in medium:
        return "medium"
    if label in low:
        return "low"

    return "low"


# ============================================================
# 4. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)
df = clean_column_names(df)

# We ignore the old empty labels, but keep them in the output for reference.
existing_ignored = [c for c in IGNORE_LABEL_COLUMNS if c in df.columns]
print("Ignoring old label columns:", existing_ignored)

required = [
    "ScenarioTime",
    "ped_x",
    "ped_z",
    "ped_speed_xz_smooth",
    "ped_accel_xz",
]

require_columns(df, required)

df = df.sort_values("ScenarioTime").reset_index(drop=True)

time = df["ScenarioTime"].to_numpy(dtype=float)
median_dt = get_median_dt_seconds(time)

window_rows = max(3, int(round(SMOOTH_WINDOW_SECONDS / median_dt)))

# Make window odd for cleaner centered rolling behavior.
if window_rows % 2 == 0:
    window_rows += 1

print(f"Median dt: {median_dt:.4f} seconds")
print(f"Behavior smoothing window: {window_rows} rows")


# ============================================================
# 5. SMOOTH PEDESTRIAN MOTION FEATURES
# ============================================================

df["ped_speed_behavior_smooth"] = rolling_smooth(
    df["ped_speed_xz_smooth"].astype(float),
    window_rows
)

df["ped_accel_behavior_smooth"] = rolling_smooth(
    df["ped_accel_xz"].astype(float),
    window_rows
)


# ============================================================
# 6. OPTIONAL: PEDESTRIAN PROGRESS / STEPPING BACK
# ============================================================

# This estimates the main direction the pedestrian moves over the scenario.
# If movement goes opposite that direction, we can call it stepping_back.

x = df["ped_x"].to_numpy(dtype=float)
z = df["ped_z"].to_numpy(dtype=float)

start_pos = np.array([np.nanmedian(x[:window_rows]), np.nanmedian(z[:window_rows])])
end_pos = np.array([np.nanmedian(x[-window_rows:]), np.nanmedian(z[-window_rows:])])

main_direction = end_pos - start_pos
main_direction_norm = np.linalg.norm(main_direction)

if main_direction_norm > 1e-6:
    unit_direction = main_direction / main_direction_norm

    rel_pos = np.column_stack([x, z]) - start_pos
    progress = rel_pos @ unit_direction

    progress_velocity = np.gradient(progress, time)

    df["ped_progress_m"] = progress
    df["ped_progress_velocity_mps"] = rolling_smooth(
        pd.Series(progress_velocity),
        window_rows
    )
else:
    df["ped_progress_m"] = np.nan
    df["ped_progress_velocity_mps"] = np.nan


# ============================================================
# 7. FRAME-LEVEL RULE-BASED LABELING
# ============================================================

speed = df["ped_speed_behavior_smooth"].to_numpy(dtype=float)
accel = df["ped_accel_behavior_smooth"].to_numpy(dtype=float)
progress_velocity = df["ped_progress_velocity_mps"].to_numpy(dtype=float)

labels = np.full(len(df), "unknown", dtype=object)

# Basic movement state
labels[speed < STOP_SPEED] = "stopped"

labels[(speed >= STOP_SPEED) & (speed < WALK_SPEED)] = "transition"

labels[speed >= WALK_SPEED] = "walking"

# Starting / stopping / slowing logic
starting_mask = (
    (speed >= STOP_SPEED)
    & (speed < WALK_SPEED)
    & (accel >= STARTING_ACCEL)
)

stopping_mask = (
    (speed >= STOP_SPEED)
    & (speed < WALK_SPEED)
    & (accel <= STOPPING_ACCEL)
)

slowing_mask = (
    (speed >= WALK_SPEED)
    & (accel <= SLOWING_ACCEL)
)

labels[starting_mask] = "starting"
labels[stopping_mask] = "stopping"
labels[slowing_mask] = "slowing"

# Stepping back logic
# Only apply if progress velocity is meaningful and opposite the main path.
stepping_back_mask = (
    np.isfinite(progress_velocity)
    & (speed >= STOP_SPEED)
    & (progress_velocity < -0.10)
)

labels[stepping_back_mask] = "stepping_back"


# ============================================================
# 8. CLEAN LABELS INTO BEHAVIOR SEGMENTS
# ============================================================

labels = smooth_short_segments(
    labels=labels,
    times=time,
    min_duration=MIN_SEGMENT_DURATION
)

labels = apply_hesitation_logic(labels, time)

labels = smooth_short_segments(
    labels=labels,
    times=time,
    min_duration=MIN_SEGMENT_DURATION
)


# ============================================================
# 9. SAVE FRAME-LEVEL LABELED CSV
# ============================================================

df["ped_behavior_inferred"] = labels

segments = create_segments(labels, time)

# Assign segment IDs back to every row
df["ped_behavior_segment_id"] = -1

for _, seg in segments.iterrows():
    start = int(seg["start_idx"])
    end = int(seg["end_idx_exclusive"])
    segment_id = int(seg["segment_id"])
    df.loc[start:end - 1, "ped_behavior_segment_id"] = segment_id

df["ped_behavior_confidence"] = df["ped_behavior_inferred"].apply(assign_confidence)

df.to_csv(OUTPUT_LABELED_CSV, index=False)


# ============================================================
# 10. SAVE SEGMENT-LEVEL SUMMARY CSV
# ============================================================

segment_rows = []

for _, seg in segments.iterrows():
    start = int(seg["start_idx"])
    end = int(seg["end_idx_exclusive"])

    chunk = df.iloc[start:end]

    segment_rows.append({
        "segment_id": int(seg["segment_id"]),
        "start_time": seg["start_time"],
        "end_time": seg["end_time"],
        "duration_seconds": seg["duration_seconds"],
        "ped_behavior_inferred": seg["ped_behavior_inferred"],
        "ped_behavior_confidence": assign_confidence(seg["ped_behavior_inferred"]),
        "mean_ped_speed": chunk["ped_speed_behavior_smooth"].mean(),
        "max_ped_speed": chunk["ped_speed_behavior_smooth"].max(),
        "mean_ped_accel": chunk["ped_accel_behavior_smooth"].mean(),
        "min_ped_accel": chunk["ped_accel_behavior_smooth"].min(),
        "start_ped_x": chunk["ped_x"].iloc[0],
        "start_ped_z": chunk["ped_z"].iloc[0],
        "end_ped_x": chunk["ped_x"].iloc[-1],
        "end_ped_z": chunk["ped_z"].iloc[-1],
    })

segments_out = pd.DataFrame(segment_rows)
segments_out.to_csv(OUTPUT_SEGMENTS_CSV, index=False)


# ============================================================
# 11. PRINT SUMMARY
# ============================================================

print("\nSaved labeled CSV:")
print(OUTPUT_LABELED_CSV)

print("\nSaved behavior segment CSV:")
print(OUTPUT_SEGMENTS_CSV)

print("\nBehavior segments:")
print(
    segments_out[
        [
            "segment_id",
            "start_time",
            "end_time",
            "duration_seconds",
            "ped_behavior_inferred",
            "ped_behavior_confidence",
            "mean_ped_speed",
            "mean_ped_accel",
        ]
    ].to_string(index=False)
)