import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the established PedNYC metrics-v1 algorithm.")
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data" / "processed" / "decoded_clean_PedNYC1_scenario3.csv")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "data" / "processed" / "features_PedNYC1_scenario3_metrics_v1.csv")
    parser.add_argument("--segment-csv", type=Path, default=ROOT / "data" / "processed" / "segments_PedNYC1_scenario3_v1.csv")
    parser.add_argument("--graph-dir", type=Path, default=ROOT / "outputs" / "graphs" / "scenario3_metrics_v1")
    parser.add_argument("--fourframe-dir", type=Path, default=ROOT / "4frame_view" / "scenario3_metrics_v1")
    return parser.parse_args()


ARGS = parse_args()

# This is the ONLY source CSV used by this script.
# All output CSVs are built from this decoded-clean file.
INPUT_CSV = ARGS.input_csv

OUTPUT_CSV = ARGS.output_csv
SEGMENT_CSV = ARGS.segment_csv

GRAPH_DIR = ARGS.graph_dir
FOURFRAME_DIR = ARGS.fourframe_dir

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
SEGMENT_CSV.parent.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
FOURFRAME_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Settings
# ============================================================

SPEED_SMOOTH_WINDOW = 9
ACCEL_SMOOTH_WINDOW = 11
DISTANCE_SMOOTH_WINDOW = 9

MIN_SEGMENT_SECONDS = 0.50

PED_NEAR_STOP_SPEED = 0.15
PED_WALKING_SPEED = 0.35
PED_SLOWING_ACCEL = -0.25
PED_STARTING_ACCEL = 0.25
PED_SPEEDING_ACCEL = 0.35

# Later you may need to flip this depending on Unity yaw convention.
POSITIVE_HEAD_YAW_LABEL = "looking_right"
NEGATIVE_HEAD_YAW_LABEL = "looking_left"

# ============================================================
# Load CSV robustly
# ============================================================

def read_csv_auto(path):
    """
    Reads CSV while handling comma, semicolon, tab, or auto-detected delimiters.
    This matters because if the CSV loads as shape (648, 1), the delimiter is wrong.
    """
    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
    ]

    best_df = None

    for kwargs in attempts:
        try:
            temp = pd.read_csv(path, **kwargs)
            temp.columns = temp.columns.astype(str).str.strip()

            if best_df is None or temp.shape[1] > best_df.shape[1]:
                best_df = temp

            if temp.shape[1] > 1:
                return temp

        except Exception:
            continue

    if best_df is None:
        raise ValueError(f"Could not read CSV: {path}")

    return best_df


if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

# Load the clean decoded file.
df = read_csv_auto(INPUT_CSV)
df.columns = df.columns.astype(str).str.strip()

print("Loaded CSV:")
print(INPUT_CSV)
print("Shape:", df.shape)

if df.shape[1] == 1:
    print("\nERROR: CSV still loaded as one column.")
    print("First column name:")
    print(df.columns[0])
    raise ValueError("CSV delimiter problem. File was loaded as one column.")

# Keep a record of the exact decoded-clean columns before we add any metric aliases.
original_decoded_columns = df.columns.tolist()

# Save original column inventory.
for out_dir in [GRAPH_DIR, FOURFRAME_DIR]:
    column_file = out_dir / "00_original_decoded_column_inventory.txt"
    with open(column_file, "w", encoding="utf-8") as f:
        for i, col in enumerate(original_decoded_columns):
            f.write(f"{i:03d}: {col}\n")

print("\nFirst 40 original decoded columns:")
for i, col in enumerate(original_decoded_columns[:40]):
    print(f"{i:03d}: {col}")

# ============================================================
# Column mapping
# ============================================================

def normalize_col_name(name):
    """
    Makes matching more forgiving:
    'B VR Pos X' -> 'bvrposx'
    'B_VR_Pos_X' -> 'bvrposx'
    """
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def make_normalized_lookup(columns):
    lookup = {}
    for col in columns:
        key = normalize_col_name(col)
        # Keep the first occurrence so original decoded columns win.
        if key not in lookup:
            lookup[key] = col
    return lookup


normalized_lookup = make_normalized_lookup(df.columns)

COLUMN_ALIASES = {
    "ScenarioTime": [
        "ScenarioTime",
        "Scenario Time",
        "scenario_time",
    ],

    "GameTime": [
        "GameTime",
        "Game Time",
        "game_time",
    ],

    "dt": [
        "dt",
        "delta time",
        "DeltaTime",
    ],

    # Car position
    "car_x": [
        "A car Pos X",
        "A Car Pos X",
        "A_car_Pos_X",
        "car_x",
    ],

    "car_z": [
        "A car Pos Z",
        "A Car Pos Z",
        "A_car_Pos_Z",
        "car_z",
    ],

    "car_yaw": [
        "A car Rot Y",
        "A Car Rot Y",
        "A_car_Rot_Y",
        "car_yaw",
    ],

    # IMPORTANT: pedestrian position uses VR position, not avatar position.
    "ped_x": [
        "B VR Pos X",
        "B_VR_Pos_X",
        "B VR position X",
        "ped_x",
    ],

    "ped_z": [
        "B VR Pos Z",
        "B_VR_Pos_Z",
        "B VR position Z",
        "ped_z",
    ],

    "ped_yaw": [
        "B VR Rot Y",
        "B_VR_Rot_Y",
        "B VR rotation Y",
        "ped_yaw",
    ],

    # Avatar position is only for noise checking.
    "ped_avatar_x": [
        "B Avatar Pos X",
        "B_Avatar_Pos_X",
        "ped_avatar_x",
    ],

    "ped_avatar_z": [
        "B Avatar Pos Z",
        "B_Avatar_Pos_Z",
        "ped_avatar_z",
    ],

    "ped_avatar_yaw": [
        "B Avatar Rot Y",
        "B_Avatar_Rot_Y",
        "ped_avatar_yaw",
    ],

    # Driver / car controls
    "A_velocity_raw": [
        "A velocity",
        "A_velocity",
        "A_velocity_raw",
    ],

    "A_accel_raw": [
        "A accel",
        "A_accel",
        "A_accel_raw",
    ],

    "A_steering": [
        "A steering",
        "A_steering",
    ],

    "A_indicators": [
        "A indicators",
        "A_indicators",
    ],

    "A_horn": [
        "A Horn Button",
        "A_horn",
        "A horn",
    ],

    # Optional future head features
    "head_yaw": [
        "B  Bone Rot Head Y",
        "B Bone Rot Head Y",
        "B Head Rot Y",
        "B HMD Rot Y",
        "B VR Head Rot Y",
        "B Headset Rot Y",
        "head_yaw",
    ],
}


def find_existing_column(possible_names):
    for name in possible_names:
        key = normalize_col_name(name)
        if key in normalized_lookup:
            return normalized_lookup[key]
    return None


# ============================================================
# Preserve original decoded-clean columns and add aliases
# ============================================================
# DO NOT rename decoded_clean columns in-place.
# We keep every original column from decoded_clean_PedNYC1_scenario3.csv.
# Then we ADD analysis-friendly alias columns such as car_x, ped_x, etc.
# This makes it impossible for ScenarioTime to disappear because of renaming.

alias_source_map = {}

print("\nColumn mapping / alias creation:")
for clean_name, possible_names in COLUMN_ALIASES.items():
    found = find_existing_column(possible_names)

    if found is not None:
        alias_source_map[clean_name] = found

        if clean_name not in df.columns:
            df[clean_name] = df[found]
            print(f"Added alias: {found}  -->  {clean_name}")
        elif clean_name == found:
            print(f"Using existing column: {clean_name}")
        else:
            print(f"Alias already exists: {clean_name}; source candidate was {found}")
    else:
        print(f"Not found: {clean_name}")

# Save final column inventory after aliases and metrics get added later.
# Another inventory is saved again near the end.

# ============================================================
# Required columns
# ============================================================

required_cols = [
    "ScenarioTime",
    "car_x",
    "car_z",
    "ped_x",
    "ped_z",
]

missing = [col for col in required_cols if col not in df.columns]

if missing:
    print("\nERROR: Missing required columns after alias creation:")
    print(missing)

    print("\nAvailable columns:")
    for i, col in enumerate(df.columns):
        print(f"{i:03d}: {col}")

    raise ValueError(f"Missing required columns after alias creation: {missing}")

# ============================================================
# Hard ScenarioTime safety check
# ============================================================

if "ScenarioTime" not in df.columns:
    raise ValueError(
        "STOPPING: ScenarioTime is missing. "
        "No output CSV will be created without ScenarioTime."
    )

# Convert ScenarioTime safely.
df["ScenarioTime"] = pd.to_numeric(df["ScenarioTime"], errors="coerce")

if df["ScenarioTime"].isna().all():
    raise ValueError(
        "STOPPING: ScenarioTime exists but all values became NaN after numeric conversion."
    )

# Keep ScenarioTime as the first column in frame-level output.
df.insert(0, "ScenarioTime", df.pop("ScenarioTime"))

# ============================================================
# Numeric conversion
# ============================================================
# Do not use errors='ignore'. Only convert columns needed for analysis.

numeric_cols = [
    "ScenarioTime",
    "GameTime",
    "dt",

    # car position / rotation
    "car_x",
    "car_z",
    "car_yaw",

    # pedestrian VR position / rotation
    "ped_x",
    "ped_z",
    "ped_yaw",

    # avatar only for noise check
    "ped_avatar_x",
    "ped_avatar_z",
    "ped_avatar_yaw",

    # these are actually numeric
    "A_accel_raw",
    "A_steering",

    # optional head metric
    "head_yaw",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ============================================================
# Time and dt
# ============================================================

time_col = "ScenarioTime"

if "dt" not in df.columns or df["dt"].isna().all():
    df["dt"] = df[time_col].diff()

df["dt"] = df["dt"].replace([np.inf, -np.inf], np.nan)
df["dt"] = df["dt"].fillna(df[time_col].diff())
df["dt"] = df["dt"].fillna(df["dt"].median())

df.loc[df["dt"] <= 0, "dt"] = np.nan
df["dt"] = df["dt"].fillna(df["dt"].median())

if df["dt"].isna().all():
    raise ValueError("STOPPING: dt could not be computed from ScenarioTime.")

# ============================================================
# Helper functions
# ============================================================

def rolling_smooth(series, window):
    return series.rolling(window=window, center=True, min_periods=1).mean()


def compute_speed_xz(x, z, dt):
    dx = x.diff()
    dz = z.diff()
    dist = np.sqrt(dx**2 + dz**2)
    speed = dist / dt
    return speed.replace([np.inf, -np.inf], np.nan).fillna(0)


def compute_accel(speed, dt):
    accel = speed.diff() / dt
    return accel.replace([np.inf, -np.inf], np.nan).fillna(0)


def angle_diff_deg(a, b):
    """
    Smallest signed angle difference between two angles in degrees.
    Output range: -180 to +180.
    """
    return ((a - b + 180) % 360) - 180


def angular_rate_deg(angle_deg, dt):
    """
    Computes angular turn rate while handling wraparound.
    """
    angle_rad = np.deg2rad(angle_deg)
    unwrapped = np.unwrap(angle_rad)
    rate_rad = pd.Series(unwrapped, index=angle_deg.index).diff() / dt
    return np.rad2deg(rate_rad).replace([np.inf, -np.inf], np.nan).fillna(0)


def clip01(x):
    return np.clip(x, 0, 1)


def move_column_first(frame, col):
    frame = frame.copy()
    if col in frame.columns:
        frame.insert(0, col, frame.pop(col))
    return frame

# ============================================================
# Create movement metrics from scratch
# ============================================================

# Car speed from car X/Z.
df["car_speed_xz"] = compute_speed_xz(
    df["car_x"],
    df["car_z"],
    df["dt"],
)

# Pedestrian speed from B VR Pos X/Z.
# Do NOT use B Avatar Pos X/Z here.
df["ped_speed_xz"] = compute_speed_xz(
    df["ped_x"],
    df["ped_z"],
    df["dt"],
)

# Smooth speed.
df["car_speed_xz_smooth"] = rolling_smooth(
    df["car_speed_xz"],
    SPEED_SMOOTH_WINDOW,
)

df["ped_speed_xz_smooth"] = rolling_smooth(
    df["ped_speed_xz"],
    SPEED_SMOOTH_WINDOW,
)

# Acceleration from smoothed speed.
df["car_accel_xz"] = compute_accel(
    df["car_speed_xz_smooth"],
    df["dt"],
)

df["ped_accel_xz"] = compute_accel(
    df["ped_speed_xz_smooth"],
    df["dt"],
)

# Smooth acceleration.
# This is important because raw acceleration is noisy.
df["car_accel_xz_smooth"] = rolling_smooth(
    df["car_accel_xz"],
    ACCEL_SMOOTH_WINDOW,
)

df["ped_accel_xz_smooth"] = rolling_smooth(
    df["ped_accel_xz"],
    ACCEL_SMOOTH_WINDOW,
)

# Car-pedestrian distance using stable VR pedestrian position.
df["car_ped_distance_xz"] = np.sqrt(
    (df["car_x"] - df["ped_x"])**2 +
    (df["car_z"] - df["ped_z"])**2
)

df["car_ped_distance_xz_smooth"] = rolling_smooth(
    df["car_ped_distance_xz"],
    DISTANCE_SMOOTH_WINDOW,
)

# Distance change.
df["distance_change"] = df["car_ped_distance_xz_smooth"].diff().fillna(0)

# Positive means the car and pedestrian are getting closer.
df["distance_closing"] = -df["distance_change"]

df["distance_closing_smooth"] = rolling_smooth(
    df["distance_closing"],
    DISTANCE_SMOOTH_WINDOW,
)

# Avatar is only a noise check.
if {"ped_avatar_x", "ped_avatar_z"}.issubset(df.columns):
    df["avatar_vr_gap_xz"] = np.sqrt(
        (df["ped_avatar_x"] - df["ped_x"])**2 +
        (df["ped_avatar_z"] - df["ped_z"])**2
    )

# ============================================================
# Optional head features
# ============================================================

if "head_yaw" in df.columns:
    df["head_turn_rate"] = angular_rate_deg(df["head_yaw"], df["dt"])

    if "ped_yaw" in df.columns:
        df["head_body_yaw_diff"] = angle_diff_deg(df["head_yaw"], df["ped_yaw"])
    else:
        df["head_body_yaw_diff"] = np.nan

    df["head_state_v3"] = "head_forward"

    df.loc[df["head_body_yaw_diff"] > 20, "head_state_v3"] = POSITIVE_HEAD_YAW_LABEL
    df.loc[df["head_body_yaw_diff"] < -20, "head_state_v3"] = NEGATIVE_HEAD_YAW_LABEL
    df.loc[df["head_turn_rate"].abs() > 45, "head_state_v3"] = "head_turning_or_scanning"

else:
    df["head_state_v3"] = "head_not_available"

# Placeholder for future hand features.
df["hand_state_v3"] = "hand_not_available"

# ============================================================
# Movement state V1/V3 from speed + acceleration
# ============================================================

speed = df["ped_speed_xz_smooth"]
accel = df["ped_accel_xz_smooth"]

df["movement_state_v3"] = "steady_walking"

df.loc[speed < PED_NEAR_STOP_SPEED, "movement_state_v3"] = "near_stopped"

df.loc[
    (speed >= PED_NEAR_STOP_SPEED) &
    (speed < PED_WALKING_SPEED) &
    (accel < 0),
    "movement_state_v3"
] = "slowing_to_stop"

df.loc[
    (speed >= PED_WALKING_SPEED) &
    (accel <= PED_SLOWING_ACCEL),
    "movement_state_v3"
] = "slowing_down"

df.loc[
    (speed < PED_WALKING_SPEED) &
    (accel >= PED_STARTING_ACCEL),
    "movement_state_v3"
] = "starting_to_walk"

df.loc[
    (speed >= PED_WALKING_SPEED) &
    (accel >= PED_SPEEDING_ACCEL),
    "movement_state_v3"
] = "speeding_up"

# ============================================================
# Scores
# ============================================================

# Higher when pedestrian is slow/stopped.
slow_component = clip01((PED_WALKING_SPEED - speed) / PED_WALKING_SPEED)

# Higher when pedestrian is decelerating.
decel_component = clip01((-accel) / 1.0)

# Higher when car and pedestrian are closing in.
closing_component = clip01(df["distance_closing_smooth"] / 0.25)

# Higher when car is relatively close.
distance_component = clip01((10.0 - df["car_ped_distance_xz_smooth"]) / 10.0)

df["hesitation_score_v3"] = clip01(
    0.45 * slow_component +
    0.35 * decel_component +
    0.20 * closing_component
)

df["yield_score_v3"] = clip01(
    0.45 * slow_component +
    0.35 * distance_component +
    0.20 * closing_component
)

# ============================================================
# Segment creation
# ============================================================

def assign_segment_ids(dataframe, state_col):
    changes = dataframe[state_col].ne(dataframe[state_col].shift()).cumsum()
    return changes - 1


def summarize_segments(dataframe):
    rows = []

    for seg_id, group in dataframe.groupby("segment_id", sort=True):
        start_t = group[time_col].iloc[0]
        end_t = group[time_col].iloc[-1]
        mid_t = (start_t + end_t) / 2

        row = {
            "segment_id": int(seg_id),

            # Explicit ScenarioTime columns kept in segment CSV.
            # ScenarioTime is the segment start time for simple compatibility.
            "ScenarioTime": start_t,
            "ScenarioTime_start": start_t,
            "ScenarioTime_end": end_t,
            "ScenarioTime_mid": mid_t,

            # Keep older names too, in case another script expects them.
            "start_time": start_t,
            "end_time": end_t,
            "duration_sec": end_t - start_t,
            "n_frames": len(group),
            "movement_state_v3": group["movement_state_v3"].mode().iloc[0],
            "head_state_v3": group["head_state_v3"].mode().iloc[0],
            "hand_state_v3": group["hand_state_v3"].mode().iloc[0],
            "mean_ped_speed_xz_smooth": group["ped_speed_xz_smooth"].mean(),
            "mean_ped_accel_xz_smooth": group["ped_accel_xz_smooth"].mean(),
            "mean_car_ped_distance_xz": group["car_ped_distance_xz_smooth"].mean(),
            "mean_distance_closing": group["distance_closing_smooth"].mean(),
            "mean_hesitation_score_v3": group["hesitation_score_v3"].mean(),
            "mean_yield_score_v3": group["yield_score_v3"].mean(),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def merge_short_segments(dataframe, min_seconds):
    """
    Merges very short segments into neighboring segments.
    This prevents tiny one-frame label flickers from becoming separate behaviors.
    """
    dataframe = dataframe.copy()

    for _ in range(3):
        dataframe["segment_id"] = assign_segment_ids(dataframe, "movement_state_v3")
        segs = summarize_segments(dataframe)

        short_segs = segs[segs["duration_sec"] < min_seconds]

        if short_segs.empty:
            break

        for _, seg in short_segs.iterrows():
            seg_id = seg["segment_id"]
            idx = dataframe.index[dataframe["segment_id"] == seg_id].tolist()

            if not idx:
                continue

            first_idx = idx[0]
            last_idx = idx[-1]

            prev_state = None
            next_state = None

            prev_rows = dataframe.loc[dataframe.index < first_idx]
            if not prev_rows.empty:
                prev_state = prev_rows["movement_state_v3"].iloc[-1]

            next_rows = dataframe.loc[dataframe.index > last_idx]
            if not next_rows.empty:
                next_state = next_rows["movement_state_v3"].iloc[0]

            replacement_state = prev_state if prev_state is not None else next_state

            if replacement_state is not None:
                dataframe.loc[idx, "movement_state_v3"] = replacement_state

    dataframe["segment_id"] = assign_segment_ids(dataframe, "movement_state_v3")
    return dataframe


df["segment_id"] = assign_segment_ids(df, "movement_state_v3")
df = merge_short_segments(df, MIN_SEGMENT_SECONDS)

# ============================================================
# Behavior phrase
# ============================================================

def build_behavior_phrase(row):
    parts = [row["movement_state_v3"]]

    if row["head_state_v3"] != "head_not_available":
        parts.append(row["head_state_v3"])

    if row["hand_state_v3"] != "hand_not_available":
        parts.append(row["hand_state_v3"])

    if row["hesitation_score_v3"] >= 0.60:
        parts.append("possible_hesitation")

    if row["yield_score_v3"] >= 0.60:
        parts.append("waiting_or_yielding")

    return " + ".join(parts)


df["ped_behavior_phrase_v3"] = df.apply(build_behavior_phrase, axis=1)

segments_df = summarize_segments(df)

# Add segment-level phrase.
segment_phrases = (
    df.groupby("segment_id")["ped_behavior_phrase_v3"]
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)

segments_df = segments_df.merge(segment_phrases, on="segment_id", how="left")

# ============================================================
# Final output safety checks and save outputs
# ============================================================

def require_scenario_time(frame, csv_name, segment_file=False):
    if segment_file:
        needed = {"ScenarioTime", "ScenarioTime_start", "ScenarioTime_end"}
    else:
        needed = {"ScenarioTime"}

    missing_time_cols = needed - set(frame.columns)

    if missing_time_cols:
        raise ValueError(
            f"STOPPING: {csv_name} is missing time columns: {missing_time_cols}"
        )

    if frame.empty:
        raise ValueError(f"STOPPING: {csv_name} is empty.")

    if frame["ScenarioTime"].isna().all():
        raise ValueError(
            f"STOPPING: {csv_name} has ScenarioTime column, but all values are NaN."
        )


# Metrics CSV must have frame-by-frame ScenarioTime.
require_scenario_time(df, OUTPUT_CSV.name, segment_file=False)

# Segment CSV must have ScenarioTime range columns.
require_scenario_time(segments_df, SEGMENT_CSV.name, segment_file=True)

# Keep ScenarioTime first in both outputs.
df = move_column_first(df, "ScenarioTime")
segments_df = move_column_first(segments_df, "ScenarioTime")

# Save final column inventories.
for out_dir in [GRAPH_DIR, FOURFRAME_DIR]:
    column_file = out_dir / "01_final_metrics_column_inventory.txt"
    with open(column_file, "w", encoding="utf-8") as f:
        for i, col in enumerate(df.columns):
            marker = " [ORIGINAL]" if col in original_decoded_columns else " [ADDED]"
            f.write(f"{i:03d}: {col}{marker}\n")

# Save outputs.
df.to_csv(OUTPUT_CSV, index=False)
segments_df.to_csv(SEGMENT_CSV, index=False)

# Also save copies for 4-frame/video comparison.
df.to_csv(FOURFRAME_DIR / OUTPUT_CSV.name, index=False)
segments_df.to_csv(FOURFRAME_DIR / SEGMENT_CSV.name, index=False)

print("\nFinal CSV checks passed.")
print(f"{OUTPUT_CSV.name}: ScenarioTime preserved as first column.")
print(f"{SEGMENT_CSV.name}: ScenarioTime, ScenarioTime_start, and ScenarioTime_end preserved.")

print("\nSaved metrics CSV:")
print(OUTPUT_CSV)

print("\nSaved segment CSV:")
print(SEGMENT_CSV)

print("\nSaved 4frame_view copies:")
print(FOURFRAME_DIR / OUTPUT_CSV.name)
print(FOURFRAME_DIR / SEGMENT_CSV.name)

# ============================================================
# Plot helpers
# ============================================================

def save_current_fig(filename):
    for out_dir in [GRAPH_DIR, FOURFRAME_DIR]:
        out_path = out_dir / filename
        plt.savefig(out_path, dpi=200)
        print(f"Saved graph: {out_path}")


def save_plot(columns, title, filename, ylabel="Value"):
    existing = [col for col in columns if col in df.columns]

    if not existing:
        print(f"Skipping plot: {title}")
        return

    plt.figure(figsize=(14, 6))

    for col in existing:
        plt.plot(df[time_col], df[col], label=col)

    plt.xlabel("ScenarioTime")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_current_fig(filename)
    plt.close()


def save_colored_segment_plot():
    color_map = {
        "steady_walking": "tab:blue",
        "slowing_down": "tab:orange",
        "slowing_to_stop": "tab:orange",
        "near_stopped": "tab:red",
        "starting_to_walk": "tab:green",
        "speeding_up": "tab:green",
    }

    plt.figure(figsize=(16, 7))
    ax = plt.gca()

    used_labels = set()

    for _, seg in segments_df.iterrows():
        seg_id = seg["segment_id"]
        state = seg["movement_state_v3"]

        group = df[df["segment_id"] == seg_id]

        color = color_map.get(state, "tab:gray")
        label = state if state not in used_labels else None
        used_labels.add(state)

        ax.plot(
            group[time_col],
            group["ped_speed_xz_smooth"],
            color=color,
            linewidth=3,
            label=label,
        )

        ax.axvspan(
            group[time_col].iloc[0],
            group[time_col].iloc[-1],
            color=color,
            alpha=0.08,
        )

    ax.set_xlabel("ScenarioTime")
    ax.set_ylabel("Pedestrian speed XZ smooth")
    ax.set_title("Pedestrian Movement Segments V1: Colored Speed Graph")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    save_current_fig("04_colored_ped_speed_segments_v1.png")
    plt.close()

# ============================================================
# Create graphs
# ============================================================

save_plot(
    ["ped_speed_xz", "ped_speed_xz_smooth"],
    "Pedestrian Speed Using B VR Position: Raw vs Smoothed",
    "01_ped_speed_raw_vs_smooth.png",
    ylabel="Speed",
)

save_plot(
    ["ped_accel_xz", "ped_accel_xz_smooth"],
    "Pedestrian Acceleration Using B VR Position: Raw vs Smoothed",
    "02_ped_accel_raw_vs_smooth.png",
    ylabel="Acceleration",
)

save_plot(
    ["car_speed_xz", "car_speed_xz_smooth"],
    "Car Speed: Raw vs Smoothed",
    "03_car_speed_raw_vs_smooth.png",
    ylabel="Speed",
)

save_colored_segment_plot()

save_plot(
    ["car_ped_distance_xz", "car_ped_distance_xz_smooth"],
    "Car-Pedestrian Distance Using B VR Position",
    "05_car_ped_distance.png",
    ylabel="Distance",
)

save_plot(
    ["distance_change", "distance_closing", "distance_closing_smooth"],
    "Distance Change and Closing Behavior",
    "06_distance_closing.png",
    ylabel="Distance change",
)

save_plot(
    ["car_accel_xz_smooth", "ped_accel_xz_smooth"],
    "Smoothed Car and Pedestrian Acceleration",
    "07_smoothed_acceleration_comparison.png",
    ylabel="Acceleration",
)

save_plot(
    ["ped_speed_xz_smooth", "ped_accel_xz_smooth", "car_ped_distance_xz_smooth"],
    "Core Movement and Interaction Signals",
    "08_core_movement_interaction_signals.png",
    ylabel="Value",
)

if "avatar_vr_gap_xz" in df.columns:
    save_plot(
        ["avatar_vr_gap_xz"],
        "Avatar vs B VR Position Gap: Noise Check Only",
        "09_avatar_vr_gap_noise_check.png",
        ylabel="Avatar-VR gap",
    )

if "head_turn_rate" in df.columns:
    save_plot(
        ["head_yaw", "head_turn_rate", "head_body_yaw_diff"],
        "Optional Head Features",
        "10_optional_head_features.png",
        ylabel="Degrees / degrees per second",
    )

print("\nDone.")
print(f"Graphs saved to: {GRAPH_DIR}")
print(f"4frame_view copies saved to: {FOURFRAME_DIR}")
