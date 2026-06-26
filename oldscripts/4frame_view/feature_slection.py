from pathlib import Path
import re
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

RAW_CSV = Path(
    r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis\decoded_clean_PedNYC1_scenario3.csv"
)

FEATURE_CSV = Path(
    r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis\feature_outputs\features_PedNYC1_scenario3_smoothed_accel.csv"
)

OUTPUT_CSV = FEATURE_CSV.with_name(
    FEATURE_CSV.stem + "_with_head_arms.csv"
)

# ============================================================
# HELPERS
# ============================================================

def normalize_col(name):
    """
    Cleans column names so:
    ' B  Bone Rot Head Y   ' becomes 'B Bone Rot Head Y'
    """
    name = str(name)
    name = name.replace('"', '').replace("'", "")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def read_csv_auto(path):
    """
    Reads CSV while trying to auto-detect delimiter.
    Useful because some decoded Unity CSVs may not be plain comma-separated.
    """
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)

    df.columns = [normalize_col(c) for c in df.columns]
    return df


def angular_diff_deg(a, b):
    """
    Returns shortest angle difference between a and b in degrees.
    Handles wraparound like 359 -> 1 degrees.
    """
    return (a - b + 180) % 360 - 180


def angular_rate_deg(angle_series, dt_series):
    angle = angle_series.astype(float).to_numpy()
    dt = dt_series.astype(float).to_numpy()

    rate = np.full(len(angle), np.nan)

    diffs = angular_diff_deg(angle[1:], angle[:-1])
    safe_dt = np.where(dt[1:] > 1e-6, dt[1:], np.nan)

    rate[1:] = diffs / safe_dt
    rate[0] = rate[1] if len(rate) > 1 else np.nan

    return rate


def rolling_smooth(series, window_rows):
    return (
        series
        .rolling(window=window_rows, center=True, min_periods=1)
        .median()
        .rolling(window=window_rows, center=True, min_periods=1)
        .mean()
    )


def xyz_speed(df, x_col, y_col, z_col, dt_col):
    dx = df[x_col].diff()
    dy = df[y_col].diff()
    dz = df[z_col].diff()

    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    dt = df[dt_col].replace(0, np.nan)

    speed = dist / dt
    speed.iloc[0] = speed.iloc[1] if len(speed) > 1 else np.nan

    return speed


# ============================================================
# LOAD DATA
# ============================================================

raw = read_csv_auto(RAW_CSV)
features = read_csv_auto(FEATURE_CSV)

print("\nRaw CSV columns:", len(raw.columns))
print("Feature CSV columns:", len(features.columns))

if len(raw) != len(features):
    raise ValueError(
        f"Row count mismatch:\n"
        f"Raw CSV rows: {len(raw)}\n"
        f"Feature CSV rows: {len(features)}\n\n"
        f"For this simple version, they need to have the same number of rows."
    )

required_feature_cols = ["ScenarioTime", "dt", "ped_yaw"]

missing_feature_cols = [c for c in required_feature_cols if c not in features.columns]
if missing_feature_cols:
    raise ValueError(f"Missing from feature CSV: {missing_feature_cols}")


for col in ["ScenarioTime", "dt", "ped_yaw"]:
    features[col] = pd.to_numeric(features[col], errors="coerce")
# ============================================================
# RAW COLUMNS TO ADD
# ============================================================

raw_to_new = {
    # -------------------------
    # Head position
    # -------------------------
    "B Bone Pos Head X": "head_x",
    "B Bone Pos Head Y": "head_y",
    "B Bone Pos Head Z": "head_z",

    # -------------------------
    # Head rotation
    # X = pitch-ish, Y = yaw-ish, Z = roll-ish
    # -------------------------
    "B Bone Rot Head X": "head_rot_x",
    "B Bone Rot Head Y": "head_rot_y",
    "B Bone Rot Head Z": "head_rot_z",

    # -------------------------
    # Neck / torso context
    # Useful later to compare head turn vs body direction
    # -------------------------
    "B Bone Rot Neck X": "neck_rot_x",
    "B Bone Rot Neck Y": "neck_rot_y",
    "B Bone Rot Neck Z": "neck_rot_z",

    "B Bone Rot Chest X": "chest_rot_x",
    "B Bone Rot Chest Y": "chest_rot_y",
    "B Bone Rot Chest Z": "chest_rot_z",

    "B Bone Rot Hips X": "hips_rot_x",
    "B Bone Rot Hips Y": "hips_rot_y",
    "B Bone Rot Hips Z": "hips_rot_z",

    # -------------------------
    # Shoulder positions
    # -------------------------
    "B Bone Pos LeftShoulder X": "left_shoulder_x",
    "B Bone Pos LeftShoulder Y": "left_shoulder_y",
    "B Bone Pos LeftShoulder Z": "left_shoulder_z",

    "B Bone Pos RightShoulder X": "right_shoulder_x",
    "B Bone Pos RightShoulder Y": "right_shoulder_y",
    "B Bone Pos RightShoulder Z": "right_shoulder_z",

    # -------------------------
    # Upper arm positions
    # -------------------------
    "B Bone Pos LeftUpperArm X": "left_upper_arm_x",
    "B Bone Pos LeftUpperArm Y": "left_upper_arm_y",
    "B Bone Pos LeftUpperArm Z": "left_upper_arm_z",

    "B Bone Pos RightUpperArm X": "right_upper_arm_x",
    "B Bone Pos RightUpperArm Y": "right_upper_arm_y",
    "B Bone Pos RightUpperArm Z": "right_upper_arm_z",

    # -------------------------
    # Lower arm positions
    # -------------------------
    "B Bone Pos LeftLowerArm X": "left_lower_arm_x",
    "B Bone Pos LeftLowerArm Y": "left_lower_arm_y",
    "B Bone Pos LeftLowerArm Z": "left_lower_arm_z",

    "B Bone Pos RightLowerArm X": "right_lower_arm_x",
    "B Bone Pos RightLowerArm Y": "right_lower_arm_y",
    "B Bone Pos RightLowerArm Z": "right_lower_arm_z",

    # -------------------------
    # Hand positions
    # These are useful for possible signaling later.
    # -------------------------
    "B Bone Pos LeftHand X": "left_hand_x",
    "B Bone Pos LeftHand Y": "left_hand_y",
    "B Bone Pos LeftHand Z": "left_hand_z",

    "B Bone Pos RightHand X": "right_hand_x",
    "B Bone Pos RightHand Y": "right_hand_y",
    "B Bone Pos RightHand Z": "right_hand_z",
}

# ============================================================
# ADD RAW COLUMNS INTO FEATURE CSV
# ============================================================

added = []
missing = []

for raw_col, new_col in raw_to_new.items():
    if raw_col in raw.columns:
        features[new_col] = pd.to_numeric(raw[raw_col], errors="coerce")
        added.append((raw_col, new_col))
    else:
        missing.append(raw_col)

print("\n==============================")
print("ADDED RAW COLUMNS")
print("==============================")
for raw_col, new_col in added:
    print(f"{raw_col}  ->  {new_col}")

print("\n==============================")
print("MISSING RAW COLUMNS")
print("==============================")
if missing:
    for col in missing:
        print(col)
else:
    print("None")

# ============================================================
# SIMPLE DERIVED HEAD + ARM FEATURES
# ============================================================

median_dt = features["dt"].median()
window_rows = max(3, int(round(0.50 / median_dt)))

if window_rows % 2 == 0:
    window_rows += 1

print(f"\nMedian dt: {median_dt}")
print(f"Smoothing window: {window_rows} rows")

# Head turn rate
if "head_rot_y" in features.columns:
    features["head_yaw"] = features["head_rot_y"]

    features["head_turn_rate_deg_s"] = angular_rate_deg(
        features["head_yaw"],
        features["dt"]
    )

    features["head_turn_rate_abs_deg_s"] = np.abs(
        features["head_turn_rate_deg_s"]
    )

    features["head_turn_rate_abs_smooth"] = rolling_smooth(
        features["head_turn_rate_abs_deg_s"],
        window_rows
    )

# Head vs body direction
# ped_yaw already exists in your feature CSV from B VR Rot Y.
if "head_yaw" in features.columns and "ped_yaw" in features.columns:
    features["head_body_yaw_diff"] = angular_diff_deg(
        features["head_yaw"],
        features["ped_yaw"]
    )

    features["head_body_yaw_diff_abs"] = np.abs(
        features["head_body_yaw_diff"]
    )

# Left/right hand speed
if all(c in features.columns for c in ["left_hand_x", "left_hand_y", "left_hand_z"]):
    features["left_hand_speed_3d"] = xyz_speed(
        features,
        "left_hand_x",
        "left_hand_y",
        "left_hand_z",
        "dt"
    )

    features["left_hand_speed_3d_smooth"] = rolling_smooth(
        features["left_hand_speed_3d"],
        window_rows
    )

if all(c in features.columns for c in ["right_hand_x", "right_hand_y", "right_hand_z"]):
    features["right_hand_speed_3d"] = xyz_speed(
        features,
        "right_hand_x",
        "right_hand_y",
        "right_hand_z",
        "dt"
    )

    features["right_hand_speed_3d_smooth"] = rolling_smooth(
        features["right_hand_speed_3d"],
        window_rows
    )

# Combined hand/arm activity score
hand_speed_cols = [
    c for c in [
        "left_hand_speed_3d_smooth",
        "right_hand_speed_3d_smooth"
    ]
    if c in features.columns
]

if hand_speed_cols:
    features["hand_activity_speed"] = features[hand_speed_cols].mean(axis=1)

# Hand height relative to head
# Useful later for possible signaling / raised hand.
if "head_y" in features.columns and "left_hand_y" in features.columns:
    features["left_hand_height_relative_to_head"] = (
        features["left_hand_y"] - features["head_y"]
    )

if "head_y" in features.columns and "right_hand_y" in features.columns:
    features["right_hand_height_relative_to_head"] = (
        features["right_hand_y"] - features["head_y"]
    )

# ============================================================
# SAVE
# ============================================================

features.to_csv(OUTPUT_CSV, index=False)

print("\n==============================")
print("SAVED NEW FEATURE CSV")
print("==============================")
print(OUTPUT_CSV)

print("\nNew columns added:")
new_cols = [new for _, new in added]
derived_cols = [
    "head_yaw",
    "head_turn_rate_deg_s",
    "head_turn_rate_abs_deg_s",
    "head_turn_rate_abs_smooth",
    "head_body_yaw_diff",
    "head_body_yaw_diff_abs",
    "left_hand_speed_3d",
    "left_hand_speed_3d_smooth",
    "right_hand_speed_3d",
    "right_hand_speed_3d_smooth",
    "hand_activity_speed",
    "left_hand_height_relative_to_head",
    "right_hand_height_relative_to_head",
]

for col in new_cols + derived_cols:
    if col in features.columns:
        print(col)

print("\nDone.")