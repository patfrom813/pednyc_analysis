from pathlib import Path
import pandas as pd
import numpy as np
import ast
import math
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis")

INPUT_FILE = BASE_DIR / "decoded_clean_PedNYC1_scenario3.csv"

OUTPUT_DIR = BASE_DIR / "feature_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

OUTPUT_FEATURE_FILE = OUTPUT_DIR / "features_PedNYC1_scenario3.csv"


# ============================================================
# 2. SETTINGS / THRESHOLDS
# ============================================================

# Pedestrian behavior thresholds
PED_STOPPED_SPEED = 0.15
PED_WALKING_SPEED = 0.30

# Car behavior thresholds
CAR_SLOWING_ACCEL = -0.50

# Interaction threshold
CLOSE_DISTANCE = 10.0

# Noise thresholds
MAX_REASONABLE_PED_SPEED = 3.0
MAX_REASONABLE_PED_ACCEL = 8.0
MAX_REASONABLE_CAR_SPEED = 40.0
MAX_REASONABLE_CAR_ACCEL = 15.0

# Avatar vs VR gap threshold
MAX_REASONABLE_AVATAR_VR_GAP = 1.5


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def clean_column_names(df):
    """
    Cleans weird quote marks and spaces from Unity-exported column names.
    """
    df.columns = (
        df.columns
        .astype(str)
        .str.replace('"', '', regex=False)
        .str.replace("'", "", regex=False)
        .str.strip()
    )
    return df


def find_col(df, possible_names, required=True):
    """
    Finds the first matching column from a list of possible column names.
    This helps if one file says 'A car Pos X' and another says 'A car X'.
    """
    for name in possible_names:
        if name in df.columns:
            return name

    if required:
        raise ValueError(
            f"Missing required column. Tried these names: {possible_names}"
        )

    return None


def numeric_col(df, possible_names, required=True):
    """
    Gets a numeric column safely.
    """
    col = find_col(df, possible_names, required=required)

    if col is None:
        return pd.Series(np.nan, index=df.index)

    return pd.to_numeric(df[col], errors="coerce")


def parse_number_or_vector_magnitude(value):
    """
    Handles values that may be:
    - normal numbers
    - strings like '1.25'
    - vector strings like '(1.0, 0.0, 2.0)'

    For vectors, it returns XZ magnitude because this project cares about ground-plane motion.
    """
    if pd.isna(value):
        return np.nan

    try:
        return float(value)
    except:
        pass

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
            x = float(parsed[0])
            z = float(parsed[2])
            return math.sqrt(x**2 + z**2)

    except:
        return np.nan

    return np.nan


def numeric_or_vector_col(df, possible_names, required=False):
    """
    Useful for columns like A velocity or A accel if they are stored as numbers
    or vector strings.
    """
    col = find_col(df, possible_names, required=required)

    if col is None:
        return pd.Series(np.nan, index=df.index)

    return df[col].apply(parse_number_or_vector_magnitude)


def compute_speed_xz(x, z, t):
    """
    Speed from X/Z position over time.
    """
    dx = x.diff()
    dz = z.diff()
    dt = t.diff()

    dt = dt.where(dt > 0, np.nan)

    speed = np.sqrt(dx**2 + dz**2) / dt
    return speed


def compute_accel(speed, t):
    """
    Acceleration from speed over time.
    """
    dt = t.diff()
    dt = dt.where(dt > 0, np.nan)

    accel = speed.diff() / dt
    return accel


def compute_distance_xz(x1, z1, x2, z2):
    """
    Ground-plane distance between two X/Z points.
    """
    return np.sqrt((x1 - x2)**2 + (z1 - z2)**2)


# ============================================================
# 4. LOAD RAW CSV
# ============================================================

df = pd.read_csv(INPUT_FILE, sep=None, engine="python")
df = clean_column_names(df)

print("Loaded file:")
print(INPUT_FILE)
print()
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print()


# ============================================================
# 5. EXTRACT IMPORTANT RAW COLUMNS
# ============================================================

features = pd.DataFrame()

# ----------------------------
# Time/frame columns
# ----------------------------

features["ScenarioTime"] = numeric_col(df, ["ScenarioTime"])
features["GameTime"] = numeric_col(df, ["GameTime"], required=False)
features["Frame Number"] = numeric_col(df, ["Frame Number", "FrameNumber"])
features["FrameRate"] = numeric_col(df, ["FrameRate"], required=False)
features["FrameRate-XRDevice"] = numeric_col(df, ["FrameRate-XRDevice"], required=False)

features["dt"] = features["ScenarioTime"].diff()
features.loc[features["dt"] <= 0, "dt"] = np.nan


# ----------------------------
# Car columns
# ----------------------------

features["car_x"] = numeric_col(df, ["A car Pos X", "A car X", "A Car Pos X"])
features["car_z"] = numeric_col(df, ["A car Pos Z", "A car Z", "A Car Pos Z"])
features["car_yaw"] = numeric_col(df, ["A car Rot Y", "A car Rotation Y", "A car Yaw"], required=False)

features["A_velocity_raw"] = numeric_or_vector_col(df, ["A velocity", "A Velocity"], required=False)
features["A_accel_raw"] = numeric_or_vector_col(df, ["A accel", "A Accel", "A acceleration"], required=False)
features["A_steering"] = numeric_col(df, ["A steering", "A Steering"], required=False)
features["A_indicators"] = numeric_col(df, ["A indicators", "A Indicators"], required=False)
features["A_horn"] = numeric_col(df, ["A Horn Button", "A horn", "A Horn"], required=False)


# ----------------------------
# Pedestrian columns
# Main pedestrian position = B VR Pos
# Avatar is only for comparison/noise
# ----------------------------

features["ped_x"] = numeric_col(df, ["B VR Pos X", "B VR X"])
features["ped_z"] = numeric_col(df, ["B VR Pos Z", "B VR Z"])
features["ped_yaw"] = numeric_col(df, ["B VR Rot Y", "B VR Rotation Y", "B VR Yaw"], required=False)

features["ped_avatar_x"] = numeric_col(df, ["B Avatar Pos X", "B Avatar X"], required=False)
features["ped_avatar_z"] = numeric_col(df, ["B Avatar Pos Z", "B Avatar Z"], required=False)
features["ped_avatar_yaw"] = numeric_col(
    df,
    ["B Avatar Rot Y", "B Avatar Rotation Y", "B Avatar Yaw"],
    required=False
)


# ============================================================
# 6. DERIVED FEATURES
# ============================================================

# ----------------------------
# Speed from position
# ----------------------------

features["car_speed_xz"] = compute_speed_xz(
    features["car_x"],
    features["car_z"],
    features["ScenarioTime"]
)

features["ped_speed_xz"] = compute_speed_xz(
    features["ped_x"],
    features["ped_z"],
    features["ScenarioTime"]
)

# Optional smoothing for cleaner behavior flags
features["car_speed_xz_smooth"] = (
    features["car_speed_xz"]
    .rolling(window=5, center=True, min_periods=1)
    .median()
)

features["ped_speed_xz_smooth"] = (
    features["ped_speed_xz"]
    .rolling(window=5, center=True, min_periods=1)
    .median()
)


# ----------------------------
# Acceleration
# ----------------------------

features["car_accel_xz"] = compute_accel(
    features["car_speed_xz_smooth"],
    features["ScenarioTime"]
)

features["ped_accel_xz"] = compute_accel(
    features["ped_speed_xz_smooth"],
    features["ScenarioTime"]
)


# ----------------------------
# Car-pedestrian distance
# IMPORTANT:
# Pedestrian position uses B VR Pos X/Z, NOT Avatar.
# ----------------------------

features["car_ped_distance_xz"] = compute_distance_xz(
    features["car_x"],
    features["car_z"],
    features["ped_x"],
    features["ped_z"]
)

features["distance_change"] = (
    features["car_ped_distance_xz"].diff()
    / features["dt"]
)

# Positive distance_closing means car and pedestrian are getting closer
features["distance_closing"] = -features["distance_change"]


# ----------------------------
# Avatar vs VR gap
# This is NOT the main pedestrian position.
# This is only for diagnosing Avatar spikes/noise.
# ----------------------------

features["avatar_vr_gap_xz"] = compute_distance_xz(
    features["ped_avatar_x"],
    features["ped_avatar_z"],
    features["ped_x"],
    features["ped_z"]
)


# ============================================================
# 7. RULE-BASED BEHAVIOR FLAGS
# ============================================================

features["ped_stopped"] = features["ped_speed_xz_smooth"] < PED_STOPPED_SPEED

features["ped_walking"] = features["ped_speed_xz_smooth"] > PED_WALKING_SPEED

features["car_slowing"] = features["car_accel_xz"] < CAR_SLOWING_ACCEL

features["close_interaction"] = (
    (features["car_ped_distance_xz"] < CLOSE_DISTANCE)
    & (features["distance_closing"] > 0)
)


# ============================================================
# 8. NOISE FLAGS
# ============================================================

features["ped_possible_noise"] = (
    (features["ped_speed_xz"] > MAX_REASONABLE_PED_SPEED)
    | (features["ped_accel_xz"].abs() > MAX_REASONABLE_PED_ACCEL)
    | (features["avatar_vr_gap_xz"] > MAX_REASONABLE_AVATAR_VR_GAP)
)

features["car_possible_noise"] = (
    (features["car_speed_xz"] > MAX_REASONABLE_CAR_SPEED)
    | (features["car_accel_xz"].abs() > MAX_REASONABLE_CAR_ACCEL)
)


# ============================================================
# 9. SAVE FEATURE CSV
# ============================================================

features.to_csv(OUTPUT_FEATURE_FILE, index=False)

print("Saved feature CSV:")
print(OUTPUT_FEATURE_FILE)
print()

print("Feature columns:")
for col in features.columns:
    print(" -", col)

print()
print("Basic summary:")
print(features.describe())


# ============================================================
# 10. PLOTS
# ============================================================

# ----------------------------
# Plot 1: Top-down XZ path
# ----------------------------

plt.figure(figsize=(8, 6))
plt.plot(features["car_x"], features["car_z"], label="Car path")
plt.plot(features["ped_x"], features["ped_z"], label="Pedestrian VR path")

if not features["ped_avatar_x"].isna().all():
    plt.plot(
        features["ped_avatar_x"],
        features["ped_avatar_z"],
        label="Pedestrian Avatar path",
        alpha=0.5
    )

plt.xlabel("X position")
plt.ylabel("Z position")
plt.title("Top-down XZ Path")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.savefig(PLOT_DIR / "01_xz_path.png", dpi=200, bbox_inches="tight")
plt.close()


# ----------------------------
# Plot 2: Distance over time
# ----------------------------

plt.figure(figsize=(10, 5))
plt.plot(features["ScenarioTime"], features["car_ped_distance_xz"])
plt.xlabel("ScenarioTime")
plt.ylabel("Car-pedestrian distance XZ")
plt.title("Car-Pedestrian Distance Over Time")
plt.grid(True)
plt.savefig(PLOT_DIR / "02_distance_over_time.png", dpi=200, bbox_inches="tight")
plt.close()


# ----------------------------
# Plot 3: Speeds over time
# ----------------------------

plt.figure(figsize=(10, 5))
plt.plot(features["ScenarioTime"], features["car_speed_xz_smooth"], label="Car speed XZ")
plt.plot(features["ScenarioTime"], features["ped_speed_xz_smooth"], label="Pedestrian VR speed XZ")
plt.xlabel("ScenarioTime")
plt.ylabel("Speed XZ")
plt.title("Car and Pedestrian Speed Over Time")
plt.legend()
plt.grid(True)
plt.savefig(PLOT_DIR / "03_speed_over_time.png", dpi=200, bbox_inches="tight")
plt.close()


# ----------------------------
# Plot 4: Acceleration over time
# ----------------------------

plt.figure(figsize=(10, 5))
plt.plot(features["ScenarioTime"], features["car_accel_xz"], label="Car accel XZ")
plt.plot(features["ScenarioTime"], features["ped_accel_xz"], label="Pedestrian accel XZ")
plt.xlabel("ScenarioTime")
plt.ylabel("Acceleration XZ")
plt.title("Car and Pedestrian Acceleration Over Time")
plt.legend()
plt.grid(True)
plt.savefig(PLOT_DIR / "04_accel_over_time.png", dpi=200, bbox_inches="tight")
plt.close()


# ----------------------------
# Plot 5: Avatar-VR gap
# ----------------------------

plt.figure(figsize=(10, 5))
plt.plot(features["ScenarioTime"], features["avatar_vr_gap_xz"])
plt.xlabel("ScenarioTime")
plt.ylabel("Avatar-VR gap XZ")
plt.title("Pedestrian Avatar vs VR Position Gap")
plt.grid(True)
plt.savefig(PLOT_DIR / "05_avatar_vr_gap.png", dpi=200, bbox_inches="tight")
plt.close()


# ----------------------------
# Plot 6: Noise flags
# ----------------------------

plt.figure(figsize=(10, 5))
plt.plot(
    features["ScenarioTime"],
    features["ped_possible_noise"].astype(int),
    label="Ped possible noise"
)
plt.plot(
    features["ScenarioTime"],
    features["car_possible_noise"].astype(int),
    label="Car possible noise"
)
plt.xlabel("ScenarioTime")
plt.ylabel("Noise flag")
plt.title("Possible Noise Flags")
plt.legend()
plt.grid(True)
plt.savefig(PLOT_DIR / "06_noise_flags.png", dpi=200, bbox_inches="tight")
plt.close()


# ----------------------------
# Plot 7: Behavior flags
# ----------------------------

plt.figure(figsize=(10, 5))
plt.plot(
    features["ScenarioTime"],
    features["ped_stopped"].astype(int),
    label="Ped stopped"
)
plt.plot(
    features["ScenarioTime"],
    features["ped_walking"].astype(int),
    label="Ped walking"
)
plt.plot(
    features["ScenarioTime"],
    features["car_slowing"].astype(int),
    label="Car slowing"
)
plt.plot(
    features["ScenarioTime"],
    features["close_interaction"].astype(int),
    label="Close interaction"
)
plt.xlabel("ScenarioTime")
plt.ylabel("Flag")
plt.title("Rule-Based Behavior Flags")
plt.legend()
plt.grid(True)
plt.savefig(PLOT_DIR / "07_behavior_flags.png", dpi=200, bbox_inches="tight")
plt.close()


print()
print("Saved plots to:")
print(PLOT_DIR)