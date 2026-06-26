from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PATH
# =========================
CSV_FILE = Path(
    r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis\feature_outputs\features_PedNYC1_scenario3.csv"
)

OUTPUT_DIR = CSV_FILE.parent / "smoothed_accel_check"
OUTPUT_DIR.mkdir(exist_ok=True)

BACKUP_FILE = CSV_FILE.with_name(CSV_FILE.stem + "_before_accel_smoothing.csv")

# =========================
# SETTINGS
# =========================
SMOOTH_WINDOW_SECONDS = 0.60   # try 0.40, 0.60, or 0.80
CAR_SLOWING_ACCEL = -0.50      # same threshold you used before

# =========================
# LOAD
# =========================
df = pd.read_csv(CSV_FILE)

df.columns = (
    df.columns.astype(str)
    .str.replace('"', '', regex=False)
    .str.replace("'", "", regex=False)
    .str.strip()
)

required = [
    "ScenarioTime",
    "dt",
    "car_accel_xz",
    "ped_accel_xz",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# BACKUP ORIGINAL CSV
# =========================
if not BACKUP_FILE.exists():
    df.to_csv(BACKUP_FILE, index=False)
    print(f"Backup saved to: {BACKUP_FILE}")
else:
    print(f"Backup already exists: {BACKUP_FILE}")

# =========================
# SAVE RAW ACCEL COLUMNS
# =========================
if "car_accel_xz_raw_unsmoothed" not in df.columns:
    df["car_accel_xz_raw_unsmoothed"] = df["car_accel_xz"]

if "ped_accel_xz_raw_unsmoothed" not in df.columns:
    df["ped_accel_xz_raw_unsmoothed"] = df["ped_accel_xz"]

# =========================
# SMOOTHING HELPER
# =========================
median_dt = df["dt"].median()
window_frames = int(round(SMOOTH_WINDOW_SECONDS / median_dt))

# Make window odd for centered rolling
if window_frames % 2 == 0:
    window_frames += 1

window_frames = max(window_frames, 3)

print(f"Median dt: {median_dt}")
print(f"Smoothing window: {SMOOTH_WINDOW_SECONDS} seconds ≈ {window_frames} frames")

def smooth_series(series, window):
    """
    Median removes sharp spikes.
    Mean then makes the line smoother.
    """
    median_smoothed = series.rolling(
        window=window,
        center=True,
        min_periods=1
    ).median()

    mean_smoothed = median_smoothed.rolling(
        window=window,
        center=True,
        min_periods=1
    ).mean()

    return mean_smoothed

# =========================
# APPLY SMOOTHING
# =========================
df["car_accel_xz"] = smooth_series(df["car_accel_xz_raw_unsmoothed"], window_frames)
df["ped_accel_xz"] = smooth_series(df["ped_accel_xz_raw_unsmoothed"], window_frames)

# Update car_slowing flag because car acceleration changed
if "car_slowing" in df.columns:
    df["car_slowing"] = (df["car_accel_xz"] < CAR_SLOWING_ACCEL).astype(int)

# =========================
# SAVE UPDATED CSV
# =========================
df.to_csv(CSV_FILE, index=False)
print(f"Updated CSV saved to: {CSV_FILE}")

# Also save a separate copy just in case
SMOOTHED_COPY = CSV_FILE.with_name(CSV_FILE.stem + "_smoothed_accel.csv")
df.to_csv(SMOOTHED_COPY, index=False)
print(f"Smoothed copy saved to: {SMOOTHED_COPY}")

# =========================
# COMPARISON PLOT
# =========================
time = df["ScenarioTime"]

plt.figure(figsize=(15, 6))
plt.plot(time, df["car_accel_xz_raw_unsmoothed"], alpha=0.35, label="car_accel_xz raw")
plt.plot(time, df["car_accel_xz"], linewidth=2, label="car_accel_xz smoothed")
plt.title("Car Acceleration: Raw vs Smoothed")
plt.xlabel("ScenarioTime (seconds)")
plt.ylabel("Acceleration")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "car_accel_raw_vs_smoothed.png", dpi=150)
plt.close()

plt.figure(figsize=(15, 6))
plt.plot(time, df["ped_accel_xz_raw_unsmoothed"], alpha=0.35, label="ped_accel_xz raw")
plt.plot(time, df["ped_accel_xz"], linewidth=2, label="ped_accel_xz smoothed")
plt.title("Pedestrian Acceleration: Raw vs Smoothed")
plt.xlabel("ScenarioTime (seconds)")
plt.ylabel("Acceleration")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ped_accel_raw_vs_smoothed.png", dpi=150)
plt.close()

plt.figure(figsize=(15, 6))
plt.plot(time, df["car_accel_xz"], label="car_accel_xz smoothed")
plt.plot(time, df["ped_accel_xz"], label="ped_accel_xz smoothed")
plt.title("07 Acceleration After Smoothing")
plt.xlabel("ScenarioTime (seconds)")
plt.ylabel("Value")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_acceleration_after_smoothing.png", dpi=150)
plt.close()

print(f"Comparison plots saved to: {OUTPUT_DIR}")
print("Done.")