from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis")

INPUT_FILE = BASE_DIR / "decoded_clean_PedNYC1_scenario3.csv"

OUTPUT_DIR = BASE_DIR / "scenario3_outputs_vr"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load semicolon-delimited corrected file
df = pd.read_csv(INPUT_FILE, sep=";", index_col=False)

# Clean column names
df.columns = (
    df.columns
    .astype(str)
    .str.replace('"', '', regex=False)
    .str.replace("'", "", regex=False)
    .str.strip()
)

# Use VR positions instead of avatar/car body positions
needed_cols = [
    "ScenarioTime",
    "A VR Pos X", "A VR Pos Z",
    "B VR Pos X", "B VR Pos Z"
]

missing = [col for col in needed_cols if col not in df.columns]
if missing:
    print("Missing columns:", missing)
    print("\nAvailable columns containing VR:")
    print([col for col in df.columns if "VR" in col])
    print("\nAll available columns:")
    print(df.columns.tolist())
    raise SystemExit("Column names or delimiter are wrong.")

# Convert to numeric
for col in needed_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Time difference
df["dt"] = df["ScenarioTime"].diff()

# Avoid divide-by-zero or bad dt values
df.loc[df["dt"] <= 0, "dt"] = np.nan

# Car/driver VR speed from A VR position change
df["car_vr_step_xz"] = np.sqrt(
    df["A VR Pos X"].diff()**2 +
    df["A VR Pos Z"].diff()**2
)

df["car_vr_speed_xz"] = df["car_vr_step_xz"] / df["dt"]

# Pedestrian VR speed from B VR position change
df["ped_vr_step_xz"] = np.sqrt(
    df["B VR Pos X"].diff()**2 +
    df["B VR Pos Z"].diff()**2
)

df["ped_vr_speed_xz"] = df["ped_vr_step_xz"] / df["dt"]

# Car-pedestrian VR distance
df["car_ped_vr_distance_xz"] = np.sqrt(
    (df["A VR Pos X"] - df["B VR Pos X"])**2 +
    (df["A VR Pos Z"] - df["B VR Pos Z"])**2
)

# Flags
df["ped_vr_speed_flag"] = df["ped_vr_speed_xz"] > 3
df["car_vr_speed_flag"] = df["car_vr_speed_xz"] > 40

# Save analysis-ready CSV
df.to_csv(OUTPUT_DIR / "analysis_ready_scenario3_vr.csv", sep=";", index=False)

# 1. VR path plot
plt.figure(figsize=(7, 7))
plt.plot(df["A VR Pos X"], df["A VR Pos Z"], label="A VR path / car-driver side")
plt.plot(df["B VR Pos X"], df["B VR Pos Z"], label="B VR path / pedestrian side")
plt.xlabel("X position (m)")
plt.ylabel("Z position (m)")
plt.title("A VR and B VR movement path")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_vr_path_plot.png", dpi=200)
plt.close()

# 2. VR distance plot
plt.figure(figsize=(10, 5))
plt.plot(df["ScenarioTime"], df["car_ped_vr_distance_xz"])
plt.xlabel("Scenario time (s)")
plt.ylabel("VR distance XZ (m)")
plt.title("A VR - B VR distance over time")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_vr_distance_plot.png", dpi=200)
plt.close()

# 3. Car/driver VR speed plot
plt.figure(figsize=(10, 5))
plt.plot(df["ScenarioTime"], df["car_vr_speed_xz"])
plt.axhline(10, linestyle="--", label="10 m/s threshold")
plt.xlabel("Scenario time (s)")
plt.ylabel("A VR speed XZ (m/s)")
plt.title("A VR speed over time")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_a_vr_speed_plot.png", dpi=200)
plt.close()

# 4. Pedestrian VR speed plot
plt.figure(figsize=(10, 5))
plt.plot(df["ScenarioTime"], df["ped_vr_speed_xz"], label="B VR speed")
plt.axhline(3, linestyle="--", label="3 m/s threshold")
plt.xlabel("Scenario time (s)")
plt.ylabel("B VR speed XZ (m/s)")
plt.title("B VR pedestrian speed over time")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_b_vr_pedestrian_speed_plot.png", dpi=200)
plt.close()

# 5. B VR position components
plt.figure(figsize=(10, 5))
plt.plot(df["ScenarioTime"], df["B VR Pos X"], label="B VR Pos X")
plt.plot(df["ScenarioTime"], df["B VR Pos Z"], label="B VR Pos Z")
plt.xlabel("Scenario time (s)")
plt.ylabel("Position (m)")
plt.title("B VR pedestrian position components over time")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_b_vr_position_components.png", dpi=200)
plt.close()

# 6. A VR position components
plt.figure(figsize=(10, 5))
plt.plot(df["ScenarioTime"], df["A VR Pos X"], label="A VR Pos X")
plt.plot(df["ScenarioTime"], df["A VR Pos Z"], label="A VR Pos Z")
plt.xlabel("Scenario time (s)")
plt.ylabel("Position (m)")
plt.title("A VR car/driver position components over time")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_a_vr_position_components.png", dpi=200)
plt.close()

print("Saved VR plots and CSV to:")
print(OUTPUT_DIR)

print("\nFlag counts:")
print("B VR pedestrian speed > 3 m/s:", df["ped_vr_speed_flag"].sum())
print("A VR car/driver speed > 40 m/s:", df["car_vr_speed_flag"].sum())

print("\nScenario time range:")
print("Min time:", df["ScenarioTime"].min())
print("Max time:", df["ScenarioTime"].max())

print("\nTop 10 B VR pedestrian speeds:")
print(
    df[[
        "ScenarioTime",
        "dt",
        "B VR Pos X",
        "B VR Pos Z",
        "ped_vr_step_xz",
        "ped_vr_speed_xz"
    ]]
    .sort_values("ped_vr_speed_xz", ascending=False)
    .head(10)
    .to_string()
)

print("\nTop 10 A VR car/driver speeds:")
print(
    df[[
        "ScenarioTime",
        "dt",
        "A VR Pos X",
        "A VR Pos Z",
        "car_vr_step_xz",
        "car_vr_speed_xz"
    ]]
    .sort_values("car_vr_speed_xz", ascending=False)
    .head(10)
    .to_string()
)

# 7. A VR and B VR position components together
plt.figure(figsize=(12, 6))

plt.plot(df["ScenarioTime"], df["A VR Pos X"], label="A VR Pos X")
plt.plot(df["ScenarioTime"], df["A VR Pos Z"], label="A VR Pos Z")

plt.plot(df["ScenarioTime"], df["B VR Pos X"], label="B VR Pos X")
plt.plot(df["ScenarioTime"], df["B VR Pos Z"], label="B VR Pos Z")

plt.xlabel("Scenario time (s)")
plt.ylabel("Position (m)")
plt.title("A VR and B VR position components over time")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "07_a_b_vr_position_components_together.png", dpi=200)
plt.close()