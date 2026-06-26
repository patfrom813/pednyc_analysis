from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path(r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis")

INPUT_FILE = BASE_DIR / "decoded_clean_PedNYC1_scenario3.csv"

OUTPUT_DIR = BASE_DIR / "scenario3_outputs"
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

# Columns needed from current decoded file
needed_cols = [
    "ScenarioTime",
    "A car Pos X", "A car Pos Z",
    "B Avatar Pos X", "B Avatar Pos Z"
]

missing = [col for col in needed_cols if col not in df.columns]
if missing:
    print("Missing columns:", missing)
    print("Available columns:")
    print(df.columns.tolist())
    raise SystemExit("Column names or delimiter are wrong.")

# Convert to numeric
for col in needed_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Time difference
df["dt"] = df["ScenarioTime"].diff()

# Car speed from position change
df["car_step_xz"] = np.sqrt(
    df["A car Pos X"].diff()**2 +
    df["A car Pos Z"].diff()**2
)
df["car_speed_xz"] = df["car_step_xz"] / df["dt"]

# Pedestrian speed from position change
df["ped_step_xz"] = np.sqrt(
    df["B Avatar Pos X"].diff()**2 +
    df["B Avatar Pos Z"].diff()**2
)
df["ped_speed_xz"] = df["ped_step_xz"] / df["dt"]

# Car-pedestrian distance
df["car_ped_distance_xz"] = np.sqrt(
    (df["A car Pos X"] - df["B Avatar Pos X"])**2 +
    (df["A car Pos Z"] - df["B Avatar Pos Z"])**2
)

# Flags
df["ped_speed_flag"] = df["ped_speed_xz"] > 3
df["car_speed_flag"] = df["car_speed_xz"] > 40

# Save analysis-ready CSV
df.to_csv(OUTPUT_DIR / "analysis_ready_scenario3.csv", sep=";", index=False)

# 1. Path plot
plt.figure(figsize=(7, 7))
plt.plot(df["A car Pos X"], df["A car Pos Z"], label="Car path")
plt.plot(df["B Avatar Pos X"], df["B Avatar Pos Z"], label="Pedestrian avatar path")
plt.xlabel("X position (m)")
plt.ylabel("Z position (m)")
plt.title("Car and pedestrian movement path")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_path_plot.png", dpi=200)
plt.close()

# 2. Distance plot
plt.figure(figsize=(10, 5))
plt.plot(df["ScenarioTime"], df["car_ped_distance_xz"])
plt.xlabel("Scenario time (s)")
plt.ylabel("Distance XZ (m)")
plt.title("Car-pedestrian distance over time")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_distance_plot.png", dpi=200)
plt.close()

# 3. Car speed plot
plt.figure(figsize=(10, 5))
plt.plot(df["ScenarioTime"], df["car_speed_xz"])
plt.xlabel("Scenario time (s)")
plt.ylabel("Car speed XZ (m/s)")
plt.title("Car speed over time")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_car_speed_plot.png", dpi=200)
plt.close()

# 4. Pedestrian speed plot
plt.figure(figsize=(10, 5))
plt.plot(df["ScenarioTime"], df["ped_speed_xz"], label="Pedestrian speed")
plt.axhline(3, linestyle="--", label="3 m/s threshold")
plt.xlabel("Scenario time (s)")
plt.ylabel("Pedestrian speed XZ (m/s)")
plt.title("Pedestrian avatar speed over time")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_pedestrian_speed_plot.png", dpi=200)
plt.close()

# 5. Pedestrian position components
plt.figure(figsize=(10, 5))
plt.plot(df["ScenarioTime"], df["B Avatar Pos X"], label="B Avatar Pos X")
plt.plot(df["ScenarioTime"], df["B Avatar Pos Z"], label="B Avatar Pos Z")
plt.xlabel("Scenario time (s)")
plt.ylabel("Position (m)")
plt.title("Pedestrian avatar position components over time")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_pedestrian_position_components.png", dpi=200)
plt.close()

# 6. Car position components
plt.figure(figsize=(10, 5))
plt.plot(df["ScenarioTime"], df["A car Pos X"], label="A car Pos X")
plt.plot(df["ScenarioTime"], df["A car Pos Z"], label="A car Pos Z")
plt.xlabel("Scenario time (s)")
plt.ylabel("Position (m)")
plt.title("Car position components over time")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_car_position_components.png", dpi=200)
plt.close()

print("Saved plots and CSV to:")
print(OUTPUT_DIR)

print("\nFlag counts:")
print("Ped speed > 3 m/s:", df["ped_speed_flag"].sum())
print("Car speed > 40 m/s:", df["car_speed_flag"].sum())

print("\nTop 10 pedestrian speeds:")
print(
    df[[
        "ScenarioTime",
        "dt",
        "B Avatar Pos X",
        "B Avatar Pos Z",
        "ped_step_xz",
        "ped_speed_xz"
    ]]
    .sort_values("ped_speed_xz", ascending=False)
    .head(10)
    .to_string()
)