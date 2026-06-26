from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
input_file = PROJECT_ROOT / "data" / "processed" / "decoded_clean_PedNYC1_scenario3.csv"

df = pd.read_csv(input_file, index_col=False)

df.columns = df.columns.map(lambda x: str(x).replace('"', '').replace("'", "").strip())

# Make sure these columns are real numbers
numeric_cols = [
    "ScenarioTime",
    "B Avatar Pos X",
    "B Avatar Pos Z"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Calculate time difference
df["dt"] = df["ScenarioTime"].diff()

# Calculate pedestrian movement between frames
df["ped_step"] = np.sqrt(
    df["B Avatar Pos X"].diff()**2 +
    df["B Avatar Pos Z"].diff()**2
)

# Calculate pedestrian speed
df["pedestrian_speed_xz"] = df["ped_step"] / df["dt"]

# Find biggest speed spike
bad_i = df["pedestrian_speed_xz"].idxmax()

print(df.loc[bad_i-3:bad_i+3, [
    "ScenarioTime",
    "dt",
    "B Avatar Pos X",
    "B Avatar Pos Z",
    "ped_step",
    "pedestrian_speed_xz"
]])
bad_i = df["pedestrian_speed_xz"].idxmax()

print(df.loc[bad_i-3:bad_i+3, [
    "ScenarioTime",
    "dt",
    "B Avatar Pos X",
    "B Avatar Pos Z",
    "ped_step",
    "pedestrian_speed_xz"
]])
