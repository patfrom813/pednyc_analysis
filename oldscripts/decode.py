import pandas as pd
import numpy as np
from pathlib import Path

input_file = Path(r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis\decoded_clean_PedNYC1_scenario3.csv")

df = pd.read_csv(input_file, sep=";", index_col=False)

# clean column names
df.columns = (
    df.columns
    .astype(str)
    .str.replace('"', '', regex=False)
    .str.replace("'", "", regex=False)
    .str.strip()
)

print("First 15 columns:")
print(df.columns[:15].tolist())

required_cols = [
    "ScenarioTime",
    "B Avatar Pos X",
    "B Avatar Pos Z"
]

missing = [col for col in required_cols if col not in df.columns]

if missing:
    print("Missing columns:", missing)
    print("Available columns containing Scenario:")
    print([c for c in df.columns if "Scenario" in c])
    print("Available columns containing Avatar:")
    print([c for c in df.columns if "Avatar" in c])
    raise SystemExit("Fix column names or delimiter before calculating speed.")

# convert to numbers
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# calculate dt
df["dt"] = df["ScenarioTime"].diff()

# calculate XZ movement
df["ped_step"] = np.sqrt(
    df["B Avatar Pos X"].diff()**2 +
    df["B Avatar Pos Z"].diff()**2
)

# calculate speed
df["pedestrian_speed_xz"] = df["ped_step"] / df["dt"]

# remove invalid first row / bad dt rows
valid = df[
    df["dt"].notna() &
    (df["dt"] > 0) &
    df["pedestrian_speed_xz"].notna()
].copy()

# top 10 speeds
print("\nTop 10 pedestrian speeds:")
print(
    valid[[
        "ScenarioTime",
        "dt",
        "B Avatar Pos X",
        "B Avatar Pos Z",
        "ped_step",
        "pedestrian_speed_xz"
    ]]
    .sort_values("pedestrian_speed_xz", ascending=False)
    .head(10)
    .to_string()
)

# rows around maximum spike
bad_i = valid["pedestrian_speed_xz"].idxmax()

print("\nRows around max speed:")
print(df.loc[bad_i-3:bad_i+3, [
    "ScenarioTime",
    "dt",
    "B Avatar Pos X",
    "B Avatar Pos Z",
    "ped_step",
    "pedestrian_speed_xz"
]].to_string())
