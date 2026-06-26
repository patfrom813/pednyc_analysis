import pandas as pd
from pathlib import Path

input_file = Path(r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis\decoded_clean_PedNYC1_scenario3.csv")

df = pd.read_csv(input_file)



# file = Path(r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis\CSV_Scenario-Ped-3_Session-temp_2024-02-22-13-48-59.csv")
# df1 = pd.read_csv(file, sep=";")
# print(df1.shape)

# df.columns = (
#     df.columns
#     .str.strip()
#     .str.strip('"')
#     .str.strip()
# )
# print([repr(col) for col in df.columns if "velocity" in col.lower()])
# vel = df["A velocity"].str.strip('"()').str.split(",", expand=True)


# print([repr(col) for col in df.columns if "time" in col.lower()])
# df.columns = df.columns.str.replace("'", "", regex=False)
# df.columns = df.columns.str.strip()

df = pd.read_csv(input_file, index_col=False)

df.columns = df.columns.map(lambda x: str(x).replace('"', '').replace("'", "").strip())

# cols = [
#     "A car Pos X",
#     "A car Pos Y",
#     "A car Pos Z",
#     "B Avatar Pos X",
#     "B Avatar Pos Y",
#     "B Avatar Pos Z"
# ]

# print(df[cols].describe())




# print([col for col in df.columns if "speed" in col.lower()])
# print([col for col in df.columns if "velocity" in col.lower()])
import numpy as np
import pandas as pd

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