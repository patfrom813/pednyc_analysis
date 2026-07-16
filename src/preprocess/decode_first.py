import base64
import argparse
import struct
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args():
    parser = argparse.ArgumentParser(description="Decode Unity float arrays without changing row-level data.")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


ARGS = parse_args()
if ARGS.input_csv is None:
    raw_matches = sorted(RAW_DIR.glob("CSV_Scenario-Ped-3_Session-temp_*.csv"))
    if not raw_matches:
        raise FileNotFoundError(f"No default raw scenario CSV found in {RAW_DIR}")
    INPUT_FILE = raw_matches[0]
else:
    INPUT_FILE = ARGS.input_csv
OUTPUT_FILE = ARGS.output_csv or (PROCESSED_DIR / "decoded_clean_PedNYC1_scenario3.csv")


def decode_float_array(value):
    if pd.isna(value):
        return None

    try:
        raw = base64.b64decode(str(value), validate=True)

        if len(raw) % 4 != 0:
            return None

        count = len(raw) // 4
        return struct.unpack("<" + "f" * count, raw)

    except Exception:
        return None


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Read original file
df = pd.read_csv(INPUT_FILE, sep=";")
df.columns = df.columns.str.strip().str.replace("]", "", regex=False)

print("Original shape:", df.shape)
print("Original rows:", len(df))
print("Original ScenarioTime max:", df["ScenarioTime"].max())
print("Original ScenarioTime tail:")
print(df["ScenarioTime"].tail(10))

clean_df = pd.DataFrame(index=df.index)

for col in df.columns:
    sample = df[col].dropna().head(20)

    decoded_values = []

    for v in sample:
        decoded = decode_float_array(v)
        if isinstance(decoded, tuple):
            decoded_values.append(decoded)

    # Not a decoded/base64 column
    if len(decoded_values) == 0:
        clean_df[col] = df[col]
        continue

    most_common_len = pd.Series([len(v) for v in decoded_values]).mode().iloc[0]
    decoded_col = df[col].apply(decode_float_array)

    if most_common_len == 3:
        clean_df[col + " X"] = decoded_col.apply(
            lambda x: x[0] if isinstance(x, tuple) and len(x) >= 3 else None
        )
        clean_df[col + " Y"] = decoded_col.apply(
            lambda x: x[1] if isinstance(x, tuple) and len(x) >= 3 else None
        )
        clean_df[col + " Z"] = decoded_col.apply(
            lambda x: x[2] if isinstance(x, tuple) and len(x) >= 3 else None
        )

    elif most_common_len == 4:
        clean_df[col + " X"] = decoded_col.apply(
            lambda x: x[0] if isinstance(x, tuple) and len(x) >= 4 else None
        )
        clean_df[col + " Y"] = decoded_col.apply(
            lambda x: x[1] if isinstance(x, tuple) and len(x) >= 4 else None
        )
        clean_df[col + " Z"] = decoded_col.apply(
            lambda x: x[2] if isinstance(x, tuple) and len(x) >= 4 else None
        )
        clean_df[col + " W"] = decoded_col.apply(
            lambda x: x[3] if isinstance(x, tuple) and len(x) >= 4 else None
        )

    else:
        for i in range(most_common_len):
            clean_df[f"{col} {i}"] = decoded_col.apply(
                lambda x: x[i] if isinstance(x, tuple) and len(x) > i else None
            )

# Save
clean_df.to_csv(OUTPUT_FILE, sep=";", index=False)

print("\nDecoded clean shape:", clean_df.shape)
print("Decoded clean rows:", len(clean_df))
print("Decoded ScenarioTime max:", clean_df["ScenarioTime"].max())
print("Decoded ScenarioTime tail:")
print(clean_df["ScenarioTime"].tail(10))

print("\nRow count same?", len(df) == len(clean_df))
print("Saved to:", OUTPUT_FILE)
