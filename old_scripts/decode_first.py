import pandas as pd
import base64
import struct
from pathlib import Path

INPUT_FILE = Path(r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis\CSV_Scenario-Ped-3_Session-temp_2024-02-22-13-48-59.csv")
OUTPUT_FILE = Path(r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis\decoded_clean_PedNYC1_scenario3.csv")

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