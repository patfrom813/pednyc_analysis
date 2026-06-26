import pandas as pd

FILE = r"C:\Users\patl5\OneDrive\Desktop\BURE\psl64_analysis\decoded_clean_PedNYC1_scenario3.csv"

df = pd.read_csv(FILE, sep=";")
df.columns = df.columns.str.strip()

# find A velocity columns
velocity_cols = [c for c in df.columns if "A velocity" in c]
print("A velocity columns:")
print(velocity_cols)

# show 3 columns left and 3 columns right around A velocity
first_velocity_col = velocity_cols[0]
idx = df.columns.get_loc(first_velocity_col)

start = max(0, idx - 3)
end = min(len(df.columns), idx + len(velocity_cols) + 3)

check_cols = df.columns[start:end]

print("\nColumns being checked:")
print(list(check_cols))

print("\nFirst 5 rows around A velocity:")
print(df[check_cols].head())

print("\nData types:")
print(df[check_cols].dtypes)
print(df["A velocity"].head())