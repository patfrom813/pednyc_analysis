from pathlib import Path
import csv
from collections import Counter

DATA_DIR = Path("/export/CAR_WORK/Goedicke_PedestrianXC/Studies")
OUT_FILE = Path.home() / "psl64_analysis" / "csv_dimensions_report.csv"

rows = []

for path in sorted(DATA_DIR.rglob("*.csv")):
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader)
            n_cols = len(header)
            n_rows = sum(1 for _ in reader) + 1

        participant = path.parts[path.parts.index("Studies") + 1]
        rows.append([str(path), participant, path.name, n_rows, n_cols, "OK"])

    except Exception as e:
        rows.append([str(path), "", path.name, "", "", f"ERROR: {e}"])

with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["full_path", "participant", "file_name", "rows_including_header", "columns", "status"])
    writer.writerows(rows)

col_counts = Counter(r[4] for r in rows if r[4] != "")
row_counts = Counter(r[3] for r in rows if r[3] != "")

print(f"Checked {len(rows)} CSV files")
print(f"Report saved to: {OUT_FILE}")

print("\nColumn count summary:")
for cols, count in sorted(col_counts.items()):
    print(f"{cols} columns: {count} files")

print("\nSmallest row counts:")
for row_count, count in sorted(row_counts.items())[:10]:
    print(f"{row_count} rows: {count} files")

print("\nFiles with non-342 columns:")
for r in rows:
    if r[4] != "" and r[4] != 342:
        print(f"{r[4]} cols | {r[3]} rows | {r[0]}")