"""Robust loading and validation for flat PedNYC scenario CSV files."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


def load_scenario(csv_path: Path) -> pd.DataFrame:
    """Load a scenario CSV, normalize headers, and zero scenario time."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scenario CSV does not exist: {path}")
    attempts = ({"sep": None, "engine": "python"}, {"sep": ";"}, {"sep": ","}, {"sep": "\t"})
    best: pd.DataFrame | None = None
    for kwargs in attempts:
        try:
            candidate = pd.read_csv(path, **kwargs)
        except (OSError, UnicodeError, pd.errors.ParserError):
            continue
        if best is None or candidate.shape[1] > best.shape[1]:
            best = candidate
        if candidate.shape[1] > 1:
            break
    if best is None or best.shape[1] <= 1:
        raise ValueError(f"Could not parse a multi-column CSV: {path}")
    frame = best.copy()
    frame.columns = [_normalize_header(column) for column in frame.columns]
    if "scenario_time" in frame.columns:
        frame["scenario_time"] = pd.to_numeric(frame["scenario_time"], errors="coerce")
        first = frame["scenario_time"].dropna()
        if not first.empty:
            frame["scenario_time"] -= float(first.iloc[0])
    return frame


def validate_scenario(df: pd.DataFrame) -> dict[str, object]:
    """Return a validation report without mutating ``df``."""
    from src.features.column_map import normalize_columns

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    mapped = normalize_columns(df)
    required = ["time", "ped_x", "ped_z", "veh_x", "veh_z"]
    missing = [column for column in required if column not in mapped.columns]
    empty_required = [column for column in required if column in mapped and mapped[column].notna().sum() == 0]
    duplicate_columns = sorted({column for column in df.columns if list(df.columns).count(column) > 1})
    return {
        "valid": bool(len(df)) and not missing and not empty_required and not duplicate_columns,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "missing_required": missing,
        "empty_required": empty_required,
        "duplicate_columns": duplicate_columns,
    }


def _normalize_header(value: object) -> str:
    text = str(value).replace('"', "").replace("'", "").replace("]", "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")
