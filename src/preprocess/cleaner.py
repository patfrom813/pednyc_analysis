"""Configurable cleaning for normalized PedNYC metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PipelineConfig


def clean_metrics(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Remove invalid samples and position spikes using configured thresholds."""
    if df.empty:
        return df.copy()
    clean = df.copy()
    if "time" not in clean:
        raise ValueError("Normalized data must contain a time column")
    clean["time"] = pd.to_numeric(clean["time"], errors="coerce")
    clean = clean.dropna(subset=["time"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if clean.empty:
        raise ValueError("Scenario contains no valid time samples")
    if config.time_start_zero:
        clean["time"] -= float(clean["time"].iloc[0])
    dt = clean["time"].diff()
    clean["dt"] = dt.where(dt > 0)
    for prefix in ("ped", "veh"):
        x_col, z_col = f"{prefix}_x", f"{prefix}_z"
        if not {x_col, z_col}.issubset(clean.columns):
            continue
        x = pd.to_numeric(clean[x_col], errors="coerce")
        z = pd.to_numeric(clean[z_col], errors="coerce")
        speed = np.sqrt(x.diff().pow(2) + z.diff().pow(2)).div(clean["dt"])
        spikes = speed.gt(config.speed_spike_threshold)
        x = x.mask(spikes).interpolate(limit_direction="both")
        z = z.mask(spikes).interpolate(limit_direction="both")
        clean[x_col], clean[z_col] = x, z
    numeric = {
        "dt", "frame", "ped_x", "ped_z", "ped_yaw", "ped_avatar_x",
        "ped_avatar_z", "veh_x", "veh_z", "veh_yaw", "head_yaw",
        "veh_accel_raw",
    }
    for column in numeric.intersection(clean.columns):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    return clean
