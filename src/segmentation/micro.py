"""Adaptive sliding-window feature extraction within macro phases."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from src.config import PipelineConfig


class MicroSegmenter:
    """Extract behavior features from windows constrained to macro segments."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def extract_windows(self, df: pd.DataFrame, macro_segments: pd.DataFrame) -> pd.DataFrame:
        """Return adaptive window features for every macro phase."""
        if df.empty or macro_segments.empty:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        for _, macro in macro_segments.iterrows():
            start_idx = int(macro.get("start_idx", 0))
            end_idx = int(macro.get("end_idx", len(df) - 1))
            segment = df.iloc[start_idx:end_idx + 1]
            if segment.empty:
                continue
            start_time = float(segment["time"].iloc[0])
            end_time = float(segment["time"].iloc[-1])
            duration = max(0.0, end_time - start_time)
            window = self._adaptive_size(duration)
            starts = [start_time] if duration <= window else list(np.arange(start_time, end_time, self.config.window_step))
            for window_start in starts:
                window_end = min(window_start + window, end_time)
                sample = segment[(segment["time"] >= window_start) & (segment["time"] <= window_end)]
                if len(sample) < 2:
                    continue
                feature = self._features(sample)
                feature.update({
                    "window_id": len(rows),
                    "macro_segment_id": int(macro["segment_id"]),
                    "macro_phase": str(macro["phase_label"]),
                    "window_start": float(sample["time"].iloc[0]),
                    "window_end": float(sample["time"].iloc[-1]),
                    "duration": float(sample["time"].iloc[-1] - sample["time"].iloc[0]),
                })
                rows.append(feature)
                if window_end >= end_time:
                    break
        return pd.DataFrame(rows)

    def _adaptive_size(self, duration: float) -> float:
        if duration <= 0:
            return self.config.window_min
        count = max(1, math.ceil(duration / self.config.window_size))
        return float(np.clip(duration / count, self.config.window_min, self.config.window_max))

    def _features(self, sample: pd.DataFrame) -> dict[str, object]:
        time = sample["time"].to_numpy(dtype=float)
        speed = self._values(sample, "ped_speed")
        accel = self._values(sample, "ped_acceleration")
        veh_distance = self._values(sample, "ped_veh_distance", np.nan)
        veh_accel = self._values(sample, "veh_acceleration", np.nan)
        closing = self._values(sample, "closing_distance", np.nan)
        head = self._values(sample, "head_rotation", 0.0)
        yaw = self._values(sample, "head_body_yaw_diff", 0.0)
        dx = np.diff(self._values(sample, "ped_x"))
        dz = np.diff(self._values(sample, "ped_z"))
        path_length = float(np.nansum(np.hypot(dx, dz)))
        displacement = float(np.hypot(sample["ped_x"].iloc[-1] - sample["ped_x"].iloc[0], sample["ped_z"].iloc[-1] - sample["ped_z"].iloc[0]))
        signs = np.sign(accel)
        reversals = int(np.sum((signs[1:] * signs[:-1]) < 0))
        slope = self._slope(time, speed)
        return {
            "mean_speed": float(np.nanmean(speed)),
            "changing_speed": slope,
            "mean_acceleration": float(np.nanmean(accel)),
            "reversals": reversals,
            "displacement": displacement,
            "path_length": path_length,
            "progress_ratio": displacement / path_length if path_length > 1e-9 else 0.0,
            "vehicle_distance": float(np.nanmin(veh_distance)) if np.any(np.isfinite(veh_distance)) else np.nan,
            "vehicle_approaching": bool(np.nanmean(closing) > self.config.closing_threshold) if np.any(np.isfinite(closing)) else False,
            "vehicle_braking": bool(np.nanmean(veh_accel) < self.config.slowing_acceleration) if np.any(np.isfinite(veh_accel)) else False,
            "head_turn_rate": float(np.nanmax(np.abs(head))) if len(head) else 0.0,
            "head_body_yaw_diff": float(np.nanmax(np.abs(yaw))) if np.any(np.isfinite(yaw)) else 0.0,
        }

    @staticmethod
    def _values(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
        if column not in frame:
            return np.full(len(frame), default, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)

    @staticmethod
    def _slope(time: np.ndarray, values: np.ndarray) -> float:
        valid = np.isfinite(time) & np.isfinite(values)
        if valid.sum() < 2 or np.ptp(time[valid]) <= 0:
            return 0.0
        return float(np.polyfit(time[valid] - time[valid][0], values[valid], 1)[0])
