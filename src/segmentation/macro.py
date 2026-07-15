"""Five-phase macro segmentation extracted from the original pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PipelineConfig


class MacroSegmenter:
    """Split a scenario into low-motion, acceleration, crossing, and deceleration phases."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ordered macro phases and their frame/time boundaries."""
        columns = ["segment_id", "start_time", "end_time", "phase_label", "start_idx", "end_idx"]
        if df.empty:
            return pd.DataFrame(columns=columns)
        if not {"time", "ped_speed"}.issubset(df.columns):
            raise ValueError("Macro segmentation requires time and ped_speed")
        work = df.reset_index(drop=True)
        time = pd.to_numeric(work["time"], errors="coerce").to_numpy(dtype=float)
        speed = pd.to_numeric(work["ped_speed"], errors="coerce").interpolate(limit_direction="both").fillna(0.0)
        trend = speed.rolling(self._window(time), center=True, min_periods=1).mean().to_numpy(dtype=float)
        if len(time) == 1 or time[-1] <= time[0]:
            label = "low_motion" if trend[0] < self.config.speed_threshold else "crossing"
            return pd.DataFrame([[0, float(time[0]), float(time[0]), label, 0, 0]], columns=columns)
        if np.nanmax(trend) < self.config.walking_speed:
            return pd.DataFrame([[0, float(time[0]), float(time[-1]), "low_motion", 0, len(work) - 1]], columns=columns)
        slope = np.gradient(trend, time)
        moving = np.flatnonzero(trend >= self.config.walking_speed)
        movement_start = int(moving[0])
        accel_candidates = np.flatnonzero((slope >= self.config.acceleration_slope) & (np.arange(len(work)) <= movement_start))
        accel_start = int(accel_candidates[0]) if len(accel_candidates) else movement_start
        search_end = int(np.searchsorted(time, time[accel_start] + self.config.macro_accel_search_seconds, side="right") - 1)
        search_end = max(accel_start, min(search_end, len(work) - 1))
        accel_end = accel_start + int(np.nanargmax(trend[accel_start:search_end + 1]))
        after_midpoint = np.arange(len(work)) >= max(accel_end, int(len(work) * self.config.final_deceleration_fraction))
        decel_candidates = np.flatnonzero(after_midpoint & (slope <= self.config.deceleration_slope))
        decel_start = int(decel_candidates[0]) if len(decel_candidates) else accel_end + int(np.nanargmax(trend[accel_end:]))
        stop_candidates = np.flatnonzero((np.arange(len(work)) > decel_start) & (trend < self.config.speed_threshold))
        stop_start = int(stop_candidates[0]) if len(stop_candidates) else len(work) - 1
        boundaries = self._unique_boundaries([0, accel_start, accel_end, decel_start, stop_start, len(work) - 1])
        canonical = ["low_motion", "acceleration", "crossing", "deceleration", "post_motion"]
        rows = []
        for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if end < start:
                continue
            label = canonical[min(index, len(canonical) - 1)]
            rows.append([len(rows), float(time[start]), float(time[end]), label, start, end])
        if not rows:
            label = "low_motion" if float(np.nanmean(trend)) < self.config.walking_speed else "crossing"
            rows.append([0, float(time[0]), float(time[-1]), label, 0, len(work) - 1])
        return pd.DataFrame(rows, columns=columns)

    def _window(self, time: np.ndarray) -> int:
        dt = float(np.nanmedian(np.diff(time))) if len(time) > 1 else np.nan
        if not np.isfinite(dt) or dt <= 0:
            return 5
        window = max(3, int(round(self.config.macro_smoothing_seconds / dt)))
        return window + 1 if window % 2 == 0 else window

    @staticmethod
    def _unique_boundaries(values: list[int]) -> list[int]:
        return sorted(set(int(value) for value in values))
