"""Rule-based behavior classification for micro windows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PipelineConfig


class BehaviorClassifier:
    """Assign one documented behavior label using multiple window features."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def classify(self, df_windows: pd.DataFrame) -> pd.DataFrame:
        """Add ``predicted_label`` and confidence to extracted windows."""
        if df_windows.empty:
            return df_windows.assign(predicted_label=pd.Series(dtype=str), confidence=pd.Series(dtype=float))
        rows = df_windows.copy()
        results = rows.apply(self._classify_row, axis=1, result_type="expand")
        results.columns = ["predicted_label", "confidence"]
        rows[["predicted_label", "confidence"]] = results
        return rows

    def _classify_row(self, row: pd.Series) -> tuple[str, float]:
        cfg = self.config
        speed = self._number(row, "mean_speed")
        slope = self._number(row, "changing_speed")
        acceleration = self._number(row, "mean_acceleration")
        head_rate = abs(self._number(row, "head_turn_rate"))
        yaw_diff = abs(self._number(row, "head_body_yaw_diff"))
        distance = self._number(row, "vehicle_distance", np.inf)
        reversals = self._number(row, "reversals")
        progress = self._number(row, "progress_ratio", 1.0)
        duration = self._number(row, "duration")
        approaching = bool(row.get("vehicle_approaching", False))
        braking = bool(row.get("vehicle_braking", False))
        if (head_rate >= cfg.head_check_turn_rate and duration <= cfg.head_check_max_duration) or yaw_diff >= cfg.head_check_yaw:
            return "head_check", min(1.0, 0.6 + 0.4 * max(head_rate / max(cfg.head_check_turn_rate, 1e-9), yaw_diff / max(cfg.head_check_yaw, 1e-9)))
        if distance <= cfg.yielding_distance and approaching and (braking or slope < -cfg.acceleration_slope or speed < cfg.walking_speed):
            return "possible_yielding", 0.85
        if reversals >= cfg.hesitation_min_reversals and progress <= cfg.hesitation_max_progress and speed > cfg.speed_threshold:
            return "possible_hesitation", 0.82
        if speed < cfg.speed_threshold:
            return "low_motion", 0.90
        if slope >= cfg.acceleration_slope or acceleration >= cfg.starting_acceleration:
            return "acceleration", 0.80
        if slope <= cfg.deceleration_slope or acceleration <= cfg.slowing_acceleration:
            return "deceleration", 0.80
        if progress >= cfg.hesitation_max_progress and speed >= cfg.behavior_walking_speed and duration >= cfg.window_min:
            return "crossing_commitment", 0.78
        return "low_motion", cfg.confidence_default

    @staticmethod
    def _number(row: pd.Series, name: str, default: float = 0.0) -> float:
        value = row.get(name, default)
        try:
            return float(value) if np.isfinite(float(value)) else default
        except (TypeError, ValueError):
            return default
