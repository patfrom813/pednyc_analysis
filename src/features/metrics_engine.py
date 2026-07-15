"""Frame-level kinematic and contextual metric computation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PipelineConfig


class MetricsEngine:
    """Compute the existing PedNYC X/Z motion metrics on standard columns."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return frame-level pedestrian, vehicle, distance, and head metrics."""
        required = {"time", "ped_x", "ped_z", "veh_x", "veh_z"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Cannot compute metrics; missing columns: {missing}")
        if df.empty:
            return df.copy()
        out = df.copy()
        out["dt"] = pd.to_numeric(out.get("dt", out["time"].diff()), errors="coerce").where(lambda s: s > 0)
        out["ped_speed"] = self._speed(out["ped_x"], out["ped_z"], out["dt"])
        out["veh_speed"] = self._speed(out["veh_x"], out["veh_z"], out["dt"])
        out["ped_speed"] = self._smooth(out["ped_speed"], self.config.smoothing_window)
        out["veh_speed"] = self._smooth(out["veh_speed"], self.config.smoothing_window)
        out["ped_acceleration"] = self._smooth(self._acceleration(out["ped_speed"], out["dt"]), self.config.acceleration_smoothing_window)
        out["veh_acceleration"] = self._smooth(self._acceleration(out["veh_speed"], out["dt"]), self.config.acceleration_smoothing_window)
        out["ped_veh_distance"] = np.hypot(out["veh_x"] - out["ped_x"], out["veh_z"] - out["ped_z"])
        out["ped_veh_distance"] = self._smooth(out["ped_veh_distance"], self.config.distance_smoothing_window)
        out["distance_change"] = out["ped_veh_distance"].diff().fillna(0.0)
        out["closing_distance"] = self._smooth(-out["distance_change"], self.config.distance_smoothing_window)
        out["closing_distance_flag"] = out["closing_distance"] > self.config.closing_threshold
        steps = np.hypot(out["ped_x"].diff(), out["ped_z"].diff()).fillna(0.0)
        out["path_length"] = steps.cumsum()
        out["displacement"] = np.hypot(out["ped_x"] - out["ped_x"].iloc[0], out["ped_z"] - out["ped_z"].iloc[0])
        out["progress_ratio"] = out["displacement"].div(out["path_length"].where(out["path_length"] > 1e-9)).fillna(0.0)
        if "head_yaw" in out:
            out["head_rotation"] = self._angular_rate(out["head_yaw"], out["dt"])
            if "ped_yaw" in out:
                out["head_body_yaw_diff"] = ((out["head_yaw"] - out["ped_yaw"] + 180.0) % 360.0) - 180.0
            else:
                out["head_body_yaw_diff"] = np.nan
        else:
            out["head_rotation"] = 0.0
            out["head_body_yaw_diff"] = np.nan
        out["stopping_indicator"] = out["ped_speed"] < self.config.speed_threshold
        out["walking_indicator"] = out["ped_speed"] >= self.config.walking_speed
        out["slowing_indicator"] = out["ped_acceleration"] <= self.config.slowing_acceleration
        out["ped_speed_xz_smooth"] = out["ped_speed"]
        out["ped_accel_xz_smooth"] = out["ped_acceleration"]
        out["car_speed_xz_smooth"] = out["veh_speed"]
        out["car_accel_xz_smooth"] = out["veh_acceleration"]
        out["car_ped_distance_xz_smooth"] = out["ped_veh_distance"]
        out["distance_closing_smooth"] = out["closing_distance"]
        out["ScenarioTime"] = out["time"]
        return out

    @staticmethod
    def _speed(x: pd.Series, z: pd.Series, dt: pd.Series) -> pd.Series:
        return np.hypot(pd.to_numeric(x, errors="coerce").diff(), pd.to_numeric(z, errors="coerce").diff()).div(dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _acceleration(speed: pd.Series, dt: pd.Series) -> pd.Series:
        return speed.diff().div(dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _smooth(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(max(1, int(window)), center=True, min_periods=1).mean()

    @staticmethod
    def _angular_rate(angle: pd.Series, dt: pd.Series) -> pd.Series:
        radians = np.unwrap(np.deg2rad(pd.to_numeric(angle, errors="coerce").interpolate(limit_direction="both")))
        rate = pd.Series(radians, index=angle.index).diff().div(dt)
        return np.rad2deg(rate).replace([np.inf, -np.inf], np.nan).fillna(0.0)
