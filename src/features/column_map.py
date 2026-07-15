"""Map Unity/VR column variations to the internal analysis schema."""

from __future__ import annotations

import base64
import re
import struct

import numpy as np
import pandas as pd


COLUMN_MAP: dict[str, tuple[str, ...]] = {
    "time": ("ScenarioTime", "scenario_time", "time", "elapsed_time_sec"),
    "frame": ("Frame Number", "frame_number", "frame"),
    "ped_x": ("B VR Pos X", "b_vr_pos_x", "PedestrianPositionX", "ped_pos_x", "PlayerPosition.x", "ped_x"),
    "ped_z": ("B VR Pos Z", "b_vr_pos_z", "PedestrianPositionZ", "ped_pos_z", "PlayerPosition.z", "ped_z"),
    "ped_yaw": ("B VR Rot Y", "b_vr_rot_y", "ped_yaw", "PlayerRotation.y"),
    "ped_avatar_x": ("B Avatar Pos X", "b_avatar_pos_x", "ped_avatar_x"),
    "ped_avatar_z": ("B Avatar Pos Z", "b_avatar_pos_z", "ped_avatar_z"),
    "veh_x": ("A car Pos X", "a_car_pos_x", "VehiclePositionX", "car_x", "veh_x"),
    "veh_z": ("A car Pos Z", "a_car_pos_z", "VehiclePositionZ", "car_z", "veh_z"),
    "veh_yaw": ("A car Rot Y", "a_car_rot_y", "car_yaw", "veh_yaw"),
    "head_yaw": ("B Bone Rot Head Y", "b_bone_rot_head_y", "HeadRotationY", "head_yaw"),
    "veh_accel_raw": ("A accel", "a_accel", "veh_accel_raw"),
}

_VECTOR_SOURCES = {
    "b_vr_pos": {0: "ped_x", 2: "ped_z"},
    "b_vr_rot": {1: "ped_yaw"},
    "b_avatar_pos": {0: "ped_avatar_x", 2: "ped_avatar_z"},
    "a_car_pos": {0: "veh_x", 2: "veh_z"},
    "a_car_rot": {1: "veh_yaw"},
    "b_bone_rot_head": {1: "head_yaw"},
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add standard numeric columns while preserving all original columns."""
    output = df.copy()
    lookup = {_token(column): column for column in output.columns}
    for standard, aliases in COLUMN_MAP.items():
        if standard in output:
            continue
        source = next((lookup[_token(alias)] for alias in aliases if _token(alias) in lookup), None)
        if source is not None:
            output[standard] = output[source]
    for source_name, components in _VECTOR_SOURCES.items():
        source = lookup.get(_token(source_name))
        if source is None:
            continue
        decoded = output[source].map(_decode_vector)
        for index, standard in components.items():
            if standard not in output:
                output[standard] = decoded.map(lambda value, i=index: value[i] if value is not None and len(value) > i else np.nan)
    for standard in COLUMN_MAP:
        if standard in output:
            output[standard] = pd.to_numeric(output[standard], errors="coerce")
    return output


def _token(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _decode_vector(value: object) -> tuple[float, ...] | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.startswith("(") and text.endswith(")"):
        try:
            return tuple(float(part.strip()) for part in text[1:-1].split(","))
        except ValueError:
            return None
    try:
        raw = base64.b64decode(text, validate=True)
        if not raw or len(raw) % 4:
            return None
        return struct.unpack("<" + "f" * (len(raw) // 4), raw)
    except (ValueError, struct.error):
        return None
