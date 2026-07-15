"""Merge overlapping or nearby classified behavior windows."""

from __future__ import annotations

import pandas as pd

from src.config import PipelineConfig


class EventMerger:
    """Collapse classified windows into duration-filtered events."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def merge(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """Merge same-label overlaps and configured gaps into event rows."""
        columns = ["event_id", "start_time", "end_time", "label", "duration", "confidence"]
        if df_events.empty:
            return pd.DataFrame(columns=columns)
        start_col = "start_time" if "start_time" in df_events else "window_start"
        end_col = "end_time" if "end_time" in df_events else "window_end"
        label_col = "label" if "label" in df_events else "predicted_label"
        work = df_events.sort_values([start_col, end_col]).reset_index(drop=True)
        merged: list[dict[str, float | str]] = []
        gap = self.config.merge_gap_threshold
        for _, row in work.iterrows():
            item = {
                "start_time": float(row[start_col]),
                "end_time": float(row[end_col]),
                "label": str(row[label_col]),
                "confidence": float(row.get("confidence", self.config.confidence_default)),
            }
            if merged and item["label"] == merged[-1]["label"] and item["start_time"] - float(merged[-1]["end_time"]) <= gap:
                merged[-1]["end_time"] = max(float(merged[-1]["end_time"]), float(item["end_time"]))
                merged[-1]["confidence"] = max(float(merged[-1]["confidence"]), float(item["confidence"]))
            else:
                merged.append(item)
        output = pd.DataFrame(merged)
        output["duration"] = output["end_time"] - output["start_time"]
        output = output[output["duration"] >= self.config.min_event_duration].reset_index(drop=True)
        output.insert(0, "event_id", range(len(output)))
        return output[columns]
