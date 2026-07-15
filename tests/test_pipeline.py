"""Focused tests for configuration discovery and reusable pipeline stages."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from run_batch import select_scenarios
from src.config import DatasetConfig, PipelineConfig
from src.features import BehaviorClassifier, EventMerger, MetricsEngine, normalize_columns
from src.preprocess import clean_metrics
from src.segmentation import MacroSegmenter, MicroSegmenter


class PipelineTests(unittest.TestCase):
    """Exercise discovery and the complete in-memory analysis chain."""

    def test_dataset_discovery_and_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            csv_dir = Path(temp) / "csv_pednyc2" / "csv"
            csv_dir.mkdir(parents=True)
            for name in ("CSV_Scenario-Ped-12_Session-x.csv", "CSV_Scenario-Ped-3_Session-x.csv", "CSV_Scenario-Practice_Session-x.csv"):
                (csv_dir / name).touch()
            dataset = DatasetConfig(Path(temp))
            self.assertEqual(dataset.discover_participants(), [2])
            self.assertEqual(dataset.discover_scenarios(2), ["3", "12", "Practice"])
            self.assertEqual(dataset.get_scenario_csv(2, "3").name, "CSV_Scenario-Ped-3_Session-x.csv")

    def test_in_memory_pipeline_stages(self) -> None:
        time = np.arange(0.0, 8.0, 0.1)
        ped_x = np.where(time < 2.0, 0.0, np.minimum((time - 2.0) * 0.8, 3.2))
        raw = pd.DataFrame({"scenario_time": time, "ped_pos_x": ped_x, "ped_pos_z": 0.0, "car_x": 6.0 - time * 0.2, "car_z": 1.0, "head_yaw": 0.0})
        config = PipelineConfig()
        normalized = normalize_columns(raw)
        cleaned = clean_metrics(normalized, config)
        metrics = MetricsEngine(config).compute(cleaned)
        macro = MacroSegmenter(config).segment(metrics)
        windows = MicroSegmenter(config).extract_windows(metrics, macro)
        classified = BehaviorClassifier(config).classify(windows)
        events = EventMerger(config).merge(classified)
        self.assertFalse(macro.empty)
        self.assertFalse(windows.empty)
        self.assertIn("predicted_label", classified)
        self.assertEqual(list(events.columns), ["event_id", "start_time", "end_time", "label", "duration", "confidence"])

    def test_batch_pattern_selection(self) -> None:
        args = Namespace(scenarios=None, count=None, pattern="1*", all=False)
        self.assertEqual(select_scenarios(["3", "12", "101", "Practice"], args), ["12", "101"])

    def test_yaml_round_trip_without_optional_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pipeline.yaml"
            expected = PipelineConfig(window_step=0.25, time_start_zero=False)
            expected.to_yaml(path)
            actual = PipelineConfig.from_yaml(path)
            self.assertEqual(actual, expected)

    def test_all_low_motion_is_one_macro_segment(self) -> None:
        frame = pd.DataFrame({"time": [0.0, 0.5, 1.0], "ped_speed": [0.0, 0.01, 0.0]})
        segments = MacroSegmenter(PipelineConfig()).segment(frame)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments.iloc[0]["phase_label"], "low_motion")

    def test_cleaning_preserves_encoded_source_columns(self) -> None:
        frame = pd.DataFrame({"time": [0.0, 1.0], "ped_x": [0.0, 0.1], "ped_z": [0.0, 0.0], "encoded_source": ["abc", "def"]})
        cleaned = clean_metrics(frame, PipelineConfig())
        self.assertEqual(cleaned["encoded_source"].tolist(), ["abc", "def"])


if __name__ == "__main__":
    unittest.main()
