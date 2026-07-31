import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.visualization.interactive_onset_speed import (
    SCENARIO_LABELS,
    ScenarioSummary,
    build_figure,
    calculate_summary,
    export_scenario,
)


class InteractiveOnsetSpeedTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for pid, end, speed in ((1, 1.0, 1.0), (2, 2.0, 2.0), (3, 2.0, 3.0)):
            for time in np.arange(0, end + 0.01, 0.1):
                rows.append({
                    "pednyc_number": pid,
                    "scenario_number": 3,
                    "recording_identifier": f"PedNYC{pid}_scenario3",
                    "aligned_time": time,
                    "pedestrian_speed": speed,
                })
        self.traces = pd.DataFrame(rows)

    def test_summary_is_median_iqr_and_suppressed_below_cutoff(self) -> None:
        summary, metadata = calculate_summary(self.traces, 0.75)
        at_zero = summary[np.isclose(summary.aligned_time, 0)].iloc[0]
        self.assertEqual(metadata.maximum_n, 3)
        self.assertEqual(metadata.minimum_summary_n, 3)
        self.assertEqual(at_zero.median_speed, 2)
        self.assertEqual(at_zero.q1_speed, 1.5)
        self.assertEqual(at_zero.q3_speed, 2.5)
        after_short = summary[np.isclose(summary.aligned_time, 1.5)].iloc[0]
        self.assertEqual(after_short.contributing_recordings, 2)
        self.assertTrue(np.isnan(after_short.median_speed))

    def test_export_keeps_full_data_and_embeds_click_handler(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "scenario_03_onset_aligned.html"
            metadata = export_scenario(self.traces, output, 0.5)
            html = output.read_text(encoding="utf-8")
            self.assertEqual(metadata.minimum_end_time, 1.0)
            self.assertEqual(metadata.maximum_end_time, 2.0)
            self.assertIn("data-click-highlight-ready", html)
            self.assertIn("plotly_doubleclick", html)
            self.assertIn('"participant_id":"PedNYC1"', html)
            self.assertIn("Contributing recordings", html)
            self.assertIn("Orthogonal approach", html)
            self.assertIn("Scenario 3", html)
            self.assertIn("Onset-aligned pedestrian speed", html)

    def test_all_default_scenarios_have_centralized_labels(self) -> None:
        self.assertEqual(set(SCENARIO_LABELS), {3, 7, 12, 15, 16, 21})

    def test_missing_scenario_label_raises_clear_error(self) -> None:
        metadata = ScenarioSummary(99, 3, 2, 0.5, 3, 1.0, 2.0, 1)
        summary = pd.DataFrame({
            "aligned_time": [0.0],
            "median_speed": [1.0],
            "q1_speed": [0.5],
            "q3_speed": [1.5],
            "contributing_recordings": [3],
            "summary_visible": [True],
        })
        traces = self.traces.assign(scenario_number=99)
        with self.assertRaisesRegex(ValueError, "No descriptive condition label configured"):
            build_figure(traces, summary, metadata)


if __name__ == "__main__":
    unittest.main()
