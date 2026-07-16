"""Behavior-protecting tests for canonical scenario-agnostic orchestration."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.scenario_pipeline.run_scenario import resolve_raw_inputs
from src.scenario_pipeline.runner import (
    EXPECTED_SCENARIO3_BOUNDARIES,
    PipelineSanityError,
    ScenarioRunner,
    assert_fresh_metrics_input,
    compare_scenario3_regression,
    macro_midpoints,
    read_raw_csv,
    reconstruct_elapsed_time,
    validate_nonempty_csvs,
)
from src.scenario_pipeline.scenario import ScenarioPaths, discover_raw_scenarios, parse_scenario_number


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScenarioPipelineTests(unittest.TestCase):
    def test_scenario_id_parsing_and_rejection(self) -> None:
        self.assertEqual(parse_scenario_number("CSV_Scenario-Ped-3_Session-temp_x.csv"), 3)
        self.assertEqual(parse_scenario_number("CSV_Scenario-Ped-127_Session-study.csv"), 127)
        with self.assertRaises(ValueError):
            parse_scenario_number("scenario3.csv")

    def test_scenario_paths_use_active_stem_everywhere(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "CSV_Scenario-Ped-12_Session-x.csv"
            paths = ScenarioPaths.from_raw(raw, PROJECT_ROOT, output_root=Path(temp), participant=2)
            self.assertEqual(paths.scenario_stem, "PedNYC2_scenario12")
            for path in (*paths.core_csvs()[:-1], *paths.expected_plots(), paths.gantt_mp4, paths.report_md):
                self.assertIn("PedNYC2_scenario12", path.name)
            self.assertIn("scenario12", str(paths.graph_root))

    def test_discovery_and_scenario_id_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True)
            for name in ("CSV_Scenario-Ped-12_Session-x.csv", "CSV_Scenario-Ped-3_Session-x.csv", "ignore.csv"):
                (raw_dir / name).touch()
            self.assertEqual([parse_scenario_number(p) for p in discover_raw_scenarios(raw_dir)], [3, 12])
            args = Namespace(all=False, scenario_id=12, raw_path=None, raw_csv=None)
            self.assertEqual(resolve_raw_inputs(args, root)[0].name, "CSV_Scenario-Ped-12_Session-x.csv")

    def test_invalid_one_column_raw_csv_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "CSV_Scenario-Ped-9_Session-x.csv"
            path.write_text("only_one_column\nvalue\n", encoding="utf-8")
            with self.assertRaises(PipelineSanityError):
                read_raw_csv(path)

    def test_time_source_selection(self) -> None:
        seconds = pd.DataFrame({"ScenarioTime": [0.0, 0.1, 0.2], "dt": [0.1, 0.1, 0.1]})
        elapsed, source = reconstruct_elapsed_time(seconds)
        self.assertEqual(source, "ScenarioTime_seconds")
        np.testing.assert_allclose(elapsed, [0.0, 0.1, 0.2])
        frame_time = pd.DataFrame({"ScenarioTime": [1000, 1001, 1002], "dt": [np.nan, 0.05, 0.05]})
        elapsed, source = reconstruct_elapsed_time(frame_time)
        self.assertEqual(source, "cumulative_dt_seconds")
        np.testing.assert_allclose(elapsed, [0.0, 0.05, 0.10])

    def test_stale_metrics_protection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = ScenarioPaths.from_raw(
                Path(temp) / "CSV_Scenario-Ped-3_Session-x.csv",
                PROJECT_ROOT,
                output_root=Path(temp),
            )
            with self.assertRaises(PipelineSanityError):
                assert_fresh_metrics_input(paths, pd.DataFrame({"ScenarioTime": [0.0]}))
            fresh = pd.DataFrame({
                "dt": [0.1],
                "ped_speed_xz_smooth": [0.0],
                "ped_accel_xz_smooth": [0.0],
                "movement_state_v3": ["near_stopped"],
            })
            assert_fresh_metrics_input(paths, fresh)

    def test_artifact_completeness_rejects_header_only_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            valid = Path(temp) / "valid.csv"
            invalid = Path(temp) / "invalid.csv"
            valid.write_text("value\n1\n", encoding="utf-8")
            invalid.write_text("value\n", encoding="utf-8")
            validate_nonempty_csvs([valid])
            with self.assertRaises(PipelineSanityError):
                validate_nonempty_csvs([invalid])

    def test_missing_ffmpeg_plans_one_preview_per_macro(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = ScenarioPaths.from_raw(
                root / "CSV_Scenario-Ped-8_Session-x.csv",
                PROJECT_ROOT,
                output_root=root,
            )
            paths.macro_csv.parent.mkdir(parents=True)
            pd.DataFrame({
                "segment_id": [0, 1],
                "time_start_sec": [0.0, 2.0],
                "time_end_sec": [2.0, 6.0],
            }).to_csv(paths.macro_csv, index=False)
            runner = ScenarioRunner(paths)
            calls: list[list[str]] = []
            with patch("src.scenario_pipeline.runner.find_ffmpeg", return_value=None), patch.object(
                runner, "_run_stage", side_effect=lambda _name, args: calls.append(args)
            ):
                runner.render_gantt()
            self.assertEqual(macro_midpoints(pd.read_csv(paths.macro_csv)), [1.0, 4.0])
            self.assertEqual(len(calls), 2)
            self.assertIn("ffmpeg was unavailable", runner.result.warnings[0])

    def test_stage3_output_construction_has_no_fixed_scenario3_name(self) -> None:
        source = (PROJECT_ROOT / "src" / "segmentation" / "micro_segmentation.py").read_text(encoding="utf-8")
        self.assertNotIn('segments_csv = output_dir / "micro_segments_descriptive_PedNYC1_scenario3_v6.csv"', source)
        self.assertIn("args.scenario_stem", source)

    def test_scenario3_committed_regression_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            artifact_root = Path(temp)
            raw = artifact_root / "CSV_Scenario-Ped-3_Session-x.csv"
            paths = ScenarioPaths.from_raw(raw, PROJECT_ROOT, output_root=artifact_root)
            ground_macro = PROJECT_ROOT / "outputs" / "graphs" / "macro_segmentation" / "macro_segments_PedNYC1_scenario3_v2.csv"
            ground_micro_dir = PROJECT_ROOT / "outputs" / "graphs" / "micro_segmentation" / "v6"
            paths.macro_csv.parent.mkdir(parents=True)
            paths.micro_v6_dir.mkdir(parents=True)
            shutil.copyfile(ground_macro, paths.macro_csv)
            shutil.copyfile(ground_micro_dir / "micro_segments_descriptive_PedNYC1_scenario3_v6.csv", paths.micro_segments_csv)
            shutil.copyfile(ground_micro_dir / "micro_change_boundaries_PedNYC1_scenario3_v6.csv", paths.micro_boundaries_csv)
            comparison = compare_scenario3_regression(paths)
            self.assertTrue(comparison["boundary_match"])
            self.assertTrue(comparison["macro_labels_match"])
            self.assertTrue(comparison["tag_sequence_match"])
            self.assertTrue(comparison["boundary_record_count_match"])
            np.testing.assert_allclose(comparison["generated_boundaries"], EXPECTED_SCENARIO3_BOUNDARIES, atol=0.06)


if __name__ == "__main__":
    unittest.main()
