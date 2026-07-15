"""End-to-end orchestration for one participant scenario."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DatasetConfig, PipelineConfig
from src.features import BehaviorClassifier, EventMerger, MetricsEngine, normalize_columns
from src.preprocess import clean_metrics, load_scenario, validate_scenario
from src.segmentation import MacroSegmenter, MicroSegmenter
from src.visualization import plot_detected_events, plot_macro_segments, plot_micro_windows, plot_speed_profile


class ScenarioPipeline:
    """Run the complete PedNYC processing chain for a discovered CSV."""

    def __init__(self, dataset_config: DatasetConfig, pipeline_config: PipelineConfig):
        self.dataset_config = dataset_config
        self.pipeline_config = pipeline_config

    def process(self, pid: int, scenario: str, save_plots: bool = True) -> dict[str, Any]:
        """Process one scenario, save tabular/plot outputs, and return all artifacts."""
        csv_path = self.dataset_config.get_scenario_csv(pid, scenario)
        raw = load_scenario(csv_path)
        normalized = normalize_columns(raw)
        validation = validate_scenario(normalized)
        if not validation["valid"]:
            raise ValueError(f"Scenario validation failed: {validation}")
        cleaned = clean_metrics(normalized, self.pipeline_config)
        metrics = MetricsEngine(self.pipeline_config).compute(cleaned)
        macro_segments = MacroSegmenter(self.pipeline_config).segment(metrics)
        micro_windows = MicroSegmenter(self.pipeline_config).extract_windows(metrics, macro_segments)
        classified = BehaviorClassifier(self.pipeline_config).classify(micro_windows)
        events = EventMerger(self.pipeline_config).merge(classified)
        output_dir = self.dataset_config.get_output_dir(pid, scenario)
        output_paths: dict[str, Path] = {}
        tables = {
            "frame_metrics": metrics,
            "macro_segments": macro_segments,
            "micro_windows": classified,
            "merged_events": events,
        }
        for name, frame in tables.items():
            path = output_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
            output_paths[name] = path
        if save_plots:
            graph_dir = self.dataset_config.get_graphs_dir(pid)
            prefix = f"scenario{scenario}"
            output_paths["speed_plot"] = plot_speed_profile(metrics, graph_dir / f"{prefix}_speed.png")
            output_paths["macro_plot"] = plot_macro_segments(metrics, macro_segments, graph_dir / f"{prefix}_macro.png")
            output_paths["micro_plot"] = plot_micro_windows(metrics, micro_windows, graph_dir / f"{prefix}_micro.png")
            output_paths["events_plot"] = plot_detected_events(metrics, events, graph_dir / f"{prefix}_events.png")
        return {
            "raw": raw,
            "cleaned": cleaned,
            "metrics": metrics,
            "macro_segments": macro_segments,
            "micro_windows": micro_windows,
            "classified_windows": classified,
            "events": events,
            "validation": validation,
            "output_paths": output_paths,
        }
