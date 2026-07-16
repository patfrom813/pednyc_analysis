# PedNYC analysis

## Scenario-agnostic canonical v2/v6 pipeline

Install `requirements.txt`, then give the pipeline a compatible logger CSV. It
runs Unity-array decoding, metrics v1, macrosegmentation v2, microsegmentation
v6, all canonical diagnostic plots, Gantt rendering, validation, and a
scenario-specific Markdown report without changing the tuned algorithms.

```powershell
python src\scenario_pipeline\run_scenario.py data\raw\CSV_Scenario-Ped-3_Session-temp_2024-02-22-13-48-59.csv
python src\scenario_pipeline\run_scenario.py --scenario-id 3
python src\scenario_pipeline\run_scenario.py --all --continue-on-error
```

Use `--output-root PATH` to relocate generated artifacts, `--skip-render` to
skip renderer outputs, `--preview-only` to request one preview per macro
segment, or `--preview-time SECONDS` for one preview. Pass an encoder using
`--ffmpeg PATH`. If no encoder is available, the normal run continues and
writes macro-midpoint previews plus a warning in the report.

Decoded and metrics tables are written below
`data/processed/pednyc1/scenarioN`. Canonical graphs and the report are written
below `outputs/graphs/exact_v6/pednyc1/scenarioN`; the complete micro output set
is under its `micro/v6` directory. Stage 1 also writes scenario-specific copies
below `4frame_view/pednyc1/scenarioN/metrics_v1`.

Every micro segment retains three independent dimensions: `motion_tag`,
`head_tag`, and `car_tag`. They are deliberately not fused. Observable tags are
direct threshold measurements; inferred tags are structured hypotheses such as
hesitation, yielding, proceeding, conflicted motion, or brief head checking.

Known limitations:

- Inputs must use the same logger column layout and filename convention
  `CSV_Scenario-Ped-N_Session-*.csv`.
- Scenario 3 is the only compatible raw CSV currently committed.
- `ScenarioTime` is used when it is seconds-like; the canonical fallback logic
  uses cleaned `dt`, then `GameTime`, then frame number timing.
- The committed scenario-3 ground-truth raw-time columns came from a different
  local feature file. Regression therefore compares elapsed boundaries, macro
  labels, and ordered motion/head/car tags rather than raw-time bytes.
- Macro boundary candidates can collapse; reports state the actual segment
  count instead of assuming every scenario has five segments.
- The Stage 1 implementation produces up to ten numbered diagnostic PNGs; the
  obsolete kink-workflow PNG is not part of the canonical Stage 1 output.

## Animated micro-segmentation Gantt MP4

Generate the exact v6 micro-segmentation data, then render the compositing-ready
video:

```powershell
python src\segmentation\micro_segmentation.py
python src\segmentation\render_micro_gantt_mp4.py
```

The renderer writes:

`outputs/graphs/micro_segmentation/v6/micro_gantt_observable_inferred_PedNYC1_scenario3_v6.mp4`

The default is 1280x720 H.264 at 30 fps. The red playhead advances in scenario
time, and the bottom card shows the active macro segment, micro-segment, motion,
head, and car-context tags. Install `requirements.txt`, put ffmpeg on `PATH`, or
pass an explicit encoder with `--ffmpeg C:\path\to\ffmpeg.exe`.

Render a single layout preview without invoking ffmpeg:

```powershell
python src\segmentation\render_micro_gantt_mp4.py --preview-time 13.577
```
