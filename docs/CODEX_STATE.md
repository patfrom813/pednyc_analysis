# CODEX State

## Objective

Generalize the canonical PedNYC raw → decode → metrics v1 → macro v2 → micro
v6 → plots/Gantt workflow without changing its tuned formulas, thresholds,
ordering, or fallback behavior.

## Branch and safety

- Active branch: `refactored`.
- Requested starting point: `origin/codex/micro-segmentation-v2` at `f2e2667`.
- The pre-existing local `refactored` branch is a descendant of `f2e2667`.
- `main` remains untouched at `a6ec5b7`.
- Pre-existing unrelated dirty changes were preserved and excluded from the
  canonical-pipeline commit scope.
- No applicable `AGENTS.md` exists.

## Architecture

- `src/scenario_pipeline/scenario.py`: filename parsing, raw discovery, and the
  immutable `ScenarioPaths` artifact map.
- `src/scenario_pipeline/runner.py`: public canonical stage methods, sanity
  checks, ffmpeg fallback, regression comparison, and Markdown reporting.
- `src/scenario_pipeline/run_scenario.py`: single/raw-ID/all-scenarios CLI.
- The package is named `scenario_pipeline` because this branch already contains
  a committed `src/pipeline.py`; a `src/pipeline/` package would shadow it.
- Canonical logic remains in `decode_first.py`,
  `create_metrics_and_graph.py`, `macro_segmentation.py`, and
  `micro_segmentation.py`. The runner supplies explicit scenario paths.

## Verified implementation discrepancies

- Committed raw `ScenarioTime` is already 0–31.002930 seconds;
  214971–216902 is `Frame Number`.
- Committed ground-truth macro/micro raw-time fields were generated from a
  different local feature source, so regression uses elapsed time and tags.
- Stage 1 retains a usable existing `dt`; otherwise it derives
  `ScenarioTime.diff()`.
- Macro boundary candidates may collapse; the implementation does not promise
  exactly five segments for arbitrary scenarios.
- Micro lag zero is a one-frame delta, peak prominence is immediate-neighbor
  based, and merged provenance follows the surviving boundary index.
- Stage 1 produces up to ten numbered plots. The old kink PNG is not canonical.
- The committed root-level scenario-3 feature CSV is stale and is never used by
  the generalized runner.

## Scenario 3 regression result

- Raw/decode rows: 648/648.
- Decoded columns: 1006.
- Elapsed duration: 31.002930 seconds.
- Median valid dt: 0.047608 seconds (~21.005 Hz).
- Macro segments: 5.
- Elapsed boundaries: 0, 7.828857, 10.790771, 20.499023, 28.346191,
  31.002930; all match with an absolute tolerance of 0.06 seconds.
- Macro labels: exact ordered match.
- Micro segments: 29/29.
- Ordered `motion_tag`/`head_tag`/`car_tag` triples: exact match after safely
  parsing the padded committed CSV with `index_col=False`.
- Boundary records: 37/37.
- Motion counts: pausing 7, speed_decreasing 6, speed_steady 5, mixed_motion 4,
  hesitating 3, speed_increasing 3, near_stationary 1.
- Head counts: head_active 18, head_still 10, head_checking 1.
- Car counts: neutral 20, yielding 5, proceeding 4.
- Scenario-specific MP4: 1280×720, 30 fps, 931 frames, H.264/yuv420p,
  endpoint 31.002930 seconds.

## Outputs

Scenario 3 generated the decoded CSV, metrics CSV, legacy v1 segments, both
inventories, ten Stage 1 diagnostic PNGs and four-frame copies, macro CSV/PNG,
all five micro CSVs, all three micro PNGs, the Gantt MP4, and the Markdown
report at:

`outputs/graphs/exact_v6/pednyc1/scenario3/PedNYC1_scenario3_report.md`

Only scenario 3 is currently available under `data/raw`; no scenarios were
fabricated.

## Commands run

```powershell
python -m unittest tests.test_scenario_pipeline -v
python src\scenario_pipeline\run_scenario.py --scenario-id 3 --force
```

The full run used the repository `.python_packages` through `PYTHONPATH` and the
available ffmpeg executable. Unit coverage includes path parsing, output names,
time-source selection, one-column rejection, stale-metrics protection, artifact
completeness, missing-ffmpeg midpoint fallback, and scenario-3 boundary/tag
regression.

## Known limitations

- Compatible input filenames must follow `CSV_Scenario-Ped-N_Session-*.csv`.
- Inputs must use the same logger column layout/units as the existing study.
- Thresholds are intentionally not retuned when a new scenario fails sanity
  checks.
- Large generated decoded/features/MP4 artifacts are ignored and are not part
  of the intended commit.

## Next scenario command

```powershell
python src\scenario_pipeline\run_scenario.py data\raw\CSV_Scenario-Ped-N_Session-....csv
```
