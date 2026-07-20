# PedNYC1_scenario3 canonical pipeline report

## Input and timing

- Input: `CSV_Scenario-Ped-3_Session-temp_2024-02-22-16-58-19.csv`
- Scenario ID: 3
- Scenario stem: `PedNYC1_scenario3`
- Raw rows: 566
- Decoded rows: 566
- Decoded columns: 1006
- Elapsed duration: 26.343018 seconds
- Median valid dt: 0.043945 seconds
- Approximate sampling frequency: 22.755717 Hz

### Time sources

- metrics_v1: `ScenarioTime.diff`
- decode: `preserve_raw_rows`
- macro_v2: `cumulative_dt_seconds`
- micro_v6: `ScenarioTime_seconds`

## Macro segments

| segment_id | macro_label | start_idx | end_idx | time_start_sec | time_end_sec | duration_sec |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | pre_movement_low_motion | 0 | 175 | 0.000000 | 8.205322 | 8.205322 |
| 1 | acceleration | 175 | 266 | 8.205322 | 12.274414 | 4.069092 |
| 2 | crossing_near_car | 266 | 423 | 12.274414 | 19.750000 | 7.475586 |
| 3 | deceleration | 423 | 545 | 19.750000 | 25.463135 | 5.713135 |
| 4 | post_movement_low_motion | 545 | 565 | 25.463135 | 26.343018 | 0.879883 |

Macro segment count: 5
Micro segment count: 24
Micro boundary record count: 32

## Micro tag counts

### Motion

- `mixed_motion`: 5
- `near_stationary`: 1
- `pausing`: 2
- `speed_decreasing`: 5
- `speed_increasing`: 8
- `speed_steady`: 3

### Head

- `head_active`: 20
- `head_checking`: 2
- `head_still`: 2

### Car context

- `neutral`: 19
- `proceeding`: 3
- `yielding`: 2

### Observability

- Micro segments with at least one inferred tag: 12
- Micro segments with only observable tags: 12

## Scenario 3 regression

- Boundary tolerance: 0.06 seconds
- Elapsed boundaries match: False
- Macro labels match: False
- Generated/expected micro segments: 24/29
- Ordered motion/head/car tag sequences match: False
- Generated/expected boundary records: 32/37

## Artifact checklist

- [x] `decoded_csv` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\data\processed\pednyc1\scenario3\decoded_clean_PedNYC1_scenario3.csv`
- [x] `fourframe_00_original_decoded_column_inventory` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1\00_original_decoded_column_inventory.txt`
- [x] `fourframe_01_final_metrics_column_inventory` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1\01_final_metrics_column_inventory.txt`
- [x] `fourframe_legacy_segments_csv` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1\segments_PedNYC1_scenario3_v1.csv`
- [x] `fourframe_metrics_csv` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1\features_PedNYC1_scenario3_metrics_v1.csv`
- [x] `fourframe_metrics_plot_01` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1\01_ped_speed_raw_vs_smooth.png`
- [x] `fourframe_metrics_plot_02` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1\02_ped_accel_raw_vs_smooth.png`
- [x] `fourframe_metrics_plot_03` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1\03_car_speed_raw_vs_smooth.png`
- [x] `fourframe_metrics_plot_04` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1\04_colored_ped_speed_segments_v1.png`
- [x] `fourframe_metrics_plot_05` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1\05_car_ped_distance.png`
- [x] `fourframe_metrics_plot_06` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1\06_distance_closing.png`
- [x] `fourframe_metrics_plot_07` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1\07_smoothed_acceleration_comparison.png`
- [x] `fourframe_metrics_plot_08` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1\08_core_movement_interaction_signals.png`
- [x] `fourframe_metrics_plot_09` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1\09_avatar_vr_gap_noise_check.png`
- [x] `fourframe_metrics_plot_10` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1\10_optional_head_features.png`
- [x] `gantt_mp4` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\micro\v6\micro_gantt_observable_inferred_PedNYC1_scenario3_v6.mp4`
- [x] `legacy_segments_csv` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\data\processed\pednyc1\scenario3\segments_PedNYC1_scenario3_v1.csv`
- [x] `macro_csv` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\macro_v2\macro_segments_PedNYC1_scenario3_v2.csv`
- [x] `macro_png` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\macro_v2\macro_segments_PedNYC1_scenario3_v2.png`
- [x] `metrics_00_original_decoded_column_inventory` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\metrics_v1\00_original_decoded_column_inventory.txt`
- [x] `metrics_01_final_metrics_column_inventory` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\metrics_v1\01_final_metrics_column_inventory.txt`
- [x] `metrics_csv` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\data\processed\pednyc1\scenario3\features_PedNYC1_scenario3_metrics_v1.csv`
- [x] `metrics_plot_01` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\metrics_v1\01_ped_speed_raw_vs_smooth.png`
- [x] `metrics_plot_02` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\metrics_v1\02_ped_accel_raw_vs_smooth.png`
- [x] `metrics_plot_03` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\metrics_v1\03_car_speed_raw_vs_smooth.png`
- [x] `metrics_plot_04` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\metrics_v1\04_colored_ped_speed_segments_v1.png`
- [x] `metrics_plot_05` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\metrics_v1\05_car_ped_distance.png`
- [x] `metrics_plot_06` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\metrics_v1\06_distance_closing.png`
- [x] `metrics_plot_07` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\metrics_v1\07_smoothed_acceleration_comparison.png`
- [x] `metrics_plot_08` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\metrics_v1\08_core_movement_interaction_signals.png`
- [x] `metrics_plot_09` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\metrics_v1\09_avatar_vr_gap_noise_check.png`
- [x] `metrics_plot_10` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\metrics_v1\10_optional_head_features.png`
- [x] `micro_boundaries_csv` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\micro\v6\micro_change_boundaries_PedNYC1_scenario3_v6.csv`
- [x] `micro_frame_csv` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\micro\v6\features_with_descriptive_micro_segments_PedNYC1_scenario3_v6.csv`
- [x] `micro_gantt_png` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\micro\v6\micro_gantt_observable_inferred_PedNYC1_scenario3_v6.png`
- [x] `micro_segments_csv` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\micro\v6\micro_segments_descriptive_PedNYC1_scenario3_v6.csv`
- [x] `micro_stacked_png` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\micro\v6\micro_stacked_observable_inferred_PedNYC1_scenario3_v6.png`
- [x] `micro_standard_png` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\micro\v6\micro_standard_3panel_PedNYC1_scenario3_v6.png`
- [x] `micro_summary_csv` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\micro\v6\micro_segment_summary_by_macro_PedNYC1_scenario3_v6.csv`
- [x] `tag_definitions_csv` — generated: `C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\micro\v6\micro_event_tag_definitions_v6.csv`

## Warnings and anomalies

- None.

## Failed or skipped outputs

- None.

## Reproduction commands

```text
C:\Users\patl5\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe src/preprocess/decode_first.py --input-csv C:\Users\patl5\OneDrive\Desktop\BURE\pednyc_analysis\data\raw\CSV_Scenario-Ped-3_Session-temp_2024-02-22-16-58-19.csv --output-csv C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\data\processed\pednyc1\scenario3\decoded_clean_PedNYC1_scenario3.csv
```
```text
C:\Users\patl5\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe src/features/create_metrics_and_graph.py --input-csv C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\data\processed\pednyc1\scenario3\decoded_clean_PedNYC1_scenario3.csv --output-csv C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\data\processed\pednyc1\scenario3\features_PedNYC1_scenario3_metrics_v1.csv --segment-csv C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\data\processed\pednyc1\scenario3\segments_PedNYC1_scenario3_v1.csv --graph-dir C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\metrics_v1 --fourframe-dir C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\4frame_view\pednyc1\scenario3\metrics_v1
```
```text
C:\Users\patl5\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe src/segmentation/macro_segmentation.py --input-csv C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\data\processed\pednyc1\scenario3\features_PedNYC1_scenario3_metrics_v1.csv --output-dir C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\macro_v2 --segments-csv C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\macro_v2\macro_segments_PedNYC1_scenario3_v2.csv --plot-path C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\macro_v2\macro_segments_PedNYC1_scenario3_v2.png
```
```text
C:\Users\patl5\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe src/segmentation/micro_segmentation.py --feature-csv C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\data\processed\pednyc1\scenario3\features_PedNYC1_scenario3_metrics_v1.csv --macro-csv C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\macro_v2\macro_segments_PedNYC1_scenario3_v2.csv --output-dir C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\micro --scenario-stem PedNYC1_scenario3
```
```text
C:\Users\patl5\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe src/segmentation/render_micro_gantt_mp4.py --segments-csv C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\micro\v6\micro_segments_descriptive_PedNYC1_scenario3_v6.csv --macro-csv C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\macro_v2\macro_segments_PedNYC1_scenario3_v2.csv --output C:\Users\patl5\OneDrive\Documents\BURE\pednyc_analysis\outputs\graphs\exact_v6\pednyc1\scenario3\micro\v6\micro_gantt_observable_inferred_PedNYC1_scenario3_v6.mp4 --ffmpeg C:\Users\patl5\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe
```
