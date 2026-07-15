# PedNYC analysis

## General pipeline

The reusable pipeline discovers participant folders and flat scenario CSVs at
runtime. No participant or scenario is selected in source code.

```powershell
python run_batch.py -p 1 --list
python run_single.py -p 1 -s 3
python run_batch.py -p 1 --pattern "1*" --no-plots
python run_batch.py -p 2 --all
python validate.py -p 1 -s 3
```

Pass `--config config_default.yaml` to load editable thresholds. Processed
tables are written below `outputs/processed/pednyc{pid}/scenario{scenario}`;
participant plots are written below `outputs/graphs/pednyc{pid}`.

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
