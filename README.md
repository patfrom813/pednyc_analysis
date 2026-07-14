# PedNYC analysis

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
