# PedNYC analysis

## Interactive micro-segmentation Gantt

Run the micro-segmentation pipeline:

```powershell
python src\segmentation\micro_segmentation.py
```

Then open
`outputs/graphs/micro_segmentation/v6/micro_gantt_observable_inferred_PedNYC1_scenario3_v6.html`
in a web browser. Align the simulation video to the same timestamp and press
**Play**. The red playhead advances in real elapsed time at 1x; the player also
supports pause, restart, millisecond seeking, playback-speed changes, and shows
the active macro segment plus motion, head, and car-context tags.
