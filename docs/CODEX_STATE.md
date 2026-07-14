# CODEX State

## Objective
Replace the v6 web Gantt player with a standalone Python renderer that produces a 1280x720, 30 fps, H.264 MP4 for the top-right panel of the PedNYC four-frame video; verify it, commit, and push on `codex/micro-segmentation-v2` without modifying `main`.

## Confirmed findings
- The task started from a clean `codex/micro-segmentation-v2` branch at `58ee5d9`; local `main` remains at `a6ec5b7`.
- No applicable `AGENTS.md` exists.
- Exact animation inputs are `outputs/graphs/micro_segmentation/v6/micro_segments_descriptive_PedNYC1_scenario3_v6.csv` (29 segments) and `outputs/graphs/macro_segmentation/macro_segments_PedNYC1_scenario3_v2.csv` (5 macro segments).
- Exact scenario duration is 31.002930 seconds.
- No system `ffmpeg`, `imageio-ffmpeg`, or other local encoder was found.
- The standalone renderer now reads the exact v6 inputs, validates tag/schema coverage, and supports 1280x720 H.264 rendering plus single-frame PNG previews.
- Visual QA at 13.577 seconds passed after two iterations: the playhead/time badge align, the x-axis is clear, and the bottom panel identifies M2, segment 11, speed_increasing, head_active, and neutral.
- Full encoding completed: 931 frames, 1280x720, 30 fps, H.264 High profile, yuv420p, 31.03-second container duration, and 683,856-byte output.
- ffmpeg decoded all 931 frames successfully. Encoded frames at the start, middle, and endpoint were visually inspected and matched the expected macro/segment/tag state.
- At 30 fps, arbitrary timestamps are quantized to 33.333 ms frame intervals (for example, 13.577 seconds displays on the 13.600-second frame); the final frame explicitly clamps to the exact 31.002930-second scenario endpoint.
- Commit `e810a33` (`Replace web Gantt with MP4 renderer`) was created and pushed to `origin/codex/micro-segmentation-v2`; `main` was not modified.

## Important decisions and reasons
- Use exact v6 CSV data, not visually approximated intervals.
- Create a separate, directly runnable MP4 renderer so the analysis pipeline and video-rendering concerns stay isolated.
- Use matplotlib animation with H.264/yuv420p output. Resolve ffmpeg from `--ffmpeg`, environment variables, PATH, or `imageio-ffmpeg` in that order.
- Remove the HTML generator and committed HTML artifact because the user explicitly wants to abandon the web path.
- Render an exact final endpoint frame. At 30 fps this requires 931 frames, yielding a container duration of about 31.033 seconds; timestamps remain real-time through 31.000 seconds and the final frame clamps to 31.002930 seconds.

## Files inspected or modified
- Inspected: `src/segmentation/micro_segmentation.py`, exact v6 segment CSV, macro CSV, README, and prior state file.
- Added: `src/segmentation/render_micro_gantt_mp4.py`.
- Added: `outputs/graphs/micro_segmentation/v6/micro_gantt_observable_inferred_PedNYC1_scenario3_v6.mp4`.
- Modified: `src/segmentation/micro_segmentation.py`, `README.md`, `requirements.txt`, and `docs/CODEX_STATE.md`.
- Removed: `outputs/graphs/micro_segmentation/v6/micro_gantt_observable_inferred_PedNYC1_scenario3_v6.html`.

## Commands and tests run
- Read all applicable repository instructions/state.
- Ran Git status/log checks and inspected exact CSV schemas.
- Searched PATH, repository packages, common Windows program locations, and user-local applications for ffmpeg; none found.
- In-memory compile check for both segmentation scripts: passed.
- Rendered and dimension-checked 1280x720 previews at 13.577 seconds.
- Visually inspected the corrected preview: passed.
- Installed the declared `imageio-ffmpeg` dependency into the existing local package directory for validation.
- Full 931-frame MP4 render using libx264: passed.
- ffmpeg stream probe confirmed H.264 High, yuv420p progressive, 1280x720, 30 fps, and 31.03 seconds.
- ffmpeg copy/decode check counted 931 frames.
- Extracted and visually inspected encoded frames at 0.000, approximately 13.577, and 31.003 seconds: passed.
- Verified the repository MP4 SHA-256 matches the fully validated temporary render.
- Final compile, CLI help, and `git diff --check`: passed.
- Staged only the seven intended paths, inspected the staged snapshot, committed, and pushed the feature branch successfully.

## Current failures
- No implementation or encoding failure remains.
- The local `imageio-ffmpeg` installation inherited restrictive ACLs in this managed environment, so validation used its copied ffmpeg executable through `--ffmpeg`; the resolver was fixed so explicit paths are accepted before optional package imports.

## Remaining work
- No required implementation work remains for the standalone MP4 renderer.
- Integration into the existing four-frame ffmpeg composition is a separate optional next step.

## Exact next step
Use `outputs/graphs/micro_segmentation/v6/micro_gantt_observable_inferred_PedNYC1_scenario3_v6.mp4` as the top-right ffmpeg input in place of the existing speed graph.
