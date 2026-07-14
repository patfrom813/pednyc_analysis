#!/usr/bin/env python3
"""
FFmpeg 4-frame compositor: Replace top-right quadrant with micro-segmentation Gantt chart.
Inspects both videos first, then generates the correct ffmpeg command.
"""

import subprocess
import json
import sys
from pathlib import Path

# === CONFIGURE YOUR PATHS HERE ===
BASE_4FRAME = r"C:\Users\patl5\OneDrive\Desktop\BURE\pednyc_analysis\4frame_view\PedNYC1_scenario3_4frame_macro_topright_slightly_smaller.mp4"
GANTT_MP4 = r"C:\Users\patl5\OneDrive\Desktop\BURE\pednyc_analysis\outputs\graphs\micro_segmentation\v6\micro_gantt_observable_inferred_PedNYC1_scenario3_v6.mp4"
OUTPUT_MP4 = r"C:\Users\patl5\OneDrive\Desktop\BURE\pednyc_analysis\4frame_view\PedNYC1_scenario3_4frame_micro_gantt_v6.mp4"


def probe_video(path):
    """Get video dimensions, duration, and frame rate using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: ffprobe failed for {path}")
        print(result.stderr)
        sys.exit(1)

    data = json.loads(result.stdout)
    video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if not video_stream:
        print(f"ERROR: No video stream found in {path}")
        sys.exit(1)

    width = int(video_stream["width"])
    height = int(video_stream["height"])

    duration = float(video_stream.get("duration", data["format"].get("duration", 0)))

    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)

    return {
        "width": width,
        "height": height,
        "duration": duration,
        "fps": fps,
        "codec": video_stream.get("codec_name", "unknown"),
        "pix_fmt": video_stream.get("pix_fmt", "unknown")
    }


def main():
    print("=" * 60)
    print("4-FRAME COMPOSITOR: Inspecting videos...")
    print("=" * 60)

    print(f"\n[1] Probing 4-frame base video:")
    print(f"    {BASE_4FRAME}")
    base_info = probe_video(BASE_4FRAME)
    print(f"    Resolution: {base_info['width']}x{base_info['height']}")
    print(f"    Duration:   {base_info['duration']:.3f}s")
    print(f"    FPS:        {base_info['fps']:.2f}")
    print(f"    Codec:      {base_info['codec']}")

    print(f"\n[2] Probing Gantt chart video:")
    print(f"    {GANTT_MP4}")
    gantt_info = probe_video(GANTT_MP4)
    print(f"    Resolution: {gantt_info['width']}x{gantt_info['height']}")
    print(f"    Duration:   {gantt_info['duration']:.3f}s")
    print(f"    FPS:        {gantt_info['fps']:.2f}")
    print(f"    Codec:      {gantt_info['codec']}")

    # Calculate layout
    W = base_info["width"]
    H = base_info["height"]
    half_W = W // 2
    half_H = H // 2

    target_w = half_W
    target_h = half_H

    print(f"\n[3] Layout analysis:")
    print(f"    Base video:        {W}x{H}")
    print(f"    Quadrant size:     {target_w}x{target_h} (top-right)")
    print(f"    Gantt original:    {gantt_info['width']}x{gantt_info['height']}")

    gantt_w = gantt_info["width"]
    gantt_h = gantt_info["height"]
    gantt_ar = gantt_w / gantt_h
    target_ar = target_w / target_h

    if gantt_ar > target_ar:
        scale_w = target_w
        scale_h = int(target_w / gantt_ar)
        pad_top = (target_h - scale_h) // 2
        pad_bottom = target_h - scale_h - pad_top
        pad_left = 0
        pad_right = 0
    else:
        scale_h = target_h
        scale_w = int(target_h * gantt_ar)
        pad_left = (target_w - scale_w) // 2
        pad_right = target_w - scale_w - pad_left
        pad_top = 0
        pad_bottom = 0

    print(f"    Scale to:          {scale_w}x{scale_h}")
    print(f"    Padding:           top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")

    x_offset = half_W
    y_offset = 0

    print(f"    Placement offset:  x={x_offset}, y={y_offset}")

    out_duration = min(base_info["duration"], gantt_info["duration"])
    print(f"    Output duration:   {out_duration:.3f}s (shorter of the two)")

    # ===== FIX: Correct filter_complex syntax =====
    # Chain 1: Process gantt video (input 1) -> scale -> pad -> label as [gantt]
    # Chain 2: Take base video (input 0) and overlay [gantt] at position -> label as [outv]
    # The semicolon separates independent filter chains.

    filter_complex = (
        f"[1:v]scale={scale_w}:{scale_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:{pad_left}:{pad_top}:black[gantt];"
        f"[0:v][gantt]overlay={x_offset}:{y_offset}:enable='between(t\\,0\\,{out_duration})'[outv]"
    )

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", BASE_4FRAME,
        "-i", GANTT_MP4,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-t", str(out_duration),
        "-movflags", "+faststart",
        OUTPUT_MP4
    ]

    print(f"\n[4] Generated FFmpeg command:")
    print("-" * 60)
    # Print with line breaks for readability
    print(" \\n    ".join(ffmpeg_cmd))
    print("-" * 60)

    # One-liner version
    one_liner = (
        f'ffmpeg -y -i "{BASE_4FRAME}" -i "{GANTT_MP4}" '
        f'-filter_complex "{filter_complex}" '
        f'-map "[outv]" -map 0:a? '
        f'-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p '
        f'-t {out_duration} -movflags +faststart "{OUTPUT_MP4}"'
    )

    print(f"\n[5] One-liner (copy-paste ready):")
    print(one_liner)

    print(f"\n[6] Output will be saved to:")
    print(f"    {OUTPUT_MP4}")

    response = input("\nRun ffmpeg now? [y/N]: ").strip().lower()
    if response == 'y':
        print("\nRunning ffmpeg...")
        result = subprocess.run(ffmpeg_cmd)
        if result.returncode == 0:
            print(f"\nSUCCESS: Output saved to {OUTPUT_MP4}")
        else:
            print(f"\nFAILED with exit code {result.returncode}")
            print("Try running the one-liner manually to see full error output.")
    else:
        print("\nAborted. Copy the command above and run manually.")


if __name__ == "__main__":
    main()