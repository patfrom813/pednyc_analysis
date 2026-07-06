from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"C:\Users\patl5\OneDrive\Desktop\BURE\pednyc_analysis")

INPUT_CSV = ROOT / "outputs" / "graphs" / "macro_segmentation" / "features_with_macro_segments.csv"

OUTPUT_DIR = ROOT / "4frame_view" / "macro_segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MP4 = OUTPUT_DIR / "ped_macro_segmentation_pointer.mp4"
OUTPUT_PNG = OUTPUT_DIR / "ped_macro_segmentation_static.png"

TIME_COL = "elapsed_time"          # use your existing elapsed_time column
SPEED_COL = "ped_speed_xz_smooth"  # your actual pedestrian smooth speed column
SEGMENT_COL = "macro_label"        # your actual macro segment label column

FPS = 30

SEGMENT_COLORS = {
    "pre_movement_low_motion": "#f4cccc",
    "acceleration": "#d9ead3",
    "main_movement": "#cfe2f3",
    "deceleration": "#fce5cd",
    "post_movement_low_motion": "#f4cccc",
}

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

# IMPORTANT: remove hidden leading/trailing spaces from column names
df.columns = df.columns.str.strip()

print("\nColumns found:")
print(df.columns.tolist())

required_cols = [TIME_COL, SPEED_COL, SEGMENT_COL]
missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=[TIME_COL, SPEED_COL, SEGMENT_COL]).copy()
df = df.sort_values(TIME_COL).reset_index(drop=True)

# Make sure elapsed time starts exactly at 0
df[TIME_COL] = df[TIME_COL] - df[TIME_COL].iloc[0]

t = df[TIME_COL].to_numpy()
y = df[SPEED_COL].to_numpy()

# ============================================================
# BUILD MACRO SEGMENTS FROM macro_label
# ============================================================

segments = []

start_idx = 0
current_label = df.loc[0, SEGMENT_COL]

for i in range(1, len(df)):
    label = df.loc[i, SEGMENT_COL]

    if label != current_label:
        start_t = float(df.loc[start_idx, TIME_COL])
        end_t = float(df.loc[i - 1, TIME_COL])
        segments.append((current_label, start_t, end_t))

        start_idx = i
        current_label = label

# Last segment
segments.append(
    (
        current_label,
        float(df.loc[start_idx, TIME_COL]),
        float(df.loc[len(df) - 1, TIME_COL]),
    )
)

print("\nMacro segments used:")
for label, start, end in segments:
    print(f"{label}: {start:.2f}s to {end:.2f}s")

# ============================================================
# PLOT SETUP
# ============================================================

fig, ax = plt.subplots(figsize=(18, 7))

# Segment shading
used_labels = set()

for label, start, end in segments:
    color = SEGMENT_COLORS.get(label, "#dddddd")
    legend_label = label if label not in used_labels else None

    ax.axvspan(start, end, color=color, alpha=0.35, label=legend_label)
    ax.axvline(start, color="firebrick", linewidth=1.8)

    used_labels.add(label)

# final red line
ax.axvline(segments[-1][2], color="firebrick", linewidth=1.8)

# Main speed line
ax.plot(t, y, color="tab:blue", linewidth=2.0, label=SPEED_COL)

# Orange smoothing/trend line
trend = pd.Series(y).rolling(window=7, center=True, min_periods=1).mean()
ax.plot(t, trend, color="tab:orange", linestyle="--", linewidth=2.0, label="macro trend smoothing")

# Segment text labels
y_text = np.nanmax(y) * 0.92

for label, start, end in segments:
    x_mid = (start + end) / 2
    ax.text(
        x_mid,
        y_text,
        label,
        ha="center",
        va="center",
        fontsize=11,
        color="black",
    )

# Moving pointer
pointer_line = ax.axvline(0, color="black", linewidth=2.5)
pointer_dot, = ax.plot([], [], "o", color="black", markersize=8)

time_text = ax.text(
    0.015,
    0.96,
    "",
    transform=ax.transAxes,
    fontsize=12,
    va="top",
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
)

ax.set_title("Pedestrian Macro Segmentation with Animated Time Pointer")
ax.set_xlabel("Scenario elapsed time, seconds")
ax.set_ylabel("Pedestrian speed XZ smooth")
ax.grid(True, alpha=0.3)

ax.set_xlim(float(t[0]), float(t[-1]))
ax.set_ylim(min(0, np.nanmin(y) - 0.05), np.nanmax(y) + 0.15)
ax.legend(loc="upper right")

# Save static plot
fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
print(f"\nSaved static plot to: {OUTPUT_PNG}")

# ============================================================
# ANIMATION
# ============================================================

duration_seconds = float(t[-1])

animation_times = np.arange(0, duration_seconds + 1 / FPS, 1 / FPS)
y_interp = np.interp(animation_times, t, y)

def update(frame_idx):
    current_t = animation_times[frame_idx]
    current_y = y_interp[frame_idx]

    pointer_line.set_xdata([current_t, current_t])
    pointer_dot.set_data([current_t], [current_y])
    time_text.set_text(f"Elapsed time: {current_t:.2f} s")

    return pointer_line, pointer_dot, time_text

anim = FuncAnimation(
    fig,
    update,
    frames=len(animation_times),
    interval=1000 / FPS,
    blit=True,
)

writer = FFMpegWriter(fps=FPS, bitrate=3000)
anim.save(OUTPUT_MP4, writer=writer)

plt.close(fig)

print(f"Saved animation to: {OUTPUT_MP4}")