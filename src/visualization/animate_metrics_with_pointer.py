from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# ============================================================
# Paths
# ============================================================

ROOT = Path(r"C:\Users\patl5\OneDrive\Desktop\BURE\pednyc_analysis")

# This animation script is meant to run AFTER scenario3_metrics_v1_fixed.py.
# It reads the frame-level metrics CSV created by that script.
INPUT_CSV = ROOT / "data" / "processed" / "features_PedNYC1_scenario3_metrics_v1.csv"

OUTPUT_DIR = ROOT / "outputs" / "animations" / "scenario3_metrics_v1"
FOURFRAME_DIR = ROOT / "4frame_view" / "scenario3_metrics_v1" / "animations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FOURFRAME_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Settings
# ============================================================

TIME_COL = "ScenarioTime"
FPS = 20
DPI = 180
FIGSIZE = (16, 7)

# If True, the animation length is based on actual ScenarioTime seconds.
# This is better for syncing with video/4-frame view.
USE_SCENARIO_TIME_DURATION = True

# ============================================================
# Load data
# ============================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input metrics CSV not found:\n{INPUT_CSV}\n\n"
        "Run scenario3_metrics_v1_fixed.py first."
    )

df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.astype(str).str.strip()

if TIME_COL not in df.columns:
    raise ValueError(
        f"Missing {TIME_COL}. Available columns are:\n{df.columns.tolist()}\n\n"
        "This animation script requires the fixed metrics CSV with ScenarioTime preserved."
    )

df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL]).copy()

df = df.sort_values(TIME_COL).reset_index(drop=True)

# Drop duplicate ScenarioTime rows for stable search/interpolation.
df = df.drop_duplicates(subset=[TIME_COL], keep="first").reset_index(drop=True)

time_min = float(df[TIME_COL].min())
time_max = float(df[TIME_COL].max())
time_range = time_max - time_min

print("Loaded:")
print(INPUT_CSV)
print("Shape:", df.shape)
print(f"{TIME_COL} min:", time_min)
print(f"{TIME_COL} max:", time_max)
print(f"{TIME_COL} range:", time_range)
print(f"{TIME_COL} unique values:", df[TIME_COL].nunique())

if time_range < 5:
    raise ValueError(
        f"{TIME_COL} range is too small: {time_range}. "
        "This means the animation would collapse around 0. "
        "Check whether the feature CSV has a broken ScenarioTime column."
    )

# Frame times based on actual ScenarioTime.
# This makes the pointer video length match the scenario duration.
if USE_SCENARIO_TIME_DURATION:
    n_frames = int(np.ceil(time_range * FPS)) + 1
    animation_times = np.linspace(time_min, time_max, n_frames)
else:
    animation_times = df[TIME_COL].to_numpy()

print("Animation frames:", len(animation_times))
print("Expected animation duration seconds:", len(animation_times) / FPS)

# ============================================================
# Helpers
# ============================================================

def save_animation(anim, fig, filename_base):
    """
    Save to both output folders.
    A new writer is created for each save so the second save does not reuse
    a consumed writer object.
    """
    saved_mp4 = True

    for out_dir in [OUTPUT_DIR, FOURFRAME_DIR]:
        out_path = out_dir / f"{filename_base}.mp4"
        try:
            writer = FFMpegWriter(fps=FPS)
            anim.save(out_path, writer=writer, dpi=DPI)
            print(f"Saved MP4: {out_path}")
        except Exception as e:
            saved_mp4 = False
            print(f"MP4 failed for {filename_base} at {out_path}: {e}")

    if not saved_mp4:
        for out_dir in [OUTPUT_DIR, FOURFRAME_DIR]:
            out_path = out_dir / f"{filename_base}.gif"
            writer = PillowWriter(fps=FPS)
            anim.save(out_path, writer=writer, dpi=DPI)
            print(f"Saved GIF: {out_path}")


def nearest_index(sorted_x, value):
    """Return index of sorted_x closest to value."""
    idx = np.searchsorted(sorted_x, value, side="left")

    if idx <= 0:
        return 0
    if idx >= len(sorted_x):
        return len(sorted_x) - 1

    before = idx - 1
    after = idx

    if abs(sorted_x[after] - value) < abs(sorted_x[before] - value):
        return after
    return before


def value_at_time(x, y, current_time):
    """
    Interpolate y at current_time when possible.
    Falls back to nearest finite point if the column has gaps.
    """
    finite = np.isfinite(x) & np.isfinite(y)

    if finite.sum() >= 2:
        return np.interp(current_time, x[finite], y[finite])

    if finite.sum() == 1:
        return y[finite][0]

    return np.nan


def animate_existing_style_plot(columns, title, ylabel, filename_base):
    existing = [c for c in columns if c in df.columns]

    if not existing:
        print(f"Skipping {filename_base}: none of these columns were found: {columns}")
        return

    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"Note for {filename_base}: missing optional columns: {missing}")

    plot_df = df[[TIME_COL] + existing].copy()

    for col in existing:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plot_df = plot_df.dropna(subset=[TIME_COL]).copy()
    plot_df = plot_df.sort_values(TIME_COL).reset_index(drop=True)
    plot_df = plot_df.drop_duplicates(subset=[TIME_COL], keep="first").reset_index(drop=True)

    x = plot_df[TIME_COL].to_numpy(dtype=float)

    if x.max() - x.min() < 5:
        raise ValueError(
            f"Bad x-axis for {filename_base}. "
            f"Time range is {x.min()} to {x.max()}."
        )

    fig, ax = plt.subplots(figsize=FIGSIZE)

    y_min = np.inf
    y_max = -np.inf

    point_artists = {}
    y_lookup = {}

    # Draw the full graph first. The pointer moves across this already-visible graph.
    for col in existing:
        y = plot_df[col].to_numpy(dtype=float)
        y_lookup[col] = y

        ax.plot(x, y, label=col, linewidth=2)

        point, = ax.plot([], [], marker="o", markersize=7, linestyle="None")
        point_artists[col] = point

        finite_y = y[np.isfinite(y)]
        if len(finite_y) > 0:
            y_min = min(y_min, finite_y.min())
            y_max = max(y_max, finite_y.max())

    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = 0, 1

    if y_min == y_max:
        y_min -= 1
        y_max += 1

    y_pad = 0.08 * (y_max - y_min)

    # Full ScenarioTime range stays visible for the entire animation.
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel("ScenarioTime")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True)

    pointer = ax.axvline(x=x[0], linewidth=2)
    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def update(frame):
        current_time = float(animation_times[frame])
        pointer.set_xdata([current_time, current_time])
        time_text.set_text(f"t = {current_time:.2f} s")

        artists = [pointer, time_text]

        for col in existing:
            y = y_lookup[col]
            current_y = value_at_time(x, y, current_time)

            if np.isfinite(current_y):
                point_artists[col].set_data([current_time], [current_y])
            else:
                point_artists[col].set_data([], [])

            artists.append(point_artists[col])

        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=len(animation_times),
        interval=1000 / FPS,
        repeat=False,
        blit=False,
    )

    plt.tight_layout()
    save_animation(anim, fig, filename_base)
    plt.close(fig)

# ============================================================
# Compatibility report
# ============================================================

expected_columns = [
    TIME_COL,
    "ped_speed_xz",
    "ped_speed_xz_smooth",
    "ped_accel_xz",
    "ped_accel_xz_smooth",
    "car_speed_xz",
    "car_speed_xz_smooth",
    "car_ped_distance_xz",
    "car_ped_distance_xz_smooth",
    "distance_change",
    "distance_closing",
    "distance_closing_smooth",
    "car_accel_xz_smooth",
]

missing_expected = [c for c in expected_columns if c not in df.columns]

if missing_expected:
    print("\nWARNING: Some expected metric columns are missing:")
    for col in missing_expected:
        print(" -", col)
    print("Some animations may be skipped.")
else:
    print("\nCompatibility check passed: required animation metric columns are present.")

# ============================================================
# Animations
# ============================================================

animate_existing_style_plot(
    ["ped_speed_xz", "ped_speed_xz_smooth"],
    "Pedestrian Speed Using B VR Position: Raw vs Smoothed",
    "Speed",
    "01_ped_speed_raw_vs_smooth_pointer",
)

animate_existing_style_plot(
    ["ped_accel_xz", "ped_accel_xz_smooth"],
    "Pedestrian Acceleration Using B VR Position: Raw vs Smoothed",
    "Acceleration",
    "02_ped_accel_raw_vs_smooth_pointer",
)

animate_existing_style_plot(
    ["car_speed_xz", "car_speed_xz_smooth"],
    "Car Speed: Raw vs Smoothed",
    "Speed",
    "03_car_speed_raw_vs_smooth_pointer",
)

animate_existing_style_plot(
    ["car_ped_distance_xz", "car_ped_distance_xz_smooth"],
    "Car-Pedestrian Distance Using B VR Position",
    "Distance",
    "04_car_ped_distance_pointer",
)

animate_existing_style_plot(
    ["distance_change", "distance_closing", "distance_closing_smooth"],
    "Distance Change and Closing Behavior",
    "Distance change",
    "05_distance_closing_pointer",
)

animate_existing_style_plot(
    ["car_accel_xz_smooth", "ped_accel_xz_smooth"],
    "Smoothed Car and Pedestrian Acceleration",
    "Acceleration",
    "06_smoothed_acceleration_comparison_pointer",
)

# Optional. This will skip cleanly if head columns do not exist.
animate_existing_style_plot(
    ["head_yaw", "head_turn_rate", "head_body_yaw_diff"],
    "Head Features",
    "Degrees / degrees per second",
    "07_head_features_pointer",
)

print("\nDone.")
print(f"Animations saved to: {OUTPUT_DIR}")
print(f"Copies saved to: {FOURFRAME_DIR}")
