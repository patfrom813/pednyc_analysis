from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


# ============================================================
# Helpers
# ============================================================

def parse_columns(values):
    """
    Allows both:
    --y car_speed_xz ped_speed_xz
    and:
    --y car_speed_xz,ped_speed_xz
    """
    cols = []
    for value in values:
        parts = [p.strip() for p in value.split(",")]
        cols.extend([p for p in parts if p])
    return cols


def clean_column_names(df):
    df.columns = (
        df.columns.astype(str)
        .str.replace('"', '', regex=False)
        .str.replace("'", "", regex=False)
        .str.strip()
    )
    return df


def read_csv_auto(path, sep):
    if sep == "auto":
        return pd.read_csv(path, sep=None, engine="python", index_col=False)
    return pd.read_csv(path, sep=sep, index_col=False)


def normalize_array(arr):
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)

    if np.isclose(arr_max, arr_min):
        return np.zeros_like(arr)

    return (arr - arr_min) / (arr_max - arr_min)


# ============================================================
# Main animation function
# ============================================================

def make_animation(
    input_file,
    output_file,
    y_columns,
    time_column="ScenarioTime",
    sep="auto",
    fps=30,
    mode="smooth",
    layout="overlay",
    normalize=False,
    title=None,
    width=960,
    height=540,
    dpi=100
):
    input_file = Path(input_file)
    output_file = Path(output_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load CSV
    # ----------------------------

    df = read_csv_auto(input_file, sep)
    df = clean_column_names(df)

    missing = []

    if time_column not in df.columns:
        missing.append(time_column)

    for col in y_columns:
        if col not in df.columns:
            missing.append(col)

    if missing:
        print("\nMissing columns:")
        for col in missing:
            print(f"  - {col}")

        print("\nAvailable columns:")
        for col in df.columns:
            print(f"  - {col}")

        raise ValueError("Some requested columns were not found.")

    # Convert time and y columns to numeric
    df[time_column] = pd.to_numeric(df[time_column], errors="coerce")

    for col in y_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[time_column]).copy()
    df = df.sort_values(time_column).reset_index(drop=True)

    time_raw = df[time_column].to_numpy(dtype=float)

    t_min = np.nanmin(time_raw)
    t_max = np.nanmax(time_raw)

    # ----------------------------
    # Animation frame times
    # ----------------------------

    if mode == "every_second":
        frame_times = np.arange(t_min, t_max + 0.001, 1.0)
        output_fps = 1
    else:
        frame_times = np.arange(t_min, t_max + (1 / fps), 1 / fps)
        output_fps = fps

    # ----------------------------
    # Interpolate each y-column
    # ----------------------------

    series_data = {}

    for col in y_columns:
        y_raw = df[col].to_numpy(dtype=float)

        valid = np.isfinite(time_raw) & np.isfinite(y_raw)

        if valid.sum() < 2:
            raise ValueError(f"Column '{col}' does not have enough valid numeric data.")

        t_valid = time_raw[valid]
        y_valid = y_raw[valid]

        # Sort again just for safety
        order = np.argsort(t_valid)
        t_valid = t_valid[order]
        y_valid = y_valid[order]

        y_interp_raw = np.interp(frame_times, t_valid, y_valid)

        if normalize:
            y_plot = normalize_array(y_valid)
            y_interp_plot = normalize_array(y_interp_raw)
            ylabel = "Normalized value"
        else:
            y_plot = y_valid
            y_interp_plot = y_interp_raw
            ylabel = "Value"

        series_data[col] = {
            "time": t_valid,
            "y_raw": y_valid,
            "y_plot": y_plot,
            "y_interp_raw": y_interp_raw,
            "y_interp_plot": y_interp_plot,
            "ylabel": ylabel
        }

    # ----------------------------
    # Create figure
    # ----------------------------

    figsize = (width / dpi, height / dpi)

    if layout == "subplots":
        fig, axes = plt.subplots(
            len(y_columns),
            1,
            figsize=figsize,
            dpi=dpi,
            sharex=True
        )

        if len(y_columns) == 1:
            axes = [axes]

    else:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        axes = [ax]

    if title is None:
        if len(y_columns) == 1:
            title = f"{y_columns[0]} Over Time"
        else:
            title = "Selected Features Over Time"

    fig.suptitle(title, fontsize=16)

    artists = {}

    # ----------------------------
    # Draw plots
    # ----------------------------

    if layout == "overlay":
        ax = axes[0]

        for col in y_columns:
            data = series_data[col]

            full_line, = ax.plot(
                data["time"],
                data["y_plot"],
                linewidth=2,
                label=col
            )

            trace_line, = ax.plot(
                [],
                [],
                linewidth=4,
                alpha=0.7
            )

            pointer_dot, = ax.plot(
                [],
                [],
                marker="o",
                markersize=9,
                linestyle="None"
            )

            artists[col] = {
                "trace_line": trace_line,
                "pointer_dot": pointer_dot
            }

        pointer_line = ax.axvline(t_min, linestyle="--", linewidth=2)

        text_box = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

        ax.set_xlabel(time_column)
        ax.set_ylabel("Normalized value" if normalize else "Value")
        ax.grid(True)
        ax.legend(loc="upper right")

        axes_objects = {
            "pointer_line": pointer_line,
            "text_box": text_box
        }

    else:
        axes_objects = {}

        for ax, col in zip(axes, y_columns):
            data = series_data[col]

            ax.plot(
                data["time"],
                data["y_plot"],
                linewidth=2,
                label=col
            )

            trace_line, = ax.plot(
                [],
                [],
                linewidth=4,
                alpha=0.7
            )

            pointer_dot, = ax.plot(
                [],
                [],
                marker="o",
                markersize=8,
                linestyle="None"
            )

            pointer_line = ax.axvline(t_min, linestyle="--", linewidth=2)

            text_box = ax.text(
                0.02,
                0.88,
                "",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
            )

            ax.set_ylabel("Normalized" if normalize else col)
            ax.grid(True)
            ax.legend(loc="upper right")

            artists[col] = {
                "trace_line": trace_line,
                "pointer_dot": pointer_dot,
                "pointer_line": pointer_line,
                "text_box": text_box
            }

        axes[-1].set_xlabel(time_column)

    for ax in axes:
        ax.set_xlim(t_min, t_max)

    plt.tight_layout()

    # ----------------------------
    # Update function
    # ----------------------------

    def update(frame_idx):
        current_time = frame_times[frame_idx]

        updated_artists = []

        if layout == "overlay":
            text_lines = [f"{time_column}: {current_time:.2f} s"]

            for col in y_columns:
                data = series_data[col]

                current_y_plot = data["y_interp_plot"][frame_idx]
                current_y_raw = data["y_interp_raw"][frame_idx]

                artists[col]["pointer_dot"].set_data(
                    [current_time],
                    [current_y_plot]
                )

                mask = data["time"] <= current_time

                artists[col]["trace_line"].set_data(
                    data["time"][mask],
                    data["y_plot"][mask]
                )

                text_lines.append(f"{col}: {current_y_raw:.3f}")

                updated_artists.extend([
                    artists[col]["pointer_dot"],
                    artists[col]["trace_line"]
                ])

            axes_objects["pointer_line"].set_xdata([current_time, current_time])
            axes_objects["text_box"].set_text("\n".join(text_lines))

            updated_artists.extend([
                axes_objects["pointer_line"],
                axes_objects["text_box"]
            ])

        else:
            for col in y_columns:
                data = series_data[col]

                current_y_plot = data["y_interp_plot"][frame_idx]
                current_y_raw = data["y_interp_raw"][frame_idx]

                artists[col]["pointer_dot"].set_data(
                    [current_time],
                    [current_y_plot]
                )

                artists[col]["pointer_line"].set_xdata(
                    [current_time, current_time]
                )

                mask = data["time"] <= current_time

                artists[col]["trace_line"].set_data(
                    data["time"][mask],
                    data["y_plot"][mask]
                )

                artists[col]["text_box"].set_text(
                    f"{time_column}: {current_time:.2f} s\n"
                    f"{col}: {current_y_raw:.3f}"
                )

                updated_artists.extend([
                    artists[col]["pointer_dot"],
                    artists[col]["pointer_line"],
                    artists[col]["trace_line"],
                    artists[col]["text_box"]
                ])

        return updated_artists

    # ----------------------------
    # Save animation
    # ----------------------------

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_times),
        interval=1000 / output_fps,
        blit=False
    )

    writer = FFMpegWriter(fps=output_fps, bitrate=3000)

    anim.save(output_file, writer=writer)

    plt.close(fig)

    print(f"\nSaved animation:")
    print(output_file)


# ============================================================
# Terminal interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create an animated graph pointer video from a feature CSV."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output MP4 file."
    )

    parser.add_argument(
        "--y",
        nargs="+",
        required=True,
        help="Y-column or columns to plot. Example: --y car_speed_xz ped_speed_xz"
    )

    parser.add_argument(
        "--time",
        default="ScenarioTime",
        help="Time column for x-axis. Default: ScenarioTime"
    )

    parser.add_argument(
        "--sep",
        default="auto",
        help="CSV separator. Use auto, comma, semicolon, or actual separator. Default: auto"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for smooth mode. Default: 30"
    )

    parser.add_argument(
        "--mode",
        choices=["smooth", "every_second"],
        default="smooth",
        help="smooth = moving pointer; every_second = jumps once per second."
    )

    parser.add_argument(
        "--layout",
        choices=["overlay", "subplots"],
        default="overlay",
        help="overlay = all variables on one graph; subplots = separate graph per variable."
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each variable to 0-1 scale. Useful when plotting different units together."
    )

    parser.add_argument(
        "--title",
        default=None,
        help="Custom graph title."
    )

    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Output video width in pixels. Default: 960"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=540,
        help="Output video height in pixels. Default: 540"
    )

    args = parser.parse_args()

    sep_map = {
        "comma": ",",
        "semicolon": ";",
        "tab": "\t",
        "auto": "auto"
    }

    sep = sep_map.get(args.sep, args.sep)

    y_columns = parse_columns(args.y)

    make_animation(
        input_file=args.input,
        output_file=args.output,
        y_columns=y_columns,
        time_column=args.time,
        sep=sep,
        fps=args.fps,
        mode=args.mode,
        layout=args.layout,
        normalize=args.normalize,
        title=args.title,
        width=args.width,
        height=args.height
    )


if __name__ == "__main__":
    main()