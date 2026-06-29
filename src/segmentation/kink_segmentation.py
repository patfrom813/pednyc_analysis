from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(r"C:\Users\patl5\OneDrive\Desktop\BURE\pednyc_analysis")

INPUT_CSV = ROOT / "data" / "processed" / "features_PedNYC1_scenario3_metrics_v1.csv"

OUTPUT_DIR = ROOT / "outputs" / "graphs" / "scenario3_metrics_v1" / "kink_segmentation"
FOURFRAME_DIR = ROOT / "4frame_view" / "scenario3_metrics_v1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FOURFRAME_DIR.mkdir(parents=True, exist_ok=True)

KINK_POINTS_CSV = OUTPUT_DIR / "kink_points_PedNYC1_scenario3_v1.csv"
SEGMENTS_CSV = OUTPUT_DIR / "kink_segments_PedNYC1_scenario3_v1.csv"
PLOT_PATH = OUTPUT_DIR / "kink_ped_speed_segments_v1.png"
PLOT_COPY_PATH = FOURFRAME_DIR / "11_kink_ped_speed_segments_v1.png"


# ============================================================
# Settings you will tune
# ============================================================

TIME_COL = "ScenarioTime"
SPEED_COL = "ped_speed_xz_smooth"

# Movement thresholds
STOP_SPEED = 0.15
WALK_SPEED = 0.30

# Kink detection settings
TREND_SMOOTH_WINDOW_SEC = 0.35
MIN_PEAK_PROMINENCE = 0.10
MIN_KINK_GAP_SEC = 0.65
MIN_SEGMENT_DURATION_SEC = 0.75

# Slope thresholds
SLOPE_DEADBAND = 0.08
SPEEDING_UP_SLOPE = 0.12
SLOWING_DOWN_SLOPE = -0.12

# Hesitation settings
HESITATION_MIN_DURATION_SEC = 1.0
HESITATION_MIN_EXTREMA = 2
HESITATION_MIN_SIGN_FLIPS = 2
HESITATION_RATIO_THRESHOLD = 3.0
HESITATION_SPEED_STD_THRESHOLD = 0.08


# ============================================================
# Helper functions
# ============================================================

def odd_window_from_seconds(time_values, window_sec):
    dt = np.nanmedian(np.diff(time_values))
    if not np.isfinite(dt) or dt <= 0:
        return 5

    window = int(round(window_sec / dt))
    window = max(window, 3)

    if window % 2 == 0:
        window += 1

    return window


def smooth_series(values, time_values, window_sec=0.35):
    window = odd_window_from_seconds(time_values, window_sec)

    s = pd.Series(values)
    smoothed = (
        s.rolling(window=window, center=True, min_periods=1)
         .median()
         .rolling(window=window, center=True, min_periods=1)
         .mean()
    )

    return smoothed.to_numpy()


def robust_z(values):
    values = np.asarray(values, dtype=float)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))

    if mad < 1e-9:
        return np.zeros_like(values)

    return 0.6745 * (values - med) / mad


def slope_sign(values, deadband):
    signs = np.zeros_like(values, dtype=int)
    signs[values > deadband] = 1
    signs[values < -deadband] = -1
    return signs


def fill_zero_signs(signs):
    """
    Converts sign arrays like [1, 1, 0, 0, -1]
    into [1, 1, 1, 1, -1] so sign changes are easier to detect.
    """
    filled = signs.copy()

    last = 0
    for i in range(len(filled)):
        if filled[i] == 0:
            filled[i] = last
        else:
            last = filled[i]

    last = 0
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] == 0:
            filled[i] = last
        else:
            last = filled[i]

    return filled


def local_prominence(y, idx, kind, time_values, window_sec=1.0):
    t0 = time_values[idx]
    left_mask = (time_values >= t0 - window_sec) & (time_values < t0)
    right_mask = (time_values > t0) & (time_values <= t0 + window_sec)

    if left_mask.sum() < 2 or right_mask.sum() < 2:
        return 0.0

    left_vals = y[left_mask]
    right_vals = y[right_mask]

    if kind == "peak":
        left_base = np.nanmin(left_vals)
        right_base = np.nanmin(right_vals)
        return y[idx] - max(left_base, right_base)

    if kind == "trough":
        left_base = np.nanmax(left_vals)
        right_base = np.nanmax(right_vals)
        return min(left_base, right_base) - y[idx]

    return 0.0


def add_candidate(candidates, idx, time_values, reason, score):
    candidates.append({
        "idx": int(idx),
        "time": float(time_values[idx]),
        "reason": reason,
        "score": float(score),
    })


def merge_nearby_candidates(candidates, min_gap_sec):
    """
    Merge kink candidates that are too close together.
    Keeps the strongest candidate, but combines the reasons.
    """
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda x: x["time"])
    merged = []

    current_group = [candidates[0]]

    for cand in candidates[1:]:
        if cand["time"] - current_group[-1]["time"] <= min_gap_sec:
            current_group.append(cand)
        else:
            merged.append(merge_candidate_group(current_group))
            current_group = [cand]

    merged.append(merge_candidate_group(current_group))
    return merged


def merge_candidate_group(group):
    best = max(group, key=lambda x: x["score"])
    reasons = sorted(set(g["reason"] for g in group))

    return {
        "idx": best["idx"],
        "time": best["time"],
        "reason": "+".join(reasons),
        "score": best["score"],
    }


def enforce_min_segment_duration(kinks, min_duration_sec):
    """
    Prevents tiny unusable segments.
    Start and end are always kept.
    """
    if len(kinks) <= 2:
        return kinks

    kept = [kinks[0]]

    for cand in kinks[1:-1]:
        if cand["time"] - kept[-1]["time"] >= min_duration_sec:
            kept.append(cand)
        else:
            if cand["score"] > kept[-1]["score"] and kept[-1]["reason"] != "start":
                kept[-1] = cand

    if kinks[-1]["time"] - kept[-1]["time"] < min_duration_sec and len(kept) > 1:
        kept.pop()

    kept.append(kinks[-1])
    return kept


def count_local_extrema_in_segment(y, time_values, start_time, end_time):
    mask = (time_values >= start_time) & (time_values <= end_time)
    idxs = np.where(mask)[0]

    if len(idxs) < 5:
        return 0, 0

    segment_y = y[idxs]
    segment_t = time_values[idxs]
    segment_slope = np.gradient(segment_y, segment_t)

    signs = slope_sign(segment_slope, SLOPE_DEADBAND)
    signs = fill_zero_signs(signs)

    peaks = 0
    troughs = 0

    for i in range(1, len(signs)):
        if signs[i - 1] == 1 and signs[i] == -1:
            peaks += 1
        elif signs[i - 1] == -1 and signs[i] == 1:
            troughs += 1

    return peaks, troughs


def count_slope_sign_flips(y, time_values):
    if len(y) < 5:
        return 0

    slope = np.gradient(y, time_values)
    signs = slope_sign(slope, SLOPE_DEADBAND)
    signs = fill_zero_signs(signs)

    flips = 0
    for i in range(1, len(signs)):
        if signs[i - 1] != 0 and signs[i] != 0 and signs[i - 1] != signs[i]:
            flips += 1

    return flips


def segment_label(features):
    mean_speed = features["mean_speed"]
    stopped_fraction = features["stopped_fraction"]
    net_slope = features["net_slope"]
    duration = features["duration_sec"]
    n_extrema = features["n_local_peaks"] + features["n_local_troughs"]
    n_sign_flips = features["n_slope_sign_flips"]
    hesitation_ratio = features["hesitation_ratio"]
    speed_std = features["speed_std"]

    is_near_stopped = stopped_fraction >= 0.60 or mean_speed < STOP_SPEED
    is_moving = mean_speed >= WALK_SPEED

    is_hesitation_like = (
        duration >= HESITATION_MIN_DURATION_SEC
        and is_moving
        and (
            n_extrema >= HESITATION_MIN_EXTREMA
            or n_sign_flips >= HESITATION_MIN_SIGN_FLIPS
        )
        and hesitation_ratio >= HESITATION_RATIO_THRESHOLD
        and speed_std >= HESITATION_SPEED_STD_THRESHOLD
    )

    if is_near_stopped:
        return "near_stopped"

    if is_hesitation_like:
        if net_slope <= SLOWING_DOWN_SLOPE:
            return "hesitating_slowdown"
        elif net_slope >= SPEEDING_UP_SLOPE:
            return "hesitating_speedup"
        else:
            return "hesitating_walk"

    if net_slope >= SPEEDING_UP_SLOPE:
        return "speeding_up"

    if net_slope <= SLOWING_DOWN_SLOPE:
        return "slowing_down"

    if mean_speed >= WALK_SPEED:
        return "steady_walking"

    return "uncertain_low_speed_motion"


# ============================================================
# Main logic
# ============================================================

def main():
    df = pd.read_csv(INPUT_CSV)

# Clean column names because some CSV headers have extra spaces
    df.columns = df.columns.astype(str).str.strip()

    required_cols = [TIME_COL, SPEED_COL]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print("\nMissing required columns:")
        for col in missing:
            print(f"  - {col}")

        print("\nAvailable columns are:")
        for col in df.columns:
            print(f"  - {col}")

        raise ValueError("Required columns missing. Check CSV column names.")

    df = df.copy()
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    time_values = df[TIME_COL].to_numpy(dtype=float)
    speed_raw = df[SPEED_COL].to_numpy(dtype=float)

    valid = np.isfinite(time_values) & np.isfinite(speed_raw)
    df = df.loc[valid].reset_index(drop=True)

    time_values = df[TIME_COL].to_numpy(dtype=float)
    speed_raw = df[SPEED_COL].to_numpy(dtype=float)

    # Extra smoothing for kink detection only.
    # We still plot the original ped_speed_xz_smooth.
    speed_trend = smooth_series(
        speed_raw,
        time_values,
        window_sec=TREND_SMOOTH_WINDOW_SEC
    )

    slope = np.gradient(speed_trend, time_values)
    curvature = np.gradient(slope, time_values)

    slope_z = np.abs(robust_z(slope))
    curvature_z = np.abs(robust_z(curvature))
    kink_strength = 0.50 * slope_z + 0.50 * curvature_z

    candidates = []

    # ------------------------------------------------------------
    # 1. Start/end boundaries
    # ------------------------------------------------------------
    add_candidate(candidates, 0, time_values, "start", 999.0)
    add_candidate(candidates, len(df) - 1, time_values, "end", 999.0)

    # ------------------------------------------------------------
    # 2. Stopped to moving / moving to stopped crossings
    # ------------------------------------------------------------
    moving = speed_trend >= WALK_SPEED

    for i in range(1, len(moving)):
        if not moving[i - 1] and moving[i]:
            add_candidate(
                candidates,
                i,
                time_values,
                "stopped_to_moving_crossing",
                8.0
            )

        elif moving[i - 1] and not moving[i]:
            add_candidate(
                candidates,
                i,
                time_values,
                "moving_to_stopped_crossing",
                8.0
            )

    # ------------------------------------------------------------
    # 3. Local peaks and troughs from slope sign changes
    # ------------------------------------------------------------
    signs = slope_sign(slope, SLOPE_DEADBAND)
    signs = fill_zero_signs(signs)

    for i in range(1, len(signs)):
        prev_sign = signs[i - 1]
        curr_sign = signs[i]

        if prev_sign == 1 and curr_sign == -1:
            prom = local_prominence(
                speed_trend,
                i,
                "peak",
                time_values,
                window_sec=1.0
            )

            if prom >= MIN_PEAK_PROMINENCE:
                add_candidate(
                    candidates,
                    i,
                    time_values,
                    "local_speed_peak",
                    5.0 + prom
                )

        elif prev_sign == -1 and curr_sign == 1:
            prom = local_prominence(
                speed_trend,
                i,
                "trough",
                time_values,
                window_sec=1.0
            )

            if prom >= MIN_PEAK_PROMINENCE:
                add_candidate(
                    candidates,
                    i,
                    time_values,
                    "local_speed_trough",
                    5.0 + prom
                )

    # ------------------------------------------------------------
    # 4. Sharp curvature kinks
    # ------------------------------------------------------------
    curvature_threshold = np.nanpercentile(kink_strength, 90)

    for i in range(1, len(kink_strength) - 1):
        is_local_max = (
            kink_strength[i] >= kink_strength[i - 1]
            and kink_strength[i] >= kink_strength[i + 1]
        )

        if is_local_max and kink_strength[i] >= curvature_threshold:
            add_candidate(
                candidates,
                i,
                time_values,
                "sharp_slope_curvature_change",
                float(kink_strength[i])
            )

    # ------------------------------------------------------------
    # 5. Merge nearby candidates
    # ------------------------------------------------------------
    merged = merge_nearby_candidates(candidates, MIN_KINK_GAP_SEC)

    # Make sure start and end remain exact
    merged = sorted(merged, key=lambda x: x["time"])

    # Replace first and last with true boundaries
    merged[0] = {
        "idx": 0,
        "time": float(time_values[0]),
        "reason": "start",
        "score": 999.0
    }

    merged[-1] = {
        "idx": len(df) - 1,
        "time": float(time_values[-1]),
        "reason": "end",
        "score": 999.0
    }

    final_kinks = enforce_min_segment_duration(
        merged,
        MIN_SEGMENT_DURATION_SEC
    )

    kink_df = pd.DataFrame(final_kinks)
    kink_df.to_csv(KINK_POINTS_CSV, index=False)

    # ------------------------------------------------------------
    # 6. Build segments from kink boundaries
    # ------------------------------------------------------------
    segment_rows = []

    for seg_id in range(len(final_kinks) - 1):
        start = final_kinks[seg_id]
        end = final_kinks[seg_id + 1]

        start_idx = start["idx"]
        end_idx = end["idx"]

        if end_idx <= start_idx:
            continue

        seg_df = df.iloc[start_idx:end_idx + 1].copy()

        seg_t = seg_df[TIME_COL].to_numpy(dtype=float)
        seg_y = seg_df[SPEED_COL].to_numpy(dtype=float)

        duration = float(seg_t[-1] - seg_t[0])

        if duration <= 0:
            continue

        mean_speed = float(np.nanmean(seg_y))
        median_speed = float(np.nanmedian(seg_y))
        speed_std = float(np.nanstd(seg_y))
        min_speed = float(np.nanmin(seg_y))
        max_speed = float(np.nanmax(seg_y))
        speed_range = float(max_speed - min_speed)

        start_speed = float(seg_y[0])
        end_speed = float(seg_y[-1])
        net_change = float(end_speed - start_speed)

        # Linear trend slope inside the segment
        x = seg_t - seg_t[0]
        if len(seg_y) >= 2:
            net_slope = float(np.polyfit(x, seg_y, 1)[0])
        else:
            net_slope = 0.0

        total_abs_speed_change = float(np.nansum(np.abs(np.diff(seg_y))))
        hesitation_ratio = float(
            total_abs_speed_change / (abs(net_change) + 0.05)
        )

        stopped_fraction = float(np.mean(seg_y < STOP_SPEED))

        n_peaks, n_troughs = count_local_extrema_in_segment(
            speed_trend,
            time_values,
            start["time"],
            end["time"]
        )

        n_sign_flips = count_slope_sign_flips(seg_y, seg_t)

        features = {
            "segment_id": seg_id,
            "ScenarioTime_start": float(seg_t[0]),
            "ScenarioTime_end": float(seg_t[-1]),
            "duration_sec": duration,
            "start_speed": start_speed,
            "end_speed": end_speed,
            "mean_speed": mean_speed,
            "median_speed": median_speed,
            "speed_std": speed_std,
            "min_speed": min_speed,
            "max_speed": max_speed,
            "speed_range": speed_range,
            "net_change": net_change,
            "net_slope": net_slope,
            "total_abs_speed_change": total_abs_speed_change,
            "hesitation_ratio": hesitation_ratio,
            "stopped_fraction": stopped_fraction,
            "n_local_peaks": int(n_peaks),
            "n_local_troughs": int(n_troughs),
            "n_slope_sign_flips": int(n_sign_flips),
            "start_kink_reason": start["reason"],
            "end_kink_reason": end["reason"],
        }

        features["behavior_label"] = segment_label(features)

        segment_rows.append(features)

    segments_df = pd.DataFrame(segment_rows)
    segments_df.to_csv(SEGMENTS_CSV, index=False)

    # ------------------------------------------------------------
    # 7. Plot colored graph
    # ------------------------------------------------------------
    label_colors = {
        "near_stopped": "#f4cccc",
        "speeding_up": "#d9ead3",
        "steady_walking": "#cfe2f3",
        "slowing_down": "#fce5cd",
        "hesitating_walk": "#fff2cc",
        "hesitating_speedup": "#d9ead3",
        "hesitating_slowdown": "#fff2cc",
        "uncertain_low_speed_motion": "#eeeeee",
    }

    fig, ax = plt.subplots(figsize=(18, 8))

    for _, row in segments_df.iterrows():
        label = row["behavior_label"]
        color = label_colors.get(label, "#eeeeee")

        ax.axvspan(
            row["ScenarioTime_start"],
            row["ScenarioTime_end"],
            color=color,
            alpha=0.35
        )

    ax.plot(
        time_values,
        speed_raw,
        linewidth=2,
        label="ped_speed_xz_smooth"
    )

    ax.plot(
        time_values,
        speed_trend,
        linewidth=1.5,
        linestyle="--",
        label="extra trend smoothing for kink detection"
    )

    for _, row in kink_df.iterrows():
        ax.axvline(
            row["time"],
            linestyle=":",
            linewidth=1.2,
            alpha=0.8
        )

        if row["reason"] not in ["start", "end"]:
            ax.text(
                row["time"],
                ax.get_ylim()[1] * 0.95,
                row["reason"],
                rotation=90,
                va="top",
                ha="right",
                fontsize=8
            )

    ax.set_title("Pedestrian Speed Kink-Based Segmentation")
    ax.set_xlabel("ScenarioTime")
    ax.set_ylabel("Pedestrian speed XZ smooth")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.savefig(PLOT_COPY_PATH, dpi=200)
    plt.close()

    print("\nDone.")
    print(f"Saved kink points CSV to:\n  {KINK_POINTS_CSV}")
    print(f"Saved segment CSV to:\n  {SEGMENTS_CSV}")
    print(f"Saved plot to:\n  {PLOT_PATH}")
    print(f"Saved plot copy to:\n  {PLOT_COPY_PATH}")

    print("\nSegment labels:")
    print(segments_df[[
        "segment_id",
        "ScenarioTime_start",
        "ScenarioTime_end",
        "duration_sec",
        "behavior_label",
        "mean_speed",
        "net_slope",
        "n_local_peaks",
        "n_local_troughs",
        "hesitation_ratio",
        "start_kink_reason",
        "end_kink_reason",
    ]])


if __name__ == "__main__":
    main()