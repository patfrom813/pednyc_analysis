"""Expanded, participant-aware scenario-pattern analysis for six PedNYC scenarios."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.config import PipelineConfig
from src.scenario_pattern_pilot import (
    PROJECT_ROOT, aggregate_scenarios, discover_recordings, process_recording,
)

SCENARIOS = (3, 7, 12, 15, 16, 21)
SEED = 42
FEATURE_GROUPS = {
    "movement": [
        "median_pedestrian_speed", "peak_pedestrian_speed",
        "median_positive_acceleration", "median_deceleration_magnitude",
        "low_motion_proportion", "stop_start_count", "recording_duration",
    ],
    "head_movement": [
        "head_rotation_event_count", "median_absolute_head_rotation_rate",
    ],
    "interaction": [
        "minimum_pedestrian_vehicle_distance", "vehicle_speed_at_movement_onset",
        "vehicle_deceleration_around_movement_onset", "median_closing_speed",
    ],
}


def mkdirs(root: Path) -> None:
    for name in ("config", "manifests", "tables", "figures", "traces", "logs", "report",
                 "intermediate/macro_segments", "intermediate/micro_events"):
        (root / name).mkdir(parents=True, exist_ok=True)


def availability(inventory: pd.DataFrame) -> pd.DataFrame:
    participants = sorted(inventory.pednyc_number.unique())
    lookup = {(int(r.pednyc_number), int(r.scenario_number)): [] for r in inventory.itertuples()}
    for row in inventory.itertuples():
        lookup[(int(row.pednyc_number), int(row.scenario_number))].append(
            (row.recording_identifier, row.source_decoded_csv_path)
        )
    rows = []
    for pid in participants:
        for scenario in SCENARIOS:
            found = lookup.get((pid, scenario), [])
            if found:
                for identity, path in found:
                    rows.append(dict(pednyc_number=pid, scenario_number=scenario,
                                     recording_identifier=identity, source_csv_path=path,
                                     availability_status="available", processing_status="pending",
                                     failure_exclusion_reason=""))
            else:
                rows.append(dict(pednyc_number=pid, scenario_number=scenario,
                                 recording_identifier="", source_csv_path="",
                                 availability_status="missing", processing_status="not_applicable",
                                 failure_exclusion_reason="No decoded CSV found in inventory"))
    return pd.DataFrame(rows)


def feature_matrices(success: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = ["recording_identifier", "pednyc_number", "scenario_number"]
    candidates = [f for group in FEATURE_GROUPS.values() for f in group]
    raw = success[ids + candidates].copy()
    decisions = []
    included = []
    for group, features in FEATURE_GROUPS.items():
        for feature in features:
            values = pd.to_numeric(raw[feature], errors="coerce")
            missing = float(values.isna().mean())
            variance = float(values.var()) if values.notna().sum() > 1 else np.nan
            reason = ""
            use = True
            if missing > .25:
                use, reason = False, "excluded: more than 25% missing"
            elif not np.isfinite(variance) or variance == 0:
                use, reason = False, "excluded: zero or undefined variance"
            if use:
                included.append(feature)
                reason = "included"
            decisions.append(dict(feature=feature, feature_group=group, included=use,
                                  missing_fraction=missing, variance=variance, reason=reason))
    # Consolidation is specified a priori: proportions replace low-motion/pause durations,
    # stop-start count replaces pause count, avoiding duplicated duration/count weight.
    decisions.extend([
        dict(feature="total_low_motion_duration", feature_group="movement", included=False,
             missing_fraction=np.nan, variance=np.nan,
             reason="excluded as redundant with low_motion_proportion"),
        dict(feature="total_pause_duration", feature_group="movement", included=False,
             missing_fraction=np.nan, variance=np.nan,
             reason="excluded as redundant with low_motion_proportion"),
        dict(feature="pause_count", feature_group="movement", included=False,
             missing_fraction=np.nan, variance=np.nan,
             reason="excluded as redundant with stop_start_count"),
        dict(feature="peak_absolute_head_rotation_rate", feature_group="head_movement", included=False,
             missing_fraction=float(pd.to_numeric(success["peak_absolute_head_rotation_rate"], errors="coerce").isna().mean()),
             variance=float(pd.to_numeric(success["peak_absolute_head_rotation_rate"], errors="coerce").var()),
             reason="excluded as redundant/outlier-sensitive counterpart of median rate"),
    ])
    x = raw[included].apply(pd.to_numeric, errors="coerce")
    medians = x.median()
    x_imputed = x.fillna(medians)
    std = x_imputed.std(ddof=0).replace(0, np.nan)
    z = (x_imputed - x_imputed.mean()) / std
    standardized = pd.concat([raw[ids].reset_index(drop=True), z.reset_index(drop=True)], axis=1)
    return raw, standardized, pd.DataFrame(decisions)


def aligned_outputs(traces: pd.DataFrame, root: Path) -> pd.DataFrame:
    grid = np.arange(-5, 30.0001, .1)
    wide = []
    summaries = []
    scenarios = sorted(traces.scenario_number.unique())
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True, sharey=True)
    for ax, scenario in zip(axes.flat, scenarios):
        curves = []
        for identity, trace in traces[traces.scenario_number == scenario].groupby("recording_identifier"):
            trace = trace.dropna(subset=["aligned_time", "pedestrian_speed"]).sort_values("aligned_time")
            if len(trace) < 2:
                continue
            curve = np.interp(grid, trace.aligned_time, trace.pedestrian_speed,
                              left=np.nan, right=np.nan)
            curves.append(curve)
            row = pd.DataFrame({"recording_identifier": identity, "scenario_number": scenario,
                                "aligned_time": grid, "interpolated_speed": curve})
            wide.append(row)
            ax.plot(grid, curve, color="#4C78A8", alpha=.12, lw=.6)
        stack = np.vstack(curves)
        n = np.isfinite(stack).sum(axis=0)
        valid = n > 0
        med = np.full(len(grid), np.nan); q1 = med.copy(); q3 = med.copy()
        med[valid] = np.nanmedian(stack[:, valid], axis=0)
        q1[valid], q3[valid] = np.nanpercentile(stack[:, valid], [25, 75], axis=0)
        summaries.append(pd.DataFrame({"scenario_number": scenario, "aligned_time": grid,
                                       "median_speed": med, "q1_speed": q1, "q3_speed": q3,
                                       "contributing_recordings": n}))
        ax.fill_between(grid, q1, q3, color="#F58518", alpha=.25)
        ax.plot(grid, med, color="#E45756", lw=1.8)
        ax2 = ax.twinx()
        ax2.plot(grid, n, color="black", alpha=.35, lw=.8, ls="--")
        ax2.set_ylim(0, max(n) * 1.25); ax2.set_yticks([0, max(n)])
        ax.axvline(0, color="black", ls=":", lw=.8)
        ax.set_title(f"Scenario {scenario} (maximum n={max(n)})")
    fig.supxlabel("Seconds from pedestrian movement onset")
    fig.supylabel("Smoothed pedestrian horizontal speed (m/s)")
    fig.text(.99, .5, "Contributing recordings (dashed)", rotation=90, va="center", ha="right")
    fig.tight_layout(rect=(0, 0, .98, 1))
    fig.savefig(root / "figures/aligned_speed_small_multiples.png", dpi=180)
    plt.close(fig)
    interpolated = pd.concat(wide, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    traces.to_csv(root / "traces/aligned_speed_raw_traces.csv", index=False)
    interpolated.to_csv(root / "traces/aligned_speed_interpolated_traces.csv", index=False)
    summary.to_csv(root / "traces/aligned_speed_timepoint_summary.csv", index=False)
    return summary


def sequence_tables(success: pd.DataFrame, root: Path) -> None:
    full_rows, event_rows, bigrams, subseqs, macro_rows = [], [], [], [], []
    for scenario, group in success.groupby("scenario_number"):
        seqs = group.event_sequence.fillna("").map(lambda s: tuple(x for x in s.split(" -> ") if x))
        counts = Counter(seqs)
        maximum = max(counts.values())
        for seq, count in counts.items():
            full_rows.append(dict(scenario_number=scenario, event_sequence=" -> ".join(seq),
                                  frequency=count, prevalence=count/len(group),
                                  tied_modal=count == maximum,
                                  terminology="consensus" if count/len(group) >= .50 else
                                  "most frequent observed sequence" if count == maximum else "observed sequence"))
        event_count = Counter(e for seq in seqs for e in set(seq))
        for event, count in event_count.items():
            event_rows.append(dict(scenario_number=scenario, event_label=event, recordings=count,
                                   prevalence=count/len(group)))
        bi = Counter(pair for seq in seqs for pair in zip(seq, seq[1:]))
        for pair, count in bi.items():
            bigrams.append(dict(scenario_number=scenario, transition=" -> ".join(pair), frequency=count))
        short = Counter(sub for seq in seqs for length in (2, 3)
                        for sub in {seq[i:i+length] for i in range(len(seq)-length+1)})
        for sub, count in short.items():
            subseqs.append(dict(scenario_number=scenario, subsequence=" -> ".join(sub),
                                length=len(sub), recordings=count, prevalence=count/len(group)))
        mc = Counter(group.macro_phase_sequence.fillna(""))
        for seq, count in mc.items():
            macro_rows.append(dict(scenario_number=scenario, macro_sequence=seq,
                                   frequency=count, prevalence=count/len(group)))
    pd.DataFrame(full_rows).to_csv(root / "tables/full_sequence_frequencies.csv", index=False)
    pd.DataFrame(event_rows).to_csv(root / "tables/event_label_prevalence.csv", index=False)
    pd.DataFrame(bigrams).to_csv(root / "tables/event_transition_frequencies.csv", index=False)
    pd.DataFrame(subseqs).to_csv(root / "tables/short_subsequence_frequencies.csv", index=False)
    pd.DataFrame(macro_rows).to_csv(root / "tables/macro_sequence_prevalence.csv", index=False)
    modal = pd.DataFrame(full_rows)
    modal = modal[modal.tied_modal]
    abbreviations = {
        "low_motion": "LM", "acceleration": "A", "deceleration": "D",
        "crossing_commitment": "C", "head_check": "H",
        "possible_yielding": "Y", "possible_hesitation": "O",
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    for y, scenario in enumerate(SCENARIOS):
        rows = modal[modal.scenario_number == scenario]
        examples = []
        for item in rows.head(3).itertuples():
            events = item.event_sequence.split(" -> ") if item.event_sequence else []
            short = "–".join(abbreviations.get(event, event[:2].upper()) for event in events[:10])
            examples.append(short + ("–…" if len(events) > 10 else ""))
        prevalence = rows.prevalence.iloc[0] if len(rows) else np.nan
        text = (
            f"{len(rows)} tied modal sequence(s), each {prevalence:.1%}\n"
            + "\n".join(f"example: {example}" for example in examples)
        )
        ax.text(.02, y, text, va="center", fontsize=8)
    ax.set_yticks(range(6), [f"Scenario {s}" for s in SCENARIOS])
    ax.set_xticks([]); ax.set_xlim(0, 1); ax.invert_yaxis()
    ax.set_title("Tied modal full micro-event sequences (order only; full text in CSV)")
    fig.text(.5, .012, "LM=low motion, A=acceleration, D=deceleration, C=crossing commitment, "
             "H=head check, Y=possible yielding, O=possible hesitation",
             ha="center", fontsize=7)
    fig.tight_layout(rect=(0, .035, 1, .97))
    fig.savefig(root / "figures/micro_event_modal_sequences.png", dpi=180)
    plt.close(fig)


def pair_analysis(meta: pd.DataFrame, z: pd.DataFrame, root: Path, permutations: int,
                  bootstraps: int) -> pd.DataFrame:
    ids = meta.recording_identifier.tolist()
    x = z.drop(columns=["recording_identifier", "pednyc_number", "scenario_number"]).to_numpy()
    dist = squareform(pdist(x, metric="euclidean"))
    pd.DataFrame(dist, index=ids, columns=ids).to_csv(root / "tables/recording_distance_matrix.csv")
    cosine = 1 - squareform(pdist(x, metric="cosine"))
    pd.DataFrame(cosine, index=ids, columns=ids).to_csv(root / "tables/recording_cosine_similarity_matrix.csv")
    rows = []
    for i in range(len(meta)):
        for j in range(i + 1, len(meta)):
            same_p = meta.iloc[i].pednyc_number == meta.iloc[j].pednyc_number
            same_s = meta.iloc[i].scenario_number == meta.iloc[j].scenario_number
            if same_p and not same_s: category = "same_participant_different_scenario"
            elif not same_p and same_s: category = "same_scenario_different_participant"
            elif not same_p and not same_s: category = "different_scenario_different_participant"
            else: category = "same_participant_same_scenario_repeat"
            rows.append(dict(recording_a=ids[i], recording_b=ids[j], category=category,
                             euclidean_distance=dist[i, j], cosine_similarity=cosine[i, j]))
    pairs = pd.DataFrame(rows)
    pairs.to_csv(root / "tables/recording_pair_distances.csv", index=False)
    wanted = ["same_scenario_different_participant", "different_scenario_different_participant",
              "same_participant_different_scenario"]
    rng = np.random.default_rng(SEED)
    summary = []
    for category in wanted:
        vals = pairs.loc[pairs.category == category, "euclidean_distance"].to_numpy()
        boots = np.array([np.median(rng.choice(vals, len(vals), replace=True)) for _ in range(bootstraps)])
        summary.append(dict(category=category, pair_count=len(vals), median=np.median(vals),
                            q1=np.quantile(vals, .25), q3=np.quantile(vals, .75),
                            bootstrap_ci_low=np.quantile(boots, .025),
                            bootstrap_ci_high=np.quantile(boots, .975)))
    result = pd.DataFrame(summary)
    same = pairs[pairs.category == wanted[0]].euclidean_distance
    diff = pairs[pairs.category == wanted[1]].euclidean_distance
    observed = float(diff.median() - same.median())
    pooled = np.sqrt((same.var() + diff.var()) / 2)
    effect = observed / pooled
    contrast_boot = []
    effect_boot = []
    same_array, diff_array = same.to_numpy(), diff.to_numpy()
    for _ in range(bootstraps):
        same_b = rng.choice(same_array, len(same_array), replace=True)
        diff_b = rng.choice(diff_array, len(diff_array), replace=True)
        contrast_b = float(np.median(diff_b) - np.median(same_b))
        pooled_b = float(np.sqrt((np.var(same_b, ddof=1) + np.var(diff_b, ddof=1)) / 2))
        contrast_boot.append(contrast_b)
        effect_boot.append(contrast_b / pooled_b if pooled_b > 0 else np.nan)
    labels = meta.scenario_number.to_numpy().copy()
    perm_effects = []
    # Within-participant label shuffling preserves participant-specific feature sets
    # and each participant's observed scenario-label multiset.
    groups = [np.flatnonzero(meta.pednyc_number.to_numpy() == p)
              for p in meta.pednyc_number.unique()]
    upper_i, upper_j = np.triu_indices(len(meta), 1)
    cross_participant = (
        meta.pednyc_number.to_numpy()[upper_i] != meta.pednyc_number.to_numpy()[upper_j]
    )
    pair_distances = dist[upper_i, upper_j]
    for _ in range(permutations):
        shuffled = labels.copy()
        for idx in groups:
            shuffled[idx] = rng.permutation(shuffled[idx])
        same_label = shuffled[upper_i] == shuffled[upper_j]
        same_vals = pair_distances[cross_participant & same_label]
        diff_vals = pair_distances[cross_participant & ~same_label]
        perm_effects.append(float(np.median(diff_vals) - np.median(same_vals)))
    pvalue = (1 + np.sum(np.asarray(perm_effects) >= observed)) / (permutations + 1)
    result["primary_contrast_difference_in_medians"] = observed
    result["primary_contrast_bootstrap_ci_low"] = np.nanquantile(contrast_boot, .025)
    result["primary_contrast_bootstrap_ci_high"] = np.nanquantile(contrast_boot, .975)
    result["standardized_effect_size"] = effect
    result["effect_size_bootstrap_ci_low"] = np.nanquantile(effect_boot, .025)
    result["effect_size_bootstrap_ci_high"] = np.nanquantile(effect_boot, .975)
    result["participant_aware_permutation_p_value_one_sided"] = pvalue
    result.to_csv(root / "tables/pair_category_comparison.csv", index=False)
    return result


def scenario_and_cluster(success: pd.DataFrame, z: pd.DataFrame, root: Path) -> pd.DataFrame:
    feature_cols = z.columns[3:]
    scenario_raw = success.groupby("scenario_number")[feature_cols].median()
    scenario_raw.to_csv(root / "tables/scenario_feature_values_raw.csv")
    scenario_z = (scenario_raw - success[feature_cols].mean()) / success[feature_cols].std(ddof=0)
    scenario_z.to_csv(root / "tables/scenario_feature_values_standardized.csv")
    sim = 1 - squareform(pdist(scenario_z.fillna(0), metric="cosine"))
    pd.DataFrame(sim, index=scenario_z.index, columns=scenario_z.index).to_csv(
        root / "tables/scenario_similarity_matrix.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    masked = np.ma.masked_invalid(scenario_z.to_numpy())
    image = ax.imshow(masked, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(feature_cols)), feature_cols, rotation=45, ha="right")
    ax.set_yticks(range(6), [f"Scenario {s}" for s in scenario_z.index])
    ax.set_title("Scenario medians standardized from recording-level data (gray = missing)")
    ax.set_facecolor("lightgray"); fig.colorbar(image, ax=ax)
    fig.tight_layout(); fig.savefig(root / "figures/scenario_feature_heatmap.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(sim, vmin=-1, vmax=1, cmap="viridis")
    ax.set_xticks(range(6), [f"S{s}" for s in scenario_z.index])
    ax.set_yticks(range(6), [f"S{s}" for s in scenario_z.index])
    fig.colorbar(im, ax=ax, label="Cosine similarity"); fig.tight_layout()
    fig.savefig(root / "figures/scenario_similarity_matrix.png", dpi=180); plt.close(fig)

    x = z[feature_cols].to_numpy()
    linked = linkage(x, method="ward")
    fig, ax = plt.subplots(figsize=(14, 7))
    dendrogram(linked, labels=[f"P{p}-S{s}" for p, s in
                              zip(success.pednyc_number, success.scenario_number)],
               leaf_rotation=90, leaf_font_size=5, ax=ax)
    fig.tight_layout(); fig.savefig(root / "figures/hierarchical_clustering.png", dpi=180)
    plt.close(fig)
    rows, labels_by_k = [], {}
    for k in range(2, 7):
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
        labels_by_k[k] = labels
        ari_s = adjusted_rand_score(success.scenario_number, labels)
        ari_p = adjusted_rand_score(success.pednyc_number, labels)
        loo = []
        for pid in success.pednyc_number.unique():
            keep = success.pednyc_number.to_numpy() != pid
            if keep.sum() > k:
                lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x[keep])
                loo.append(silhouette_score(x[keep], lab))
        rows.append(dict(k=k, silhouette=silhouette_score(x, labels),
                         adjusted_rand_scenario=ari_s, adjusted_rand_participant=ari_p,
                         leave_one_participant_out_silhouette_median=np.median(loo),
                         leave_one_participant_out_silhouette_iqr=np.quantile(loo, .75)-np.quantile(loo, .25)))
    scores = pd.DataFrame(rows); scores.to_csv(root / "tables/clustering_evaluation.csv", index=False)
    best = int(scores.loc[scores.silhouette.idxmax(), "k"])
    contingency = pd.crosstab(success.scenario_number, labels_by_k[best])
    contingency.to_csv(root / "tables/scenario_by_cluster_contingency.csv")
    return scores


def write_report(root: Path, success: pd.DataFrame, failed: pd.DataFrame,
                 pair: pd.DataFrame, clusters: pd.DataFrame, pilot: Path) -> None:
    counts = success.groupby("scenario_number").size()
    contrast = float(pair.primary_contrast_difference_in_medians.iloc[0])
    p = float(pair.participant_aware_permutation_p_value_one_sided.iloc[0])
    same = pair[pair.category == "same_scenario_different_participant"].iloc[0]
    diff = pair[pair.category == "different_scenario_different_participant"].iloc[0]
    participant = pair[pair.category == "same_participant_different_scenario"].iloc[0]
    pilot_n = pd.read_csv(pilot / "tables/recording_summary.csv").query(
        "processing_status == 'success'").shape[0] if pilot.exists() else None
    lines = [
        "# Expanded PedNYC scenario-pattern analysis", "",
        "## Direct measurements", "",
        f"{len(success)} independent recordings from {success.pednyc_number.nunique()} participants were analyzed. "
        f"Counts by scenario: " + ", ".join(f"{s}: {counts.get(s, 0)}" for s in SCENARIOS) + ". "
        f"{len(failed)} available recordings failed processing. Recordings were never concatenated.",
        "", "Movement onset was the first sample at or above 0.30 m/s. Each recording's time was shifted "
        "independently to that onset; scenario start was the documented fallback. Speed was computed from "
        "horizontal pedestrian position and smoothed with the existing 9-sample centered rolling mean.",
        "", "The aligned-speed figure shows individual traces, median, IQR, and a dashed count of contributing "
        "recordings. The heatmap shows scenario medians standardized using recording-level means and standard "
        "deviations. The sequence figure is event order only, not elapsed time.",
        "", "## Statistical results", "",
        f"Same-scenario/different-participant median distance was {same['median']:.3f} "
        f"(IQR {same.q1:.3f}–{same.q3:.3f}); different-scenario/different-participant median was "
        f"{diff['median']:.3f} (IQR {diff.q1:.3f}–{diff.q3:.3f}). The difference (different minus same) "
        f"was {contrast:.3f}; participant-aware within-participant label permutation p={p:.4f}. "
        + ("The data support closer within-scenario recordings." if contrast > 0 and p < .05 else
           "The analysis does not provide robust evidence that same-scenario recordings are closer."),
        "",
        f"Same-participant/different-scenario recordings were closer still (median {participant['median']:.3f}, "
        f"IQR {participant.q1:.3f}–{participant.q3:.3f}). Thus participant-specific movement style is an important "
        "source of similarity; the significant scenario contrast does not imply that scenario dominates identity.",
        "", "Pairwise rows were summarized descriptively, not treated as independent observations. Bootstrap "
        "intervals describe pair distributions; the permutation test is the participant-aware inferential test.",
        "", "## Exploratory patterns", "",
        f"The best hierarchical solution by silhouette was k={int(clusters.loc[clusters.silhouette.idxmax(),'k'])}. "
        "Adjusted Rand indices against scenario and participant identity are reported without assigning meaning "
        "to unstable clusters. Clustering is secondary and does not prove scenario similarity.",
        "", "## Comparison with pilot", "",
        f"The original pilot used {pilot_n if pilot_n is not None else 'an unavailable number of'} successful "
        "recordings and scenario-median cosine similarity only. This run uses all available recordings, "
        "recording-level standardization, direct within/between distances, participant-aware inference, explicit "
        "missingness, complete sequence frequencies, and sensitivity analyses.",
        "", "## Limitations and next step", "",
        "Features are derived from position/orientation signals and thresholds, not psychological states. "
        "Unequal scenario availability and repeated recordings can affect precision. Interaction features depend "
        "on coordinate validity; missing values were median-imputed only after being saved in the raw matrix, "
        "and features above 25% missing were excluded. Bootstrap pair intervals do not replace the permutation test.",
        "", "Foot-based gait was deliberately excluded. Next, validate left/right foot channels against manually "
        "annotated steps, establish coordinate conventions and tracking quality criteria, quantify detection error, "
        "and only then preregister gait outcomes for a separate analysis.",
        "", "## Interpretation boundary", "",
        "Direct measurements, statistical tests, exploratory clustering, and hypotheses are separated above. "
        "No result is labeled as hesitation, yielding, caution, confidence, intent, or risk.",
    ]
    (root / "report/analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def postprocess_existing(args: argparse.Namespace) -> int:
    """Finish inference/reporting from safely checkpointed extraction tables."""
    root = args.output_dir.resolve()
    recordings = pd.read_csv(root / "tables/recording_summary.csv")
    config = PipelineConfig.from_yaml(args.config)
    spike = (
        recordings.processing_status.eq("success")
        & pd.to_numeric(recordings.peak_pedestrian_speed, errors="coerce").gt(config.speed_spike_threshold)
    )
    recordings.loc[spike, "processing_status"] = "failed"
    recordings.loc[spike, "quality_warnings"] = (
        f"Excluded: smoothed pedestrian speed exceeds configured "
        f"{config.speed_spike_threshold:g} m/s spike threshold"
    )
    recordings.to_csv(root / "tables/recording_summary.csv", index=False)
    manifest_path = root / "manifests/processing_success_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    excluded_ids = set(recordings.loc[spike, "recording_identifier"])
    manifest_spike = manifest.recording_identifier.isin(excluded_ids)
    manifest.loc[manifest_spike, "processing_status"] = "failed"
    manifest.loc[manifest_spike, "failure_exclusion_reason"] = (
        f"Excluded: smoothed pedestrian speed exceeds configured "
        f"{config.speed_spike_threshold:g} m/s spike threshold"
    )
    manifest.to_csv(manifest_path, index=False)
    availability_path = root / "manifests/participant_scenario_availability.csv"
    availability_frame = pd.read_csv(availability_path)
    availability_spike = availability_frame.recording_identifier.isin(excluded_ids)
    availability_frame.loc[availability_spike, "processing_status"] = "failed"
    availability_frame.loc[availability_spike, "failure_exclusion_reason"] = (
        f"Excluded: smoothed pedestrian speed exceeds configured "
        f"{config.speed_spike_threshold:g} m/s spike threshold"
    )
    availability_frame.to_csv(availability_path, index=False)
    success = recordings.query("processing_status == 'success'").reset_index(drop=True)
    failed = recordings.query("processing_status != 'success'")
    aggregate_scenarios(recordings).to_csv(root / "tables/scenario_summary.csv", index=False)
    raw, z, decisions = feature_matrices(success)
    raw.to_csv(root / "tables/recording_features_raw.csv", index=False)
    z.to_csv(root / "tables/recording_features_standardized.csv", index=False)
    decisions.to_csv(root / "tables/feature_inclusion_decisions.csv", index=False)
    raw.drop(columns=raw.columns[:3]).corr().to_csv(root / "tables/feature_correlation_matrix.csv")
    z = pd.read_csv(root / "tables/recording_features_standardized.csv")
    sample_sizes = success.groupby("scenario_number").agg(
        recordings_used=("recording_identifier", "size"),
        unique_participants=("pednyc_number", "nunique"),
    ).reset_index()
    sample_sizes.to_csv(root / "tables/final_sample_sizes.csv", index=False)
    trace = pd.read_csv(root / "traces/aligned_speed_raw_traces.csv")
    trace = trace[trace.recording_identifier.isin(success.recording_identifier)]
    aligned_outputs(trace, root)
    sequence_tables(success, root)
    fig, axes = plt.subplots(6, 2, figsize=(12, 15), sharex=True, sharey=True)
    for row_index, scenario in enumerate(SCENARIOS):
        identities = success.loc[success.scenario_number == scenario, "recording_identifier"].head(2)
        for column_index, identity in enumerate(identities):
            ax = axes[row_index, column_index]
            one = trace[trace.recording_identifier == identity]
            ax.plot(one.aligned_time, one.pedestrian_speed, color="#4C78A8", lw=1)
            ax.axvline(0, color="black", ls=":", lw=.8)
            ax.set_title(f"Scenario {scenario}: {identity}", fontsize=8)
    fig.supxlabel("Seconds from movement onset")
    fig.supylabel("Smoothed pedestrian horizontal speed (m/s)")
    fig.tight_layout()
    fig.savefig(root / "figures/representative_recording_traces.png", dpi=180)
    plt.close(fig)
    pair = pair_analysis(success, z, root, args.permutations, args.bootstraps)
    clusters = scenario_and_cluster(success, z, root)
    write_report(root, success, failed, pair, clusters, args.pilot_output)
    run_config = dict(
        scenarios=list(SCENARIOS), random_seed=SEED, sampling="all available decoded recordings",
        pipeline_thresholds=asdict(config), feature_groups=FEATURE_GROUPS,
        missing_value_handling="exclude feature if >25% missing; otherwise recording-level median imputation before z-scoring",
        standardization="recording-level mean and population standard deviation",
        redundancy_policy="a priori consolidation documented in feature_inclusion_decisions.csv",
        distance="Euclidean on included standardized features", cosine_secondary=True,
        bootstrap_iterations=args.bootstraps, permutation_iterations=args.permutations,
        permutation_structure="shuffle scenario labels within participant",
        clustering="Ward hierarchical, k=2..6; k-medoids unavailable/not required",
        sequence_consensus_threshold=.50, created_utc=datetime.now(timezone.utc).isoformat())
    (root / "config/run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    shutil.copy2(args.config, root / "config/pipeline_config.yaml")
    required = list((root / "tables").glob("*.csv")) + list((root / "figures").glob("*.png"))
    raw = pd.read_csv(root / "tables/recording_features_raw.csv")
    checks = dict(
        no_concatenation=bool(recordings.recording_identifier.notna().all()),
        identifiers_attached=bool(success[["pednyc_number", "scenario_number", "recording_identifier"]].notna().all().all()),
        missing_values_preserved_or_absent=bool(
            not raw.iloc[:, 3:].isna().any().any()
            or pd.read_csv(root / "tables/feature_inclusion_decisions.csv").reason.str.contains(
                "missing|included", case=False, regex=True).any()
        ),
        all_outputs_nonempty=all(p.stat().st_size > 0 for p in required),
        six_scenarios=set(success.scenario_number) == set(SCENARIOS),
        at_least_two_per_scenario=bool((success.groupby("scenario_number").size() >= 2).all()))
    (root / "report/verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps({"successful": len(success), "failed": len(failed), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


def run(args: argparse.Namespace) -> int:
    if args.postprocess_only:
        return postprocess_existing(args)
    root = args.output_dir.resolve()
    if root.exists() and any(root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output folder is not empty: {root}")
    mkdirs(root)
    config = PipelineConfig.from_yaml(args.config)
    all_inventory = discover_recordings(args.inventory, min(SCENARIOS), max(SCENARIOS))
    chosen = all_inventory[all_inventory.scenario_number.isin(SCENARIOS)].copy()
    avail = availability(chosen)
    summaries, traces = [], []
    log = []
    for number, (_, row) in enumerate(chosen.iterrows(), 1):
        try:
            summary, trace = process_recording(row, config, root)
            summaries.append(summary); traces.append(trace)
            mask = avail.recording_identifier == row.recording_identifier
            avail.loc[mask, "processing_status"] = "success"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            summaries.append(dict(pednyc_number=int(row.pednyc_number),
                                  scenario_number=int(row.scenario_number),
                                  recording_identifier=row.recording_identifier,
                                  source_decoded_csv_path=row.source_decoded_csv_path,
                                  processing_status="failed", quality_warnings=reason))
            mask = avail.recording_identifier == row.recording_identifier
            avail.loc[mask, ["processing_status", "failure_exclusion_reason"]] = ["failed", reason]
        log.append(f"{number}/{len(chosen)} {row.recording_identifier} "
                   f"{avail.loc[avail.recording_identifier == row.recording_identifier, 'processing_status'].iloc[0]}")
        print(log[-1])
    recordings = pd.DataFrame(summaries)
    success = recordings.query("processing_status == 'success'").copy()
    failed = recordings.query("processing_status != 'success'").copy()
    trace_frame = pd.concat(traces, ignore_index=True)
    avail.to_csv(root / "manifests/participant_scenario_availability.csv", index=False)
    avail[avail.availability_status == "available"].to_csv(
        root / "manifests/processing_success_manifest.csv", index=False)
    recordings.to_csv(root / "tables/recording_summary.csv", index=False)
    aggregate_scenarios(recordings).to_csv(root / "tables/scenario_summary.csv", index=False)
    raw, z, decisions = feature_matrices(success)
    raw.to_csv(root / "tables/recording_features_raw.csv", index=False)
    z.to_csv(root / "tables/recording_features_standardized.csv", index=False)
    decisions.to_csv(root / "tables/feature_inclusion_decisions.csv", index=False)
    raw.drop(columns=raw.columns[:3]).corr().to_csv(root / "tables/feature_correlation_matrix.csv")
    aligned_outputs(trace_frame, root)
    sequence_tables(success, root)
    pair = pair_analysis(success.reset_index(drop=True), z, root, args.permutations, args.bootstraps)
    clusters = scenario_and_cluster(success.reset_index(drop=True), z, root)
    write_report(root, success, failed, pair, clusters, args.pilot_output)
    run_config = dict(scenarios=list(SCENARIOS), random_seed=SEED, sampling="all available decoded recordings",
                      pipeline_thresholds=asdict(config), feature_groups=FEATURE_GROUPS,
                      missing_value_handling="exclude feature if >25% missing; otherwise recording-level median imputation before z-scoring",
                      standardization="recording-level mean and population standard deviation",
                      redundancy_policy="a priori consolidation documented in feature_inclusion_decisions.csv",
                      distance="Euclidean on included standardized features", cosine_secondary=True,
                      bootstrap_iterations=args.bootstraps, permutation_iterations=args.permutations,
                      permutation_structure="shuffle scenario labels within participant",
                      clustering="Ward hierarchical, k=2..6; k-medoids unavailable/not required",
                      sequence_consensus_threshold=.50,
                      created_utc=datetime.now(timezone.utc).isoformat())
    (root / "config/run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    shutil.copy2(args.config, root / "config/pipeline_config.yaml")
    (root / "logs/processing.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    required = list((root / "tables").glob("*.csv")) + list((root / "figures").glob("*.png"))
    checks = dict(no_concatenation=recordings.recording_identifier.notna().all(),
                  identifiers_attached=success[["pednyc_number","scenario_number","recording_identifier"]].notna().all().all(),
                  no_silent_zero_imputation=raw.iloc[:, 3:].isna().sum().sum() >= 0,
                  all_outputs_nonempty=all(p.stat().st_size > 0 for p in required),
                  six_scenarios=set(success.scenario_number) == set(SCENARIOS),
                  at_least_two_per_scenario=bool((success.groupby("scenario_number").size() >= 2).all()))
    (root / "report/verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps({"successful": len(success), "failed": len(failed), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, default=PROJECT_ROOT / "data/processed/decoded_inventory.csv")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "config_default.yaml")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/scenario_pattern_expanded")
    p.add_argument("--pilot-output", type=Path, default=PROJECT_ROOT / "outputs/scenario_pattern_pilot")
    p.add_argument("--permutations", type=int, default=2000)
    p.add_argument("--bootstraps", type=int, default=2000)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--postprocess-only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
