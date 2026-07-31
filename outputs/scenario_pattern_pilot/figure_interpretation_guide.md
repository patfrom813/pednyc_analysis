# Interpretation Guide for the Scenario Pattern Pilot Figures

## Scope and provenance

All four figures were created by `create_figures()` in
`src/scenario_pattern_pilot.py` (lines 275–375). The upstream recording-level
calculations are in `process_recording()` (lines 144–223), and the
scenario-level aggregation is in `aggregate_scenarios()` (lines 234–266).

The pilot used random seed 42 and selected PedNYC1, PedNYC4, PedNYC9,
PedNYC22, and PedNYC25. It included 29 successful recordings: five each for
Scenarios 3, 7, 15, 16, and 21, and four for Scenario 12. These facts are
recorded in `selection_manifest.csv` and `run_config.json`.

The prompt refers to `recording_summary.csv` and `scenario_summary.csv` at the
pilot-folder root. Their actual locations are:

- `tables/recording_summary.csv`
- `tables/scenario_summary.csv`

The original inputs were the semicolon-delimited decoded CSVs named in the
manifest's `source_decoded_csv_path` column. No participant recordings were
concatenated.

The shared upstream pipeline was:

1. `load_scenario()` reads a decoded CSV and normalizes headers.
2. `normalize_columns()` maps source columns to canonical variables such as
   `time`, `ped_x`, `ped_z`, `veh_x`, and `veh_z`.
3. `clean_metrics()` cleans the canonical values.
4. `MetricsEngine.compute()` creates frame-level kinematics.
5. `MacroSegmenter.segment()` creates macro phases.
6. `MicroSegmenter.extract_windows()` creates windows inside each macro phase.
7. `BehaviorClassifier.classify()` assigns one label to each window.
8. `EventMerger.merge()` merges adjacent/overlapping windows with the same
   label and removes events shorter than 0.25 seconds.

Important shared thresholds from `run_config.json` include walking speed
0.30 m/s, low-motion speed 0.15 m/s, pause speed 0.10 m/s, minimum low-motion
run 0.45 seconds, minimum pause 0.12 seconds, acceleration slope
0.12 m/s², deceleration slope −0.25 m/s², event merge gap 0.15 seconds, and
minimum merged-event duration 0.25 seconds.

---

# 1. `aligned_speed_small_multiples.png`

## What this graph shows

This graph asks: **after placing pedestrian movement onset at a common time
zero, how similar or different are the participants' horizontal speed
profiles within each scenario?**

Each panel is one scenario, while each faint trace is one independent
participant-recording. This makes it possible to see both the typical
scenario trajectory and departures from it without combining recordings into
one continuous series.

Exact source:

- `src/scenario_pattern_pilot.py`
- Upstream: `process_recording()` at lines 144–223
- Plot: `create_figures()` at lines 343–375
- Speed calculation: `MetricsEngine.compute()` and `_speed()` in
  `src/features/metrics_engine.py`, especially lines 17–31 and 64–65

## How it was constructed

### Inputs and speed calculation

This plot does **not** read `recording_summary.csv` to obtain the traces.
During the pilot run, `process_recording()` creates an in-memory `traces`
table from every decoded CSV. Its columns are:

- `pednyc_number`
- `scenario_number`
- `recording_identifier`
- `original_time`
- `aligned_time`
- `pedestrian_speed`

For successive frames \(i-1\) and \(i\), horizontal X/Z speed is initially:

\[
v_i =
\frac{\sqrt{(x_i-x_{i-1})^2+(z_i-z_{i-1})^2}}{\Delta t_i}.
\]

Infinite values are changed to missing and then filled with 0. The speed is
then centered-rolling-mean smoothed over **9 frames**, with at least one
sample allowed at an edge:

\[
\tilde v_i = \operatorname{mean}(v_j\text{ in the centered 9-frame window}).
\]

Therefore, the plotted `pedestrian_speed` is the pipeline's **smoothed
horizontal speed**, not raw frame-to-frame speed. Although thresholds are in
physical units, this particular smoothing width is frame-based, so its
effective duration can vary with frame rate.

### Movement-onset alignment

In `process_recording()` lines 172–176, movement onset is the first frame at
which the smoothed speed is at least the configured walking threshold:

\[
t_{\text{onset}} = \min\{t_i:\tilde v_i \geq 0.30\ \mathrm{m/s}\}.
\]

Aligned time is:

\[
t_{\text{aligned},i}=t_i-t_{\text{onset}}.
\]

If no such frame exists, the code uses the recording's scenario-relative
start:

\[
t_{\text{aligned},i}=t_i-t_{\text{start}}.
\]

All 29 recordings in this generated pilot used
`pedestrian_movement_onset`; none used the fallback. The dotted line at zero
therefore marks detected movement onset for every displayed recording.

Alignment shifts the time origin only. Recordings remain at their original
speed and duration; they are **not time-normalized, stretched, or compressed**.

### Common grid and aggregation

The plotting grid is fixed at −5.0 through 31.0 seconds in 0.1-second steps:

\[
G=\{-5.0,-4.9,\ldots,31.0\}.
\]

Within each recording, source samples are sorted by aligned time, rows missing
aligned time or speed are dropped, and `numpy.interp` performs piecewise
linear interpolation onto \(G\). Grid points before or after that recording's
observed range are set to missing (`NaN`), not zero.

At each grid time \(g\), using only recordings that have a finite interpolated
value there:

- central line = cross-recording median,
  \(\operatorname{median}_r \tilde v_r(g)\);
- lower edge of band = 25th percentile;
- upper edge of band = 75th percentile.

Thus, the band is the **interquartile range (IQR), Q1–Q3**, not a standard
deviation, standard error, or confidence interval. Missing tails do not become
zero. However, the number contributing to a time point can decline near the
ends even though the panel title reports the total number of plotted
recordings.

## How to read every visual element

- **Panels/rows and columns:** one subplot per represented scenario.
- **X-axis:** seconds from detected movement onset. Negative values are
  pre-onset observations; positive values are post-onset observations.
- **Y-axis:** smoothed pedestrian X/Z speed in m/s.
- **Faint blue lines:** one independently aligned recording. They are not raw
  traces because the 9-frame speed smoothing has already occurred.
- **Strong red line:** pointwise median across available participant traces.
  It is not a mean.
- **Orange band:** pointwise 25th–75th percentile range.
- **Black vertical dotted line:** aligned time zero, which is detected
  movement onset in this run.
- **`n`:** the number of recordings with at least two usable samples that
  contributed a trace to that panel: 5 for all panels except Scenario 12,
  where `n=4`.
- **Wide band:** greater cross-recording dispersion at that time.
- **Narrow band:** more similar speeds among the recordings still observed at
  that time. It does not necessarily mean all original participants are still
  contributing at late times.

The six panels share x and y axes. This makes their heights directly
comparable.

## What this particular result suggests

Measured observations visible in the generated figure include:

- All scenarios show a rise from low speed around aligned time zero, as
  expected from defining onset at 0.30 m/s.
- Scenarios 3, 7, 12, 15, and 21 generally reach approximately 1 m/s or more
  in their median profiles. Scenario 16 has the lowest scenario-level median
  speed in the summary table (0.12 m/s), compared with 0.35 m/s for
  Scenario 3.
- Scenario 7 shows an early high-speed interval, a mid-recording reduction,
  another high-speed interval, and then a later decline.
- Scenario 16 shows the clearest apparent subgrouping: several traces return
  to low speed after the first movement period, while some recordings show a
  later second movement period. Consequently, a pointwise median can jump
  when the mix of active recordings changes.
- Scenario 21 commonly shows two movement pulses separated by a marked
  slowdown around roughly 10–12 seconds.
- Scenario 3 has a broad IQR during much of the post-onset period, indicating
  substantial cross-recording speed variability.

The measured pattern “deceleration → low speed → acceleration” can be
described from these traces. Calling it hesitation, yielding, or interaction
with a vehicle requires vehicle-distance/closing evidence and preferably
video confirmation.

## What it does not prove

- Onset alignment forces every trace to cross the 0.30 m/s criterion near
  zero, so apparent agreement at onset is partly built into the method.
- Linear interpolation creates values between observed samples; it does not
  create new measured frames.
- At late times, the median and IQR may be based on fewer recordings because
  durations differ. Across this pilot, recording durations range from about
  21 to 57 seconds, while the graph stops at 31 seconds.
- Activity after 31 seconds and more than 5 seconds before onset is not shown.
- A pointwise median trajectory need not be a trajectory followed by any
  individual participant.
- The plot cannot establish why speed changed or whether a vehicle,
  instruction, geometry, tracking issue, or participant choice caused it.

---

# 2. `consensus_gantt.png`

## What this graph shows

This graph asks: **what complete ordered micro-event sequence occurred most
often within each scenario, and how often did that exact sequence occur?**

Despite the filename, it is not a conventional elapsed-time Gantt chart and
does not calculate consensus time intervals. It is an **ordinal sequence
diagram**.

Exact source:

- `src/scenario_pattern_pilot.py`
- Event construction: `process_recording()` lines 159–185
- Scenario mode: `aggregate_scenarios()` lines 246–254
- Plot: `create_figures()` lines 314–341
- Window extraction: `MicroSegmenter.extract_windows()` in
  `src/segmentation/micro.py`
- Label rules: `BehaviorClassifier.classify()` in
  `src/features/behavior.py`
- Merging: `EventMerger.merge()` in `src/features/merger.py`

## How it was constructed

Micro windows are generated separately inside every macro segment. A macro
segment is a contiguous phase labeled from the existing macro algorithm's
canonical sequence: `low_motion`, `acceleration`, `crossing`,
`deceleration`, and `post_motion`, subject to which detected boundaries are
unique and present. Macro boundaries use the smoothed speed trend,
walking-speed threshold 0.30 m/s, acceleration slope 0.12 m/s²,
deceleration slope −0.25 m/s², a 0.45-second macro trend window, and the
algorithm's search/final-fraction rules.

Inside a macro segment, adaptive windows are approximately 1 second and are
advanced by 0.5 seconds. Their allowed size is 0.25–2.5 seconds. Each window
gets exactly one `predicted_label` from the existing decision order:

1. `head_check`
2. `possible_yielding`
3. `possible_hesitation`
4. `low_motion`
5. `acceleration`
6. `deceleration`
7. `crossing_commitment`
8. fallback `low_motion`

Some important rules are:

- `head_check`: brief head-turn rate at least 120°/s within at most 0.5 s,
  or head/body yaw difference at least 20°;
- `low_motion`: mean speed below 0.15 m/s;
- `acceleration`: speed slope at least 0.12 m/s² or mean acceleration at
  least 0.25 m/s²;
- `deceleration`: speed slope at most −0.25 m/s² or mean acceleration at
  most −0.25 m/s²;
- `crossing_commitment`: progress ratio at least 0.70, speed at least
  0.25 m/s, and window duration at least 0.25 s;
- `possible_yielding`: vehicle distance at most 12 m, vehicle approaching,
  plus vehicle braking, pedestrian negative slope, or pedestrian speed below
  0.30 m/s.

Same-label windows that overlap or have a gap no larger than 0.15 seconds are
merged. A merged event's duration is end minus start; events shorter than
0.25 seconds are removed.

For each recording, `_sequence()` removes only consecutive duplicate labels
and joins the remaining event labels in chronological order. For each
scenario, `Counter(...).most_common(1)` selects the most frequent **entire
sequence string**. Its prevalence is:

\[
p_s =
\frac{\text{recordings in scenario }s\text{ with that exact full sequence}}
{\text{successful recordings in scenario }s}.
\]

There is no minimum prevalence threshold. Even a sequence seen once is the
mode if every sequence differs. Ties are resolved by `Counter.most_common()`
using encounter/insertion order; no scientifically meaningful tie-breaker is
implemented.

The plot splits the modal string on ` -> `. Event position \(k\) becomes a
bar from \(x=k\) to \(x=k+1\). Every bar therefore has length one, regardless
of the real event duration. The opacity is:

\[
\alpha = 0.25 + 0.75p_s.
\]

## How to read every visual element

- **Each row:** one scenario, not one participant.
- **Horizontal position:** first, second, third, and later events in the modal
  order. It is not seconds, aligned time, or normalized duration.
- **Bar length:** always one ordinal event slot. It does not encode elapsed
  duration.
- **Color:** categorical event label, assigned from Matplotlib's `tab10`
  palette after alphabetically sorting all labels found in the modal
  sequences.
- **Letters:** `LM` low motion, `A` acceleration, `D` deceleration, `C`
  crossing commitment, `H` head check, `Y` possible yielding, and `O`
  possible hesitation.
- **Opacity and percentage at row end:** percentage of recordings whose full
  event sequence exactly matches the displayed sequence.
- **Row length:** number of events in the modal sequence, not recording
  duration.
- **`n`:** not printed on this figure. The denominators are 5, 5, 4, 5, 5,
  and 5 for Scenarios 3, 7, 12, 15, 16, and 21 respectively.

Events are not drawn at their true times, so visual overlap is impossible in
this plot. Upstream classifier windows can overlap because of the 0.5-second
step and adaptive window length, but the chart displays only the ordered,
merged event labels.

## What this particular result suggests

- Scenario 12's displayed sequence has 25% prevalence: one of four recordings
  followed that exact full sequence.
- Every other displayed sequence has 20% prevalence: one of five recordings
  followed it.
- Therefore, no scenario has a strong exact micro-sequence consensus. The
  chart primarily illustrates one representative modal ordering, not a
  broadly shared template.
- Repeated measured transitions among acceleration, crossing commitment, and
  deceleration appear in several rows.
- Low motion occurs at the beginning or end of many modal sequences.
- Some modal sequences contain `possible_yielding`, but that label is a
  rule-based combination of distance/approach and motion evidence. It should
  not be interpreted as verified yielding without vehicle context and video.

At the broader macro level, which is not what this Gantt directly draws, the
canonical five-phase sequence had much stronger agreement: 100% in Scenarios
3, 7, and 21; 60% in 15 and 16; and 50% in 12.

## What it does not prove

- “Consensus” does not mean majority agreement here; 20–25% is enough to be
  displayed because no minimum was imposed.
- Equal-width blocks must not be read as event durations.
- The modal sequence may be tied with other one-recording sequences.
- The row may not represent most participants, and an aggregate mode need not
  be a biologically meaningful prototype.
- The chart suppresses event timing, duration, confidence, and overlap.
- Psychological interpretations such as hesitation or intent are not
  established by the event order.

---

# 3. `scenario_feature_heatmap.png`

## What this graph shows

This graph asks: **which scenarios are relatively high or low on a compact set
of speed, duration, event-count, and event-duration summaries?**

It puts heterogeneous measurements on a common standardized scale so broad
scenario fingerprints can be compared visually.

Exact source:

- Feature list: `src/scenario_pattern_pilot.py` lines 29–45
- Recording summaries: `process_recording()` lines 177–210
- Scenario aggregation: `aggregate_scenarios()` lines 234–266
- Standardization: `_standardize()` lines 269–273
- Plot: `create_figures()` lines 275–290

Inputs:

- decoded source CSVs named in `selection_manifest.csv`;
- recording-level output `tables/recording_summary.csv`;
- scenario-level plot input `tables/scenario_summary.csv`.

## How it was constructed

Only successful recording rows are aggregated. For each scenario and each
displayed feature, the scenario value is the **median across recordings**.
The scenario summary also contains IQR columns, but the heatmap does not plot
them.

The 13 displayed columns are:

1. **Duration** — final recording time minus initial recording time, seconds.
2. **Macro segments** — count of macro phases in the recording.
3. **Low motion** — total duration of speed-below-0.15 m/s runs lasting at
   least 0.45 s, seconds.
4. **Acceleration count** — number of merged events labeled acceleration.
5. **Acceleration duration** — sum of acceleration-event durations, seconds.
6. **Deceleration count** — number of merged deceleration events.
7. **Deceleration duration** — sum of deceleration-event durations, seconds.
8. **Pause count** — number of speed-below-0.10 m/s runs lasting at least
   0.12 s.
9. **Pause duration** — sum of those pause-run durations, seconds.
10. **Stop-start count** — pauses whose start is after recording start and
    whose end is before recording end; in other words, internal pauses bounded
    temporally by the recording, not a frequency-domain oscillation measure.
11. **Head rotation count** — count of merged events labeled `head_check`.
12. **Median speed** — median smoothed frame-level pedestrian speed within a
    recording, then median of those recording medians within the scenario.
13. **Peak speed** — maximum smoothed frame-level pedestrian speed within a
    recording, then median of those recording maxima within the scenario.

For every feature column \(f\), the six scenario medians are standardized:

\[
z_{s,f} =
\frac{x_{s,f}-\bar x_f}{\sigma_f},
\]

where \(\bar x_f\) and \(\sigma_f\) are the mean and population standard
deviation (`ddof=0`) of the represented scenarios' medians for that feature.

Non-numeric values become missing. A zero standard deviation is replaced by
missing. The resulting missing standardized cells are then filled with 0.
Thus, a constant or unavailable feature would appear at the color midpoint,
not as a special missing-data symbol.

The color display is fixed to −2 through +2; values beyond this range would
saturate at the end colors.

## How to read every visual element

- **Rows:** represented scenarios.
- **`n` in a row label:** number of successful selected recordings used for
  that scenario median. Scenario 12 has 4; the others have 5.
- **Columns:** the 13 features defined above.
- **Red/warm cell:** scenario median is above the cross-scenario mean for that
  feature.
- **Blue/cool cell:** below the cross-scenario mean.
- **Near-white cell:** near the cross-scenario mean or replaced with zero
  after an undefined standardization.
- **Darker/more saturated:** farther from the mean in standard-deviation
  units, capped visually at ±2.

Comparisons are most literal **within a column**, because every column was
standardized separately. Comparing colors across columns compares relative
standing, not physical magnitude. For example, equally red pause count and
speed cells do not mean equal numbers, seconds, or m/s.

A visually high value means “higher relative to the other represented
scenario medians,” not necessarily objectively high behavior.

## What this particular result suggests

The strongest relative contrasts confirmed from the plotted standardized
values are:

- Scenario 3 is relatively high in acceleration count/duration,
  deceleration count/duration, median speed, and peak speed.
- Scenario 7 is highest in head-rotation count and peak speed, while its
  acceleration count/duration are relatively low.
- Scenario 12 is lowest in duration, macro-segment count, and low-motion
  duration; it is also low in total pause duration.
- Scenario 15 is highest in pause count and stop-start count.
- Scenario 16 is lowest in median speed and acceleration/deceleration
  duration, consistent with its lower-speed small-multiple profile.
- Scenario 21 is highest in recording duration, low-motion duration, and
  total pause duration, but relatively low in median and peak speed.

These are measured relative profiles. For example, Scenario 15's higher
pause/stop-start summaries support the observation “more internal low-speed
runs” in this sample. They do not by themselves establish hesitation,
yielding, confusion, or risk.

## What it does not prove

- With only 4–5 recordings per row, scenario medians are uncertain and can
  change materially with one additional participant.
- Standardization is based only on these six represented scenarios, so colors
  would change if more scenarios were added even if raw values did not.
- The heatmap hides within-scenario IQR and individual subgroups.
- Counts can rise with recording duration and event fragmentation.
- A zero-colored cell can mean average, constant-feature replacement, or
  missing standardization; the plot does not distinguish these cases.
- The heatmap cannot identify causal mechanisms or verify behavioral intent.

---

# 4. `scenario_similarity_matrix.png`

## What this graph shows

This graph asks: **which scenarios have similarly shaped multifeature
fingerprints after their summary features are standardized?**

It is useful as an exploratory map of scenario-level resemblance, not as a
test that participants behaved identically.

Exact source:

- Same 13-feature list and scenario medians as the heatmap
- `_standardize()` at `src/scenario_pattern_pilot.py` lines 269–273
- `create_figures()` lines 292–312

Its immediate input is the same standardized heatmap matrix derived from
`tables/scenario_summary.csv`.

## How it was constructed

Each scenario is represented by the 13-element vector:

\[
\mathbf z_s = (
z_{\text{duration}},
z_{\text{macro count}},
z_{\text{low motion}},
\ldots,
z_{\text{peak speed}}
).
\]

These are standardized scenario medians, not participant-level observations.
No explicit feature weights are applied; after standardization, each retained
column enters the dot product once.

The code intends to retain columns with at least 50% nonmissing data:
`heat.notna().mean() >= .5`. However, `heat` was already filled with zero at
line 278, so every column passes this test in this run. Therefore all 13
features are used, including any value that might previously have become a
zero-filled standardized value.

Similarity is cosine similarity:

\[
\operatorname{sim}(a,b)=
\frac{\mathbf z_a\cdot\mathbf z_b}
{\|\mathbf z_a\|_2\|\mathbf z_b\|_2}.
\]

Nonfinite similarity values are replaced with zero.

Scenarios are reordered by descending mean similarity to all scenarios:

\[
\operatorname{order} =
\operatorname{argsort}\left(-\operatorname{mean}_b
\operatorname{sim}(a,b)\right).
\]

This is a simple ordering heuristic, not hierarchical clustering.

## How to read every visual element

- **Rows and columns:** the same scenario set in the same reordered order.
- **A cell:** cosine similarity between two standardized scenario median
  fingerprints.
- **Diagonal:** 1 for each nonzero vector because every scenario is identical
  to itself.
- **Symmetry:** cell \((a,b)\) equals cell \((b,a)\).
- **Color scale:** fixed from −1 to +1.
  - near +1: vectors point in almost the same direction—features tend to be
    relatively high and low in a similar pattern;
  - near 0: little directional agreement;
  - near −1: approximately opposite relative profiles.
- **Bright yellow:** high positive similarity.
- **Dark purple:** strong negative similarity.
- **`S3`, `S7`, etc.:** scenario labels.
- **`low n`:** would be printed on a diagonal when a scenario has fewer than
  three successful recordings. No such warning appears because every
  represented scenario has at least four.

The plot does not print numeric values in cells; exact values must be
recalculated from the scenario summary and the formula above.

## What this particular result suggests

The most similar off-diagonal pairs are:

1. Scenarios 15 and 21: 0.333
2. Scenarios 7 and 16: 0.215
3. Scenarios 16 and 21: 0.154

These are only modest positive similarities, not near-identity.

The least similar pairs are:

1. Scenarios 12 and 21: −0.732
2. Scenarios 3 and 16: −0.705
3. Scenarios 7 and 21: −0.547

The negative Scenario 3/16 relationship is consistent with their opposing
relative speed and acceleration/deceleration summaries. Scenario 12/21
contrast strongly in duration, macro count, low-motion duration, and pause
duration.

Similarity here means their **scenario-level standardized medians** were
alike or opposed across 13 features. It does not mean that each participant
within one scenario had a counterpart with identical behavior in the other.

## What it does not prove

- Cosine similarity is sensitive to which features and scenarios are
  included.
- Correlated features—such as event count and event duration, or pause
  duration and low-motion duration—can effectively emphasize related
  phenomena more than once.
- Standardization gives each feature column comparable scale but does not
  make the features independent or scientifically equally important.
- Zero-filling occurs before the “50% nonmissing” filter, so the stated
  missing-feature exclusion is ineffective in the current implementation.
- The matrix ignores participant-level variability and sequence timing.
- A positive similarity is not evidence of the same causal process, and a
  negative value is not evidence of opposite psychological behavior.

---

# Concise comparison

| Graph | Unit of analysis | Axes | Main calculation | Appropriate research claim |
|---|---|---|---|---|
| Aligned speed small multiples | Recording traces grouped by scenario | X: seconds from movement onset; Y: smoothed X/Z speed (m/s) | Linear interpolation to 0.1-s grid; pointwise median and Q1–Q3 | “After onset alignment, these scenarios show these typical speed profiles and this cross-recording dispersion.” |
| Consensus Gantt | One modal full micro-event sequence per scenario | X: ordinal event position; Y: scenario | Mode of exact sequence strings; opacity/annotation = exact-sequence prevalence | “This was the most frequent complete event ordering, but it occurred in only this proportion of recordings.” |
| Scenario feature heatmap | Scenario medians | X: 13 features; Y: scenario | Median across recordings, then column-wise population z-score | “Relative to the represented scenarios, this scenario is high or low on these summary features.” |
| Scenario similarity matrix | Pairs of scenario fingerprints | Both axes: reordered scenarios | Cosine similarity of 13 standardized scenario-median features | “These scenario-level summary profiles are more alike or opposed in their relative feature patterns.” |

# Plain-English takeaway

The four graphs answer complementary questions. The aligned traces preserve
individual recordings and show when participants' speeds agree or diverge.
The sequence chart reduces each recording to an ordered event string and
shows that exact micro-event order generalizes poorly in this small sample.
The heatmap summarizes which scenarios are relatively high or low on measured
counts, durations, and speeds. The similarity matrix then compares those
multifeature scenario fingerprints.

Together they support a cautious conclusion: the existing pipeline found
recurring broad macro structure, especially in Scenarios 3, 7, and 21, while
participant-level timing and exact micro-event order remained variable.
That is evidence of partial cross-recording generalization of objective
movement structure—not proof of common intent, hesitation, yielding, or any
other psychological state.

# Details that could not be confirmed

1. The in-memory aligned `traces` table was not saved as a CSV, so the exact
   interpolated point values used in the speed figure cannot be audited from
   a standalone trace artifact after the run. Their construction is fully
   specified in code and the decoded sources remain listed in the manifest.
2. The consensus selection does not record whether the modal full sequence
   was tied with another sequence. `Counter.most_common(1)` returns one mode
   without saving tie metadata.
3. The figures do not save the per-time-point number of traces contributing to
   each speed median/IQR.
4. The similarity matrix does not save its numeric matrix or ordering as a
   table; exact pair values must be recomputed from `scenario_summary.csv`.
5. No video or independently coded interaction context was included, so
   psychological or vehicle-interaction interpretations cannot be confirmed.
