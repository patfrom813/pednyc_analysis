# Expanded PedNYC scenario-pattern analysis

## Direct measurements

155 independent recordings from 30 participants were analyzed. Counts by scenario: 3: 26, 7: 23, 12: 26, 15: 28, 16: 25, 21: 27. 8 available recordings failed processing. Recordings were never concatenated.

Movement onset was the first sample at or above 0.30 m/s. Each recording's time was shifted independently to that onset; scenario start was the documented fallback. Speed was computed from horizontal pedestrian position and smoothed with the existing 9-sample centered rolling mean.

The aligned-speed figure shows individual traces, median, IQR, and a dashed count of contributing recordings. The heatmap shows scenario medians standardized using recording-level means and standard deviations. The sequence figure is event order only, not elapsed time.

## Statistical results

Same-scenario/different-participant median distance was 3.960 (IQR 3.097–5.116); different-scenario/different-participant median was 4.390 (IQR 3.508–5.490). The difference (different minus same) was 0.430; participant-aware within-participant label permutation p=0.0005. The data support closer within-scenario recordings.

Same-participant/different-scenario recordings were closer still (median 3.529, IQR 2.872–4.558). Thus participant-specific movement style is an important source of similarity; the significant scenario contrast does not imply that scenario dominates identity.

Pairwise rows were summarized descriptively, not treated as independent observations. Bootstrap intervals describe pair distributions; the permutation test is the participant-aware inferential test.

## Exploratory patterns

The best hierarchical solution by silhouette was k=4. Adjusted Rand indices against scenario and participant identity are reported without assigning meaning to unstable clusters. Clustering is secondary and does not prove scenario similarity.

## Comparison with pilot

The original pilot used 29 successful recordings and scenario-median cosine similarity only. This run uses all available recordings, recording-level standardization, direct within/between distances, participant-aware inference, explicit missingness, complete sequence frequencies, and sensitivity analyses.

## Limitations and next step

Features are derived from position/orientation signals and thresholds, not psychological states. Unequal scenario availability and repeated recordings can affect precision. Interaction features depend on coordinate validity; missing values were median-imputed only after being saved in the raw matrix, and features above 25% missing were excluded. Bootstrap pair intervals do not replace the permutation test.

Foot-based gait was deliberately excluded. Next, validate left/right foot channels against manually annotated steps, establish coordinate conventions and tracking quality criteria, quantify detection error, and only then preregister gait outcomes for a separate analysis.

## Interpretation boundary

Direct measurements, statistical tests, exploratory clustering, and hypotheses are separated above. No result is labeled as hesitation, yielding, caution, confidence, intent, or risk.
