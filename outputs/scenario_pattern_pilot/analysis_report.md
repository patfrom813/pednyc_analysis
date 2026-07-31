# Scenario Pattern Pilot

- Selected participants (seed 42): PedNYC1, PedNYC4, PedNYC9, PedNYC22, PedNYC25.
- Processed successfully: 29 of 29 eligible selected recordings.
- Represented scenarios: 3, 7, 12, 15, 16, 21.
- Scenarios with at least three recordings for pilot comparison: 3, 7, 12, 15, 16, 21.
- Low-sample scenarios: none.

## Objective pattern comparison

- Scenario 3 (n=5): dominant macro sequence `low_motion -> acceleration -> crossing -> deceleration -> post_motion` (100%); dominant micro-event sequence `low_motion -> acceleration -> crossing_commitment -> acceleration -> crossing_commitment -> acceleration -> crossing_commitment -> acceleration -> crossing_commitment -> acceleration -> possible_yielding -> crossing_commitment -> possible_yielding -> crossing_commitment -> deceleration -> acceleration -> crossing_commitment -> acceleration -> crossing_commitment -> deceleration -> low_motion` (20%).
- Scenario 7 (n=5): dominant macro sequence `low_motion -> acceleration -> crossing -> deceleration -> post_motion` (100%); dominant micro-event sequence `head_check -> low_motion -> head_check -> low_motion -> acceleration -> crossing_commitment -> acceleration -> crossing_commitment -> acceleration -> deceleration -> crossing_commitment -> acceleration -> deceleration -> crossing_commitment -> deceleration -> crossing_commitment -> deceleration -> crossing_commitment -> low_motion -> head_check -> low_motion -> possible_yielding -> low_motion` (20%).
- Scenario 12 (n=4): dominant macro sequence `low_motion -> acceleration -> crossing -> deceleration` (50%); dominant micro-event sequence `head_check -> low_motion -> head_check -> acceleration -> crossing_commitment -> deceleration -> acceleration -> crossing_commitment -> acceleration -> crossing_commitment -> acceleration -> crossing_commitment -> deceleration -> low_motion` (25%).
- Scenario 15 (n=5): dominant macro sequence `low_motion -> acceleration -> crossing -> deceleration -> post_motion` (60%); dominant micro-event sequence `head_check -> low_motion -> acceleration -> crossing_commitment -> acceleration -> deceleration -> crossing_commitment -> deceleration -> crossing_commitment -> low_motion -> acceleration -> possible_yielding -> acceleration -> possible_yielding -> low_motion` (20%).
- Scenario 16 (n=5): dominant macro sequence `low_motion -> acceleration -> crossing -> deceleration -> post_motion` (60%); dominant micro-event sequence `acceleration -> deceleration -> low_motion -> acceleration -> crossing_commitment -> acceleration -> crossing_commitment -> acceleration -> possible_yielding -> crossing_commitment -> deceleration -> low_motion` (20%).
- Scenario 21 (n=5): dominant macro sequence `low_motion -> acceleration -> crossing -> deceleration -> post_motion` (100%); dominant micro-event sequence `low_motion -> head_check -> low_motion -> crossing_commitment -> acceleration -> crossing_commitment -> acceleration -> crossing_commitment -> possible_yielding -> deceleration -> crossing_commitment -> acceleration -> crossing_commitment -> acceleration -> crossing_commitment -> deceleration -> possible_yielding -> low_motion` (20%).

## Strongest recurrence, differences, and similarity

- All represented scenarios contained low-motion events in every successful recording.
- The canonical five-phase macro sequence was shared by 100% of recordings in Scenarios 3, 7, and 21; agreement was 60% in Scenarios 15 and 16, and 50% in Scenario 12.
- Scenario 3 had the highest median participant speed (0.35 m/s), while Scenario 16 had the lowest (0.12 m/s), a 0.23 m/s difference.
- Exact micro-event order was highly variable: no scenario's dominant full micro sequence occurred in more than 25% of its recordings.
- The closest exploratory scenario fingerprints were Scenarios 15 and 21 (similarity 0.33), Scenarios 7 and 16 (similarity 0.22), Scenarios 16 and 21 (similarity 0.15).
- Macro-sequence agreement ranged from 50% to 100%, showing stronger cross-participant recurrence at the macro level than for exact micro-event sequences.

## Interpretation and limitations

- Labels are mathematically defined movement/head/context evidence; they are not psychological judgments.
- A possible behavioral interpretation (for example, yielding) requires interaction context or video confirmation.
- Dominant sequences are unreliable when participant coverage is low, and missing head signals are retained as warnings rather than converted to zero evidence.
- Fingerprint similarity is exploratory: it uses standardized aggregate medians and marks scenarios with fewer than three successful recordings.
- Before scaling, review strong speed outliers, confirm coordinate/time consistency across participants, and validate event thresholds against a small video-checked sample.
- The existing algorithm is considered consistent only where all recordings passed identical stages and thresholds; sequence agreement percentages quantify, rather than assume, generalization.
