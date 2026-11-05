# Pricing wave 3 (P07-P14) — decision rule, written before any run

- Thresholds: `thresholds-v2`, unchanged. Battery: `valkit.campaign.pricing_checks`, run as-is (no wrapper; the brief now states the fixed-strike convention).
- Detected: a defective model gets verdict 2 (not fit) or 1 (with conditions) from `bench.verdict`. Missed: verdict 0.
- False alarm: a healthy model gets verdict 1 or 2. Accepted: verdict 0.
- Report counts with Wilson 95 % intervals; healthy n is small by construction, so no specificity claim beyond the interval.
- Wave 1 failure (strike convention) is addressed in the brief only; no threshold change. Any rerun after seeing results is labelled post hoc.
- Authors: sub-agents of one model family; stated as a limit.
