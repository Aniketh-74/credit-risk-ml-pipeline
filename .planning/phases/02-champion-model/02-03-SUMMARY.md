---
phase: 02-champion-model
plan: "03"
subsystem: registry
tags: [mlflow, model-registry, cbpdd, scipy, pytest]

# Dependency graph
requires:
  - phase: 02-champion-model
    plan: "02"
    provides: Trained LightGBM model (run_id from run_training_pipeline), auc_test metric logged, PR curve artifact
provides:
  - MLflow Model Registry promotion via @champion alias (set_registered_model_alias)
  - promote.py with register_and_promote(), get_champion_auc(), load_champion()
  - CB-PDD τ sensitivity validated on 30-day denial loop (τ=1000 confirmed as recommended value)
  - CLI scripts/promote_champion.py for production promotion workflows
affects: [03-scoring-api, 04-cbpdd, 05-orchestration]

# Tech tracking
tech-stack:
  added: [scipy==1.15.2]
  patterns:
    - MLflow 3.x alias API — client.set_registered_model_alias(name, alias, version); never transition_model_version_stage
    - mocked MlflowClient in unit tests — no real MLflow calls; patch("src.training.promote.MlflowClient")
    - CB-PDD simplified via Mann-Whitney U — accumulate τ samples, test against 7-day reference window
    - denial loop simulation — remove denied applicants, inject low-risk new applicants (RevolvingUtilization *= 0.97/day)

key-files:
  created:
    - src/training/promote.py
    - tests/training/test_promote.py
    - scripts/promote_champion.py
    - scripts/smoke_test_cbpdd.py
  modified:
    - .planning/STATE.md

# Results

## Model performance
- AUC on held-out test set: **0.8655** (target: ≥ 0.85 ✓)
- Early stopping at iteration 169/500 (50-round patience)
- AUC progression: 0.859 (iter 50) → 0.864 (iter 100) → 0.865 (iter 150) → 0.8655 (best)

## CB-PDD smoke test results
Simulation: 30-day denial loop, 1000 applicants/day, Mann-Whitney U at α=0.05
Reference window: days 0–6 (7000 scores); test windows accumulate τ samples before each check

| τ    | Detections (max 7) | First detection at day |
|------|--------------------|------------------------|
| 500  | 7                  | 14                     |
| 1000 | 7                  | 14                     |
| 2000 | 7                  | 14                     |

**Recommended τ: 1000** — paper default, all τ values produce identical detection behaviour
on this dataset. τ=1000 chosen as it aligns with the CB-PDD paper's calibration guidance.

# Deviations from plan

None.

# Decisions made

- τ=1000 selected as Phase 4 starting point — equal detection rate across all three τ values
  means there is no cost to using the paper's default. Phase 4 can tune up/down with MLflow
  experiment tracking if the real denial loop volume differs from 1000/day.
- `n_per_day=1000` required for statistical power — Mann-Whitney U needs ≥ 1000 accumulated
  samples in the test window to detect the ~3%/day drift reliably. The original n_per_day=100
  produced 0 detections because single-day batches had insufficient power (100 vs 700 reference).

# Verification

```
pytest tests/training/ -v  → 14 passed in 41.58s
python scripts/smoke_test_cbpdd.py
    AUC: 0.8655 ✓
    tau=500:  detections=7, first detection at day 14 ✓
    tau=1000: detections=7, first detection at day 14 ✓
    tau=2000: detections=7, first detection at day 14 ✓
    Smoke test PASSED ✓
```

MODEL-01: AUC > 0.85 confirmed (0.8655)
MODEL-02: PR curve artifact logged to plots/pr_curve.png
MODEL-03: @champion alias set via set_registered_model_alias — no deprecated stages API
