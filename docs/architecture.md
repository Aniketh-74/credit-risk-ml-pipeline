# Architecture: Credit Risk ML Pipeline

## Overview

A production ML system that detects **performative drift** — when a model's own predictions cause the future data distribution to shift — and automatically closes the loop with drift-triggered retraining.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Daily Airflow DAG                            │
│                                                                     │
│  feedback_simulate → batch_score → drift_check ─┬─ trigger_retrain │
│                                                  │       ↓          │
│                                                  │  promote_if_     │
│                                                  │  improved        │
│                                                  └─ skip_retrain    │
└─────────────────────────────────────────────────────────────────────┘
         ↓                    ↑
┌────────────────┐    ┌───────────────────┐    ┌──────────────────┐
│  FastAPI       │    │   PostgreSQL       │    │  MLflow Registry │
│  /score        │───▶│  predictions       │───▶│  @champion alias │
│  /outcome      │    │  outcomes          │    │  auc_test metric │
│  /health       │    │  drift_scores      │    └──────────────────┘
│                │    │  alerts            │
│  split-path    │    └───────────────────┘
│  router (10%   │             ↑
│  checkerboard) │    ┌────────────────────┐
└────────────────┘    │  Streamlit         │
                      │  Dashboard         │
                      │  - drift timeline  │
                      │  - CB-PDD vs PSI   │
                      │  - τ sensitivity   │
                      │  - model registry  │
                      └────────────────────┘
```

---

## Feedback Loop Simulation

The core problem CB-PDD is designed to detect:

```
Day 1:  Applicant applies → model scores 0.72 → denied
        Applicant learns they need to improve → returns Day 3

Day 3:  Same applicant, nudged features → model scores 0.58 → denied again
        Score is drifting down but the MODEL caused this drift

Day 14: CB-PDD detects the density change pattern
        → Retraining triggered with bias-corrected data
        → New @champion model promoted if AUC improves
```

This is **performative drift**: the model's prediction influences the next observation. Standard PSI would see the score distribution shift and flag it, but cannot tell whether the cause is organic population change or the model's own decisions creating a feedback loop.

---

## CB-PDD Detection Mechanism

From arXiv 2412.10545, adapted for production:

1. **CheckerBoardPredictor**: routes 10% of predictions through a checkerboard assignment — alternating approve/deny for similar applicants across trial periods of τ instances. This creates a controlled A/B structure.

2. **DensityChangeTracker**: for each completed trial, computes:
   ```
   a = correction_rate(last_w) − correction_rate(first_w)
   ```
   where `correction_rate` = fraction of denied predictions that were correct (applicant actually defaulted). A denial-loop feedback pattern produces a declining correction rate as denied applicants game the score.

3. **PerformativeDriftDetector**: runs Mann-Whitney U test on Group A vs Group B density change distributions. Requires **2 consecutive windows** with p < α before setting `is_drift=True`.

**Why 2 consecutive windows?** A single trial with p < 0.05 has a ~5% false positive rate by construction. Two consecutive detections reduce this to ~0.25% while still catching genuine drift within 2–3 weeks of onset.

---

## Data Flow

```
POST /score ──────────────┐
                           ▼
                   predictions table
                   (id, score, decision, path, predicted_at)
                           │
                    label delay (14-30 days)
                           │
POST /outcome ─────────────▼
                   outcomes table
                   (prediction_id, actual_default, outcome_received_at)
                           │
                    Airflow: drift_check task
                           │
                    compute_drift(db_url, window_days=30)
                    ├── _fetch_labeled_predictions()
                    │   WHERE predicted_at IS NOT NULL
                    │   AND outcome_received_at IS NOT NULL
                    │   AND outcome_received_at >= max - window_days
                    ├── PerformativeDriftDetector.add() per row
                    └── _compute_psi(reference, current)
                           │
                   drift_scores table
                   (drift_score, psi_score, threshold_crossed, window_days)
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Drift detector | CB-PDD | Causally targets performative feedback loops; see ADR-001 |
| Trial length τ | 1000 | Balances detection speed (Day 14) and false-positive risk; see ADR-001 |
| Window anchor | max(outcome_received_at) | Works with historical simulation data; wall-clock NOW() would exclude all historical rows |
| Label delay modeling | outcome_received_at ≠ predicted_at | Captures realistic 14–30 day feedback delay |
| MLflow alias | @champion | MLflow 3.x API; never stage="Production" (deprecated since 2.9) |
| Async DB driver | asyncpg for FastAPI, psycopg2 for sync tasks | AsyncPG incompatible with sync SQLAlchemy used by Airflow tasks and dashboard |

---

## Production Limitations

1. **Label delay sensitivity**: CB-PDD only consumes labeled rows. If outcome labels arrive with >30 days delay (realistic for loan defaults), the rolling window may contain too few labeled rows for statistical power. Mitigation: increase `DRIFT_WINDOW_DAYS` or use τ=500 for faster detection.

2. **τ recalibration**: τ=1000 was calibrated at n_per_day=1000. At lower volumes (e.g. 200/day), the first trial period takes 5× longer to complete. The τ sensitivity chart in the dashboard makes this trade-off visible.

3. **Checkerboard routing overhead**: the 10% split-path router makes 10% of lending decisions sub-optimally (by design) to enable CB-PDD measurement. This is acceptable in simulation; a live deployment would negotiate this rate with the business.

4. **Single-node Airflow**: LocalExecutor is used for simplicity. At production scale, CeleryExecutor with Redis and a dedicated worker pool is appropriate.
