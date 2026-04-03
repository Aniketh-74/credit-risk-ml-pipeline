# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-24)

**Core value:** Detect performative drift — when the model's own predictions cause the future distribution to shift — and automatically close the loop with bias-corrected retraining, before the model silently degrades.
**Current focus:** Phase 2 — Champion Model

## Current Position

Phase: 2 of 7 (Champion Model)
Plan: 3 of 3 in current phase — COMPLETE
Status: Phase 2 complete — all 3 plans done; champion model registered with @champion alias, AUC=0.8655
Last activity: 2026-04-03 — Plan 02-03 complete (MLflow registry promotion, @champion alias, CB-PDD smoke test passing at all τ)

Progress: [██████████] 100% (Phase 2)

## Performance Metrics

**Velocity:**
- Total plans completed: 5
- Average duration: ~12 min
- Total execution time: ~0.85 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-solid-ground | 4 | ~40 min | ~10 min |
| 02-champion-model | 2 | ~65 min | ~32 min |

**Recent Trend:**
- Last 5 plans: 01-02 (5 min), 01-03 (15 min), 01-04 (15 min), 02-01 (20 min)
- Trend: stable

*Updated after each plan completion*
| Phase 01 P01-01 | 5 | 2 tasks | 9 files |
| Phase 01 P01-04 | 15 | 1 task | 8 files |
| Phase 02 P02-01 | 20 | 1 task | 2 files |
| Phase 02 P02-02 | 45 | 2 tasks | 9 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Design]: Split-path prediction router (API-02) must be designed with Phase 1 schema support — `path` column in `predictions` table needed from day one or CB-PDD cannot distinguish intervention vs model-assigned predictions
- [Design]: Class imbalance (7% default rate) addressed in Phase 2 via SMOTE before Phase 4 builds CB-PDD validation on top — avoids late discovery of detector not firing
- [Design]: CB-PDD runs as an Airflow task reading from PostgreSQL `predictions` table — not embedded in FastAPI and not a microservice
- [Tooling]: MLflow 3.x — use `@champion` alias, never `stage="Production"` (deprecated since 2.9)
- [Tooling]: Airflow 3.x — `PythonOperator` is in `apache-airflow-providers-standard` (separate package)
- [Phase 01-solid-ground]: No version: key in docker-compose.yml — Docker Compose v2 plugin does not require it
- [Phase 01-solid-ground]: start_period: 30s on postgres healthcheck allows init-multiple-dbs.sh to finish before dependents probe
- [Phase 01-solid-ground]: MLflow artifact root uses named volume /mlflow/artifacts — decouples artifacts from container, enables GCS swap in Phase 7
- [Phase 01]: sqlalchemy bumped to 2.0.48 for Python 3.14 Union typing compatibility
- [Phase 01]: predictions.path column present from migration 0001 — CB-PDD router requires it before any prediction is written
- [Phase 01-03]: All secret placeholders use angle-bracket syntax so grep reveals any accidental literal value
- [Phase 01-03]: pyproject.toml uses E501 ignored in ruff — black handles line length, ruff handles everything else
- [Phase 01-04]: Plotly chosen over Altair/matplotlib for drift chart — richer interactivity and threshold annotation support
- [Phase 01-04]: Dashboard component render() functions accept list[dict] — Phase 6 passes DB rows with zero UI code changes
- [Phase 01-04]: Dockerfile.dashboard updated with plotly==5.24.1 and pandas==2.2.3
- [Phase 02-01 EDA]: clip(upper=17) for NumberOfTimes90DaysLate and NumberOfTime30-59DaysPastDueNotWorse — 96/98 are sentinel codes, not real delinquency counts
- [Phase 02-01 EDA]: Median imputation for MonthlyIncome (19.8% NaN) — right-skewed, fitted on train only; MonthlyIncome_was_missing flag preserves missingness signal
- [Phase 02-01 EDA]: SMOTE applied after train/test split only — pre-split SMOTE causes test-set synthetic sample leakage
- [Phase 02-01 EDA]: Spearman correlation used for heatmap — credit features violate Pearson's linearity and normality assumptions
- [Phase 02-01 EDA]: Retain all 3 late-payment count features despite ~0.98 pairwise correlation — LightGBM handles multicollinearity; each captures distinct timing signal
- [Phase 02-02]: build_train_test() returns 5-tuple including fitted SimpleImputer — scoring API needs same imputer statistics for live prediction requests
- [Phase 02-02]: is_unbalance=False in LightGBM — SMOTE already balances training set to 50/50; double-applying minority upweighting degrades model quality
- [Phase 02-02]: MLflow tests use sqlite:// backend instead of file:// — MLflow 3.x rejects file:// URIs on Windows with backslash paths; SQLite recommended by MLflow deprecation warnings
- [Phase 02-02]: compute_auc takes (y_true, y_score) not (model, X_test, y_test) — keeps evaluate.py stateless and reusable for non-LightGBM models
- [Phase 02-03]: τ=1000 selected for CB-PDD Phase 4 baseline — all three τ values (500, 1000, 2000) produced identical detection rates (7/7 windows, first detection day 14) on the 30-day denial loop simulation; τ=1000 aligns with paper default and can be tuned in Phase 4 experiments
- [Phase 02-03]: CB-PDD smoke test requires n_per_day=1000 for statistical power — Mann-Whitney U needs ≥τ accumulated samples per test window; n_per_day=100 produced 0 detections due to insufficient group size (100 vs 700 reference)

### Pending Todos

None yet.

### Blockers/Concerns

- [Resolved 2026-04-03]: CB-PDD τ=1000 validated — 7/7 detection windows fired on 30-day denial loop simulation; all τ ∈ {500, 1000, 2000} produced equal detection rates, τ=1000 selected as Phase 4 default
- [Open question]: MLflow 3.x + GCS artifact store on Cloud Run bucket permissions pattern — verify before Phase 7 deployment
- [Open question]: Exact version of `apache-airflow-providers-standard` shipping with Airflow 3.1.8 — verify before writing DAGs in Phase 5

## Session Continuity

Last session: 2026-04-01
Stopped at: Completed 02-02-PLAN.md — LightGBM + SMOTE training pipeline with MLflow autolog and test suite
Resume file: None
