# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-24)

**Core value:** Detect performative drift — when the model's own predictions cause the future distribution to shift — and automatically close the loop with bias-corrected retraining, before the model silently degrades.
**Current focus:** Phase 1 — Solid Ground

## Current Position

Phase: 1 of 7 (Solid Ground)
Plan: 2 of 4 in current phase
Status: In progress — plan 02 complete
Last activity: 2026-03-28 — Plan 01-02 complete (Docker infrastructure, FastAPI health stub)

Progress: [██░░░░░░░░] 20%

## Performance Metrics

**Velocity:**
- Total plans completed: 1
- Average duration: 5 min
- Total execution time: 0.1 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-solid-ground | 1 | 5 min | 5 min |

**Recent Trend:**
- Last 5 plans: 01-02 (5 min)
- Trend: —

*Updated after each plan completion*

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

### Pending Todos

None yet.

### Blockers/Concerns

- [Open question]: CB-PDD τ parameter has no principled calibration guidance from the paper — validate 5 trials/day sensitivity at τ=1000 early in Phase 2 before orchestration is built around it
- [Open question]: MLflow 3.x + GCS artifact store on Cloud Run bucket permissions pattern — verify before Phase 7 deployment
- [Open question]: Exact version of `apache-airflow-providers-standard` shipping with Airflow 3.1.8 — verify before writing DAGs in Phase 5

## Session Continuity

Last session: 2026-03-28
Stopped at: Completed 01-02-PLAN.md — Docker infrastructure and FastAPI health stub
Resume file: None
