# Roadmap: credit-risk-ml-pipeline

## Overview

This pipeline goes from an empty repo to a live, recruiter-ready portfolio artifact demonstrating a novel (2024) performative drift detection algorithm running in production. Phase 1 lays a schema and Docker foundation that every later phase depends on. Phase 2 trains the champion model with SMOTE imbalance correction so Phase 4's CB-PDD validation has a calibrated predictor to work with. Phase 3 builds the scoring API — critically including the split-path prediction router that CB-PDD requires — before the simulator or detector are written. Phase 4 ports the arxiv algorithm faithfully and proves both feedback loop types trigger it while random noise does not. Phase 5 wires everything into an Airflow DAG that runs the daily drift-detect-retrain cycle autonomously. Phase 6 surfaces the story visually in a Streamlit dashboard. Phase 7 deploys publicly and delivers the README that makes a hiring manager stop scrolling.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Solid Ground** - PostgreSQL schema, Alembic migrations, and full Docker Compose stack running with a single command
- [ ] **Phase 2: Champion Model** - LightGBM trained on Give Me Credit with SMOTE, AUC > 0.85, logged to MLflow Registry under `@champion` alias
- [ ] **Phase 3: The Scoring API** - FastAPI service with split-path prediction router, outcome logging, and model-at-startup loading
- [ ] **Phase 4: Proving the Feedback Loop** - CB-PDD algorithm ported from arxiv 2412.10545, both simulators implemented, unit tests prove the detector fires on drift and not on noise
- [ ] **Phase 5: Autonomous Daily Cycle** - Airflow DAG orchestrating simulate → score → detect → branch → retrain → promote end-to-end without manual intervention
- [ ] **Phase 6: The Drift Story Visualized** - Streamlit dashboard showing drift score evolution, model version history, alert log, and CB-PDD vs PSI comparison
- [ ] **Phase 7: Live and Recruiter-Ready** - CI/CD pipeline, GCP Cloud Run deployment, and a README that explains what was built and why it matters

## Phase Details

### Phase 1: Solid Ground
**Goal**: Every downstream service has a stable, migrated schema and a single-command local environment to run in
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04
**Success Criteria** (what must be TRUE):
  1. `docker-compose up` starts all five services (FastAPI, Airflow, MLflow, PostgreSQL, Streamlit) without manual intervention and all health checks pass
  2. PostgreSQL contains the `predictions`, `outcomes`, `drift_scores`, and `alerts` tables, each with the correct columns and constraints, created via Alembic migration
  3. The custom MLflow Dockerfile successfully installs psycopg2 and MLflow can write experiment runs to the PostgreSQL backend
  4. All secrets and service URLs are read from `.env`; `.env.example` documents every required variable with no hardcoded credentials in any source file
**Plans**: 4 plans

Plans:
- [ ] 01-01-PLAN.md — SQLAlchemy 2.0 models and Alembic initial schema migration (Wave 1)
- [ ] 01-02-PLAN.md — Custom MLflow Dockerfile and docker-compose stack assembly (Wave 1)
- [ ] 01-03-PLAN.md — .env configuration, pyproject.toml, and full-stack smoke test (Wave 2)
- [ ] 01-04-PLAN.md — Production-grade Streamlit dashboard UI with frontend-design skill (Wave 2)

### Phase 2: Champion Model
**Goal**: A trained LightGBM credit risk model exists in MLflow Registry under the `@champion` alias, with SMOTE applied to handle class imbalance, ready to serve predictions and be retrained against
**Depends on**: Phase 1
**Requirements**: MODEL-01, MODEL-02, MODEL-03, MODEL-04
**Success Criteria** (what must be TRUE):
  1. The EDA notebook documents Give Me Credit's class distribution (7% default rate), missing value profile (MonthlyIncome 19.8% NaN), and feature correlations — making the imbalance problem explicit before training
  2. A LightGBM model trained with SMOTE achieves AUC > 0.85 on the held-out test set, logged as an MLflow run with hyperparameters, precision-recall curve, and feature importances
  3. The trained model is registered in MLflow Registry and the `@champion` alias resolves to it — no deprecated `stage="Production"` references exist anywhere in the codebase
  4. A quick synthetic validation confirms CB-PDD fires at least once on a simulated denial loop using this model's score distribution — catching the class imbalance pitfall before Phase 4 builds on it
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md — EDA notebook: class distribution, missing values, Spearman correlations, outlier analysis (Wave 1)
- [ ] 02-02-PLAN.md — LightGBM training pipeline with SMOTE, MLflow autolog, PR curve artifact, and test suite (Wave 1)
- [ ] 02-03-PLAN.md — MLflow Registry promotion using `@champion` alias and CB-PDD smoke test at τ∈{500,1000,2000} (Wave 2)

### Phase 3: The Scoring API
**Goal**: A FastAPI service that scores loan applications, routes a configurable fraction through the CheckerBoard predictor (required by CB-PDD), logs predictions asynchronously, and records outcomes — all with the model loaded once at startup
**Depends on**: Phase 2
**Requirements**: API-01, API-02, API-03, API-04, API-05, API-06
**Success Criteria** (what must be TRUE):
  1. `POST /score` returns a probability score, approve/deny decision, and the current `@champion` model version within a single request, with no per-request model loading
  2. The split-path prediction router routes `mix` fraction of requests (default 0.1) through the CheckerBoard predictor, with `mix` configurable via environment variable — every prediction record in PostgreSQL includes a `path` column indicating which router handled it
  3. `POST /outcome` records an actual default outcome tied to a `prediction_id` and sets `outcome_received_at` separately from `predicted_at` (label delay is modeled from day one)
  4. `GET /health` returns service status and the current `@champion` model version, confirming the API knows which model version is active
  5. Prediction logging to PostgreSQL happens via BackgroundTask and does not add latency to the scoring response
**Plans**: TBD

Plans:
- [ ] 03-01: FastAPI skeleton with lifespan model loading, health endpoint, and Pydantic schemas
- [ ] 03-02: Split-path prediction router (CheckerBoard + model path) and `/score` endpoint
- [ ] 03-03: `/outcome` endpoint, async BackgroundTask logging, and API integration tests

### Phase 4: Proving the Feedback Loop
**Goal**: The CB-PDD algorithm is faithfully ported from arxiv 2412.10545 into a production Python module; both feedback loop simulators are implemented; unit tests prove the detector fires on denial loop and score gaming drift while remaining silent on random variation
**Depends on**: Phase 3
**Requirements**: DRIFT-01, DRIFT-02, DRIFT-03, DRIFT-04, DRIFT-05, SIM-01, SIM-02, SIM-03, SIM-04
**Success Criteria** (what must be TRUE):
  1. `src/drift/cb_pdd.py` contains `CheckerBoardPredictor`, `DensityChangeTracker`, and `PerformativeDriftDetector` classes implementing the arxiv 2412.10545 algorithm — no external drift detection libraries used
  2. CB-PDD parameters (τ, w, α) are read from config and not hardcoded; the chosen τ value is documented with rationale
  3. Unit tests pass on three synthetic datasets: denial loop data (detector fires), score gaming data (detector fires), random predictor data (detector does NOT fire) — proving the algorithm distinguishes performative drift from random variation
  4. The drift scorer reads the `predictions` table from PostgreSQL over a rolling window, computes both the CB-PDD score and PSI, and returns `drift_score`, `psi_score`, and `is_drift` flag
  5. `outcome_received_at` is treated as the label availability timestamp — the drift detector only consumes prediction rows where both `predicted_at` and `outcome_received_at` are non-null
**Plans**: TBD

Plans:
- [ ] 04-01: Denial loop and score gaming simulators with label delay modeling
- [ ] 04-02: CB-PDD algorithm port — CheckerBoardPredictor, DensityChangeTracker, PerformativeDriftDetector
- [ ] 04-03: Drift scorer integrating PostgreSQL reads, PSI computation, and CB-PDD output
- [ ] 04-04: Unit test suite — synthetic drift fires, synthetic no-drift silent, null control experiment

### Phase 5: Autonomous Daily Cycle
**Goal**: An Airflow DAG runs the complete detect-retrain cycle daily without manual steps — simulating feedback, scoring a batch, checking drift, branching to retrain or skip, and promoting a better model if one is found
**Depends on**: Phase 4
**Requirements**: ORCH-01, ORCH-02, ORCH-03, ORCH-04, ORCH-05, ORCH-06
**Success Criteria** (what must be TRUE):
  1. The `credit_risk_daily` DAG is visible in the Airflow UI with five tasks in the correct order: `feedback_simulate → batch_score → drift_check → retrain_or_skip → promote_if_improved`
  2. When drift score exceeds threshold, `BranchPythonOperator` routes to `trigger_retrain`; when below threshold, it routes to `skip_retrain` — confirmed by running both branches against synthetic threshold scenarios
  3. The retrain task fires an MLflow run on recent labeled data, logs AUC, and the promote task only sets the `@champion` alias if the new AUC exceeds the current champion's AUC
  4. DAG imports cleanly in CI with no parse errors; `catchup=False` is set and each task has retry configuration
**Plans**: TBD

Plans:
- [ ] 05-01: Airflow DAG scaffold with `feedback_simulate` and `batch_score` tasks
- [ ] 05-02: `drift_check` task and `BranchPythonOperator` routing logic
- [ ] 05-03: `retrain_or_skip` and `promote_if_improved` tasks with MLflow AUC comparison
- [ ] 05-04: DAG validation test in CI and retry/catchup configuration

### Phase 6: The Drift Story Visualized
**Goal**: A Streamlit dashboard that lets anyone — including a hiring manager with no access to the codebase — see the drift score evolve over 30 days, watch the model version change after retraining, and compare CB-PDD to PSI as a baseline
**Depends on**: Phase 5
**Requirements**: DASH-01, DASH-02, DASH-03, DASH-04, DASH-05
**Success Criteria** (what must be TRUE):
  1. The drift score time series chart shows 30 days of simulated data with the alert threshold as a horizontal reference line — a viewer can see exactly when drift crossed the threshold
  2. The model version history table shows each `@champion` promotion event, the date, and the AUC delta — sourced from the MLflow REST API without using the MLflow Python SDK
  3. The alert log table shows each drift event: when it fired, whether retraining ran, and whether the model was promoted
  4. A side-by-side chart compares PSI and CB-PDD drift scores on the same time axis, making it visually clear that CB-PDD detects the feedback-loop pattern that PSI misses (or detects later)
  5. The τ sensitivity chart shows drift detection latency across τ values of 500, 1000, and 2000 — giving the viewer intuition for the key tuning parameter
**Plans**: TBD

Plans:
- [ ] 06-01: Streamlit app skeleton with PostgreSQL data layer and drift score time series chart
- [ ] 06-02: Model version history table (MLflow REST API) and alert log table
- [ ] 06-03: PSI vs CB-PDD comparison chart and τ sensitivity analysis chart

### Phase 7: Live and Recruiter-Ready
**Goal**: The project is publicly deployed on GCP Cloud Run, CI/CD runs on every PR and merge, and the README tells the complete story of what was built, why it matters, and how to run it
**Depends on**: Phase 6
**Requirements**: DEPLOY-01, DEPLOY-02, DEPLOY-03, DEPLOY-04, DEPLOY-05, DOC-01, DOC-02, DOC-03, DOC-04
**Success Criteria** (what must be TRUE):
  1. Every PR triggers GitHub Actions: ruff linting passes, pytest suite passes, and Docker image builds successfully — a broken commit cannot merge
  2. Merging to `main` automatically pushes the FastAPI image to GCR and deploys to Cloud Run; the live `/score` endpoint responds to a curl request within 5 seconds (min-instances=1, model baked into image)
  3. The Streamlit dashboard is publicly accessible via a URL in the README and shows live drift score data without requiring the viewer to install anything
  4. The README contains: one-line project description, Mermaid architecture diagram, tech stack badges, a quick-start in fewer than 5 commands, a "what I learned" section covering CB-PDD and performative drift, a live demo URL, and a production limitations section covering label delay and τ calibration
  5. ADR-001 documents why CB-PDD was chosen over standard drift detectors, and `docs/architecture.md` explains the feedback loop simulation mechanism and null control experiment
**Plans**: TBD

Plans:
- [ ] 07-01: GitHub Actions CI workflow (lint, test, Docker build)
- [ ] 07-02: GitHub Actions deploy workflow (GCR push, Cloud Run deploy) with min-instances and baked model artifact
- [ ] 07-03: Streamlit deployment to Cloud Run or Streamlit Community Cloud with public URL
- [ ] 07-04: README, ADR-001, architecture doc, and production limitations section

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Solid Ground | 4/4 | Complete | 2026-03-28 |
| 2. Champion Model | 0/3 | In progress | - |
| 3. The Scoring API | 0/3 | Not started | - |
| 4. Proving the Feedback Loop | 0/4 | Not started | - |
| 5. Autonomous Daily Cycle | 0/4 | Not started | - |
| 6. The Drift Story Visualized | 0/3 | Not started | - |
| 7. Live and Recruiter-Ready | 0/4 | Not started | - |
