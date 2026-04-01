# Requirements: credit-risk-ml-pipeline

**Defined:** 2026-03-24
**Core Value:** Detect performative drift — when the model's own predictions cause the future distribution to shift — and automatically close the loop with bias-corrected retraining, before the model silently degrades.

---

## v1 Requirements

### Infrastructure

- [x] **INFRA-01**: PostgreSQL schema with tables for predictions, outcomes, drift_scores, and alerts — with Alembic migrations
- [x] **INFRA-02**: Full docker-compose stack (FastAPI, Airflow, MLflow, PostgreSQL, Streamlit) running with a single `docker-compose up`
- [x] **INFRA-03**: Custom MLflow Dockerfile that includes psycopg2 (official image lacks it)
- [ ] **INFRA-04**: Environment variable configuration via .env with a complete .env.example

### Baseline Model

- [ ] **MODEL-01**: LightGBM model trained on Give Me Credit dataset (150k rows, AUC > 0.85) with SMOTE for class imbalance
- [ ] **MODEL-02**: All training runs logged to MLflow — hyperparameters, AUC, precision-recall curve, feature importances
- [ ] **MODEL-03**: Trained model registered in MLflow Registry and promoted using `@champion` alias (no deprecated stages API)
- [x] **MODEL-04**: EDA notebook for Give Me Credit — missing value analysis (MonthlyIncome 19.8% NaN), class distribution, feature correlations

### Serving API

- [ ] **API-01**: FastAPI POST /score endpoint that accepts loan application features and returns probability score + approve/deny decision + model version
- [ ] **API-02**: Split-path prediction router — configurable `mix` parameter (default 0.1) routes a fraction of requests through CheckerBoard predictor for CB-PDD intervention
- [ ] **API-03**: POST /outcome endpoint to record actual default outcome tied to a prediction_id
- [ ] **API-04**: GET /health endpoint returning service status and current champion model version
- [ ] **API-05**: Async prediction logging to PostgreSQL via BackgroundTask (non-blocking)
- [ ] **API-06**: Model loaded at startup via FastAPI lifespan context (not per-request)

### Performative Drift Detection

- [ ] **DRIFT-01**: CB-PDD algorithm (`src/drift/cb_pdd.py`) faithfully ported from arxiv 2412.10545 — CheckerBoardPredictor, DensityChangeTracker, PerformativeDriftDetector classes
- [ ] **DRIFT-02**: CB-PDD parameters (τ, w, α) exposed as config — not hardcoded
- [ ] **DRIFT-03**: CB-PDD unit tests pass on synthetic known-drift data (detector fires) and synthetic no-drift data (detector does not fire)
- [ ] **DRIFT-04**: Drift scorer (`src/drift/scorer.py`) reads predictions table from PostgreSQL and returns drift score + is_drift flag
- [ ] **DRIFT-05**: PSI (Population Stability Index) computed alongside CB-PDD as a standard baseline comparison

### Feedback Loop Simulation

- [ ] **SIM-01**: Denial loop simulator (`src/simulation/denial_loop.py`) — denied applicants removed from next day's pool, modeling the self-defeating feedback loop
- [ ] **SIM-02**: Score gaming simulator (`src/simulation/score_gaming.py`) — applicants near decision boundary shift features toward approval, modeling the self-fulfilling feedback loop
- [ ] **SIM-03**: Null control experiment — random predictor does NOT trigger CB-PDD drift detection (proves algorithm distinguishes performative from random variation)
- [ ] **SIM-04**: Label delay modeled — `outcome_received_at` column separate from `predicted_at`; drift detector only consumes instances where both timestamps exist

### Orchestration

- [ ] **ORCH-01**: Airflow DAG (`credit_risk_daily`) with 5 tasks: feedback_simulate → batch_score → drift_check → retrain_or_skip (BranchPythonOperator) → promote_if_improved
- [ ] **ORCH-02**: BranchPythonOperator returns `trigger_retrain` or `skip_retrain` based on drift_score vs threshold
- [ ] **ORCH-03**: Retraining task fires MLflow run, trains new model on recent labeled data, logs AUC
- [ ] **ORCH-04**: Promotion task compares new AUC vs champion AUC — only promotes if improvement (AUC delta > 0)
- [ ] **ORCH-05**: DAG validation test in CI — imports all DAG files and checks for parse errors
- [ ] **ORCH-06**: `catchup=False` on all DAGs; meaningful task-level retries configured

### Monitoring Dashboard

- [ ] **DASH-01**: Streamlit dashboard showing drift score time series over 30-day simulation window with threshold line
- [ ] **DASH-02**: Model version history table (reads MLflow REST API — no SDK needed in Streamlit)
- [ ] **DASH-03**: Alert log table showing when drift fired, retrain outcome, and whether model was promoted
- [ ] **DASH-04**: PSI vs CB-PDD comparison chart showing the two detection approaches side by side
- [ ] **DASH-05**: τ sensitivity analysis chart — drift detection latency vs τ values {500, 1000, 2000}

### Deployment & CI/CD

- [ ] **DEPLOY-01**: GitHub Actions CI — lint (ruff), tests (pytest), Docker build on every PR
- [ ] **DEPLOY-02**: GitHub Actions deploy — push to GCR and deploy FastAPI to Cloud Run on merge to main
- [ ] **DEPLOY-03**: Cloud Run min-instances=1 for FastAPI (prevents cold-start demo failures)
- [ ] **DEPLOY-04**: Model artifact baked into Docker image at build time (eliminates cold-start MLflow fetch)
- [ ] **DEPLOY-05**: Streamlit dashboard deployed to Cloud Run or Streamlit Community Cloud with public URL

### Documentation

- [ ] **DOC-01**: README with one-line description, Mermaid architecture diagram, tech stack badges, quick start (<5 commands), "what I learned" section, live demo URL
- [ ] **DOC-02**: ADR-001 documenting why CB-PDD was chosen over standard drift detectors
- [ ] **DOC-03**: Architecture doc (`docs/architecture.md`) explaining the feedback loop simulation mechanism and null control experiment
- [ ] **DOC-04**: README section on production limitations — label delay, τ calibration, PII handling note

---

## v2 Requirements

### Bias-Corrected Retraining

- **BIAS-01**: Inverse propensity score weighting applied during retraining to correct for selection bias from denial loop
- **BIAS-02**: Reject inference — imputing outcomes for denied applicants using propensity-weighted estimates

### Explainability

- **EXPL-01**: SHAP summary plot in README showing top feature importances pre- and post-drift
- **EXPL-02**: Per-prediction SHAP values logged to MLflow as artifacts

### Advanced Monitoring

- **MON-01**: Evidently data quality report as supplementary monitoring alongside CB-PDD
- **MON-02**: Email/Slack alert on drift threshold crossing (webhook integration)

---

## Out of Scope

| Feature | Reason |
|---------|--------|
| Kafka real-time streaming | Adds 1-2 weeks of infrastructure, obscures the drift story; batch pipeline is sufficient and closer to how credit lenders actually operate |
| Multi-model ensemble | Complicates drift narrative; single LightGBM model version history is cleaner for the dashboard story |
| Auth / multi-tenant API | Adds 3-5 days of plumbing invisible to hiring managers; single-user demo is sufficient |
| Kubernetes deployment | Overkill for 30-day project; Cloud Run gives the same "deployed in cloud" signal |
| Optuna hyperparameter tuning | Squeezing 0.5% AUC improvement is not the story |
| Real credit bureau PII data | Legal complexity, access barriers; Give Me Credit is sufficient and well-known |
| Full Evidently dashboard | Would dilute focus from CB-PDD as the narrative center |
| Online learning / streaming retraining | Changes architecture fundamentally; CB-PDD was designed for batch scenarios |

---

## Traceability

| Requirement | Phase | Phase Name | Status |
|-------------|-------|------------|--------|
| INFRA-01 | Phase 1 | Solid Ground | Pending |
| INFRA-02 | Phase 1 | Solid Ground | Complete |
| INFRA-03 | Phase 1 | Solid Ground | Pending |
| INFRA-04 | Phase 1 | Solid Ground | Pending |
| MODEL-01 | Phase 2 | Champion Model | Pending |
| MODEL-02 | Phase 2 | Champion Model | Pending |
| MODEL-03 | Phase 2 | Champion Model | Pending |
| MODEL-04 | Phase 2 | Champion Model | Complete |
| API-01 | Phase 3 | The Scoring API | Pending |
| API-02 | Phase 3 | The Scoring API | Pending |
| API-03 | Phase 3 | The Scoring API | Pending |
| API-04 | Phase 3 | The Scoring API | Pending |
| API-05 | Phase 3 | The Scoring API | Pending |
| API-06 | Phase 3 | The Scoring API | Pending |
| DRIFT-01 | Phase 4 | Proving the Feedback Loop | Pending |
| DRIFT-02 | Phase 4 | Proving the Feedback Loop | Pending |
| DRIFT-03 | Phase 4 | Proving the Feedback Loop | Pending |
| DRIFT-04 | Phase 4 | Proving the Feedback Loop | Pending |
| DRIFT-05 | Phase 4 | Proving the Feedback Loop | Pending |
| SIM-01 | Phase 4 | Proving the Feedback Loop | Pending |
| SIM-02 | Phase 4 | Proving the Feedback Loop | Pending |
| SIM-03 | Phase 4 | Proving the Feedback Loop | Pending |
| SIM-04 | Phase 4 | Proving the Feedback Loop | Pending |
| ORCH-01 | Phase 5 | Autonomous Daily Cycle | Pending |
| ORCH-02 | Phase 5 | Autonomous Daily Cycle | Pending |
| ORCH-03 | Phase 5 | Autonomous Daily Cycle | Pending |
| ORCH-04 | Phase 5 | Autonomous Daily Cycle | Pending |
| ORCH-05 | Phase 5 | Autonomous Daily Cycle | Pending |
| ORCH-06 | Phase 5 | Autonomous Daily Cycle | Pending |
| DASH-01 | Phase 6 | The Drift Story Visualized | Pending |
| DASH-02 | Phase 6 | The Drift Story Visualized | Pending |
| DASH-03 | Phase 6 | The Drift Story Visualized | Pending |
| DASH-04 | Phase 6 | The Drift Story Visualized | Pending |
| DASH-05 | Phase 6 | The Drift Story Visualized | Pending |
| DEPLOY-01 | Phase 7 | Live and Recruiter-Ready | Pending |
| DEPLOY-02 | Phase 7 | Live and Recruiter-Ready | Pending |
| DEPLOY-03 | Phase 7 | Live and Recruiter-Ready | Pending |
| DEPLOY-04 | Phase 7 | Live and Recruiter-Ready | Pending |
| DEPLOY-05 | Phase 7 | Live and Recruiter-Ready | Pending |
| DOC-01 | Phase 7 | Live and Recruiter-Ready | Pending |
| DOC-02 | Phase 7 | Live and Recruiter-Ready | Pending |
| DOC-03 | Phase 7 | Live and Recruiter-Ready | Pending |
| DOC-04 | Phase 7 | Live and Recruiter-Ready | Pending |

**Coverage:**
- v1 requirements: 44 total
- Mapped to phases: 44
- Unmapped: 0

---
*Requirements defined: 2026-03-24*
*Last updated: 2026-03-24 after roadmap creation*
