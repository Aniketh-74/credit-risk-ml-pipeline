# Research Summary — credit-risk-ml-pipeline

**Synthesized:** 2026-03-24
**Sources:** STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md

---

## Key Findings

### Stack

**Core:** Python 3.11, LightGBM 4.6, FastAPI 0.135 + Pydantic v2, MLflow 3.10, Airflow 3.1, PostgreSQL 16, Streamlit 1.44, Docker, GCP Cloud Run

**Critical finding — no library does performative drift detection.** Evidently, River, Alibi-Detect, NannyML all detect covariate/concept drift. CB-PDD (arxiv 2412.10545, AAAI 2025) is the only published algorithm for causal feedback-loop drift. The research code exists at GitHub (BrandonGower-Winter/CheckerboardDetection v1.0) but it is experiment-grade. We port it into `src/drift/cb_pdd.py` as a clean production module — ~200-300 lines, `numpy` + `scipy` only.

**Two gotchas on versions:** MLflow 3.x uses model aliases (`@champion`) not stages — stages are deprecated since 2.9. Airflow 3.x moves PythonOperator to `apache-airflow-providers-standard` (separate package).

### Table Stakes

Every serious MLOps portfolio must have: trained model with MLflow tracking, FastAPI serving endpoint, Docker + compose, PostgreSQL prediction logging, Airflow DAG orchestration, CI/CD, Cloud Run deployment, recruiter README with architecture diagram.

### Differentiators

The entire thesis is the CB-PDD implementation. Two simulation modes make the story richer: **denial loop** (denied applicants never default → model sees them as risk-free → loop) and **score gaming loop** (applicants game features → population shifts). Showing both modes trigger CB-PDD while a random predictor does not is the empirical proof. No competitor portfolio has this.

### Watch Out For

**Top 3 pitfalls that will derail the project:**

1. **CB-PDD needs a split-path prediction router from day one.** The algorithm requires a fraction of predictions to be CheckerBoard-assigned (intervention), not model-assigned. This must be in the FastAPI design from Phase 1 — retrofitting it later requires rewriting the prediction pipeline. `mix=0.1` (10% intervention rate) is the starting value.

2. **Class imbalance (7% default rate on Give Me Credit) can break CB-PDD.** Apply SMOTE during training and validate the detector fires on a synthetic denial loop before building orchestration around it. Do not discover this in Phase 4.

3. **MLflow's official Docker image lacks psycopg2.** Build a custom `Dockerfile.mlflow` that adds `pip install psycopg2-binary`. Use separate Postgres databases for Airflow metadata, MLflow tracking, and application data. Resolve docker-compose configuration completely before writing ML code.

**Other pitfalls to design around early:**
- Label delay: true credit defaults arrive 30-90 days after origination. Model this with `outcome_received_at` separate from `predicted_at` in the schema.
- τ parameter has no principled calibration guidance from the paper. Expose as config, document the choice, add sensitivity chart.
- Never use `stage="Production"` in MLflow code. Use `client.set_registered_model_alias(name, "champion", version)`.
- Add `catchup=False` to all Airflow DAGs.
- Load the model at FastAPI startup (lifespan context), not per-request. Set Cloud Run min-instances=1 for the demo.

### Architecture Decision

CB-PDD runs as an **Airflow task** (not embedded in FastAPI, not as a microservice). It reads from the PostgreSQL `predictions` table over a rolling window, writes to `drift_scores`, and returns a task ID to a BranchPythonOperator. This is the correct production placement: batch computation, independently testable, visible in Airflow UI, wired to the retrain branch.

### Build Order

1. PostgreSQL schema + Alembic migrations (everything depends on this)
2. Baseline model training + MLflow tracking
3. FastAPI scoring endpoint with split-path prediction router (CheckerBoard + model)
4. Feedback loop simulator (denial + gaming modes)
5. CB-PDD module — implement, unit-test against synthetic data, validate it fires
6. Airflow DAG — batch score → drift check → retrain branch → model promote
7. Streamlit dashboard — reads PostgreSQL + MLflow REST API
8. Docker Compose — full local integration
9. CI/CD + GCP Cloud Run deployment

---

## Open Questions

1. CB-PDD parameter τ: need to validate that 5 trials/day (at ~5k instances/day with τ=1000) is sensitive enough for 30-day simulation. Run quick experiment in Phase 2 before building orchestration.
2. MLflow 3.x + GCS artifact store on Cloud Run: verify bucket permissions pattern (less documented than S3 equivalent).
3. Airflow 3.x `apache-airflow-providers-standard` exact version that ships with 3.1.8 — verify before writing DAGs.

---
*Research synthesized from 4 parallel agents: STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md*
