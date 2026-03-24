# credit-risk-ml-pipeline

## What This Is

A production-grade MLOps pipeline that implements performative drift detection for credit risk scoring — based on arxiv 2412.10545 ("Identifying Predictions That Influence the Future"). Unlike standard ML pipelines that detect drift after it happens, this system detects when the model's own predictions are causing the future data distribution to shift (feedback loops in lending), then automatically triggers bias-corrected retraining. Built as a portfolio project targeting AI/ML Engineer and Data Engineer roles.

## Core Value

Detect performative drift — when the model's predictions cause the future distribution to shift — and automatically close the loop with bias-corrected retraining, before the model silently degrades.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Simulate two performative drift types: denial loop (model denies loans → denied group never defaults → model sees them as high risk) and score gaming loop (applicants game features → population shifts)
- [ ] Implement the performative drift detection algorithm from arxiv 2412.10545 faithfully as a production Python module
- [ ] Train a baseline credit risk model on Give Me Credit (Kaggle, 150k rows) with MLflow experiment tracking
- [ ] FastAPI service that scores loan applications and logs predictions + outcomes
- [ ] Performative drift score computed continuously; alert + auto-retrain trigger when threshold crossed
- [ ] MLflow auto-retraining run fired on drift alert, new model promoted if AUC improves
- [ ] Airflow DAG orchestrating: daily score → detect drift → retrain if needed → promote model
- [ ] Streamlit dashboard showing drift score evolution over 30 days of simulated data, model version history, alert log
- [ ] Full Docker + docker-compose local setup
- [ ] GCP Cloud Run deployment with GitHub Actions CI/CD
- [ ] Recruiter-ready README with architecture diagram, live demo link, "what I learned" section

### Out of Scope

- Real-time Kafka streaming — batch/scheduled pipeline is sufficient to demonstrate the concept
- Multi-model ensemble — single gradient boosting model keeps focus on the drift story
- Auth/multi-tenant API — single-user demo is sufficient for portfolio
- Mobile/frontend beyond Streamlit — keeps scope achievable in 30 days

## Context

- **Research paper:** arxiv 2412.10545 — "Identifying Predictions That Influence the Future: Detecting Performative Concept Drift in Data Streams" (December 2024). No production implementation exists. Algorithm faithfully implemented, then wrapped in production MLOps.
- **Dataset:** Give Me Credit (Kaggle) — 150k rows, real credit default features (income, debt ratio, age, delinquency history). Performative drift simulated on top by replaying predictions as if they affected future applicant populations.
- **Differentiation:** 99% of ML portfolios use static datasets and never model feedback loops. This project implements a novel 2024 algorithm that financial ML teams face in production but have no off-the-shelf solution for.
- **Target audience:** Hiring managers at fintechs, banks, MLOps platforms, and AI/ML teams who care about model reliability in production.
- **Developer:** Aniketh Mahadik — 1-2 hours/day, daily commits, 30-day timeline (Phase 1 of 90-day portfolio roadmap).

## Constraints

- **Timeline:** 30 days — scope must be completable in 1-2 hours/day
- **Tech stack:** Python 3.11+, FastAPI, MLflow, Airflow, PostgreSQL, Docker, GCP Cloud Run, GitHub Actions — fixed per portfolio plan
- **Paper fidelity:** Core algorithm implemented faithfully; production wrapper (API, orchestration, monitoring) added on top
- **Daily commits:** Every day must produce at least one meaningful commit — tasks sized accordingly

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Give Me Credit dataset | 150k rows, real features, well-known to hiring managers | — Pending |
| Alert + auto-retrain (not counterfactual) | Simpler to explain in interviews, closer to what companies deploy | — Pending |
| Both drift types (denial + gaming) | Demonstrates two distinct causal mechanisms, more research depth | — Pending |
| Full stack: Airflow + MLflow + FastAPI + Streamlit | Signals both ML Engineer and Data Engineer competency | — Pending |
| arxiv 2412.10545 — core algorithm + production wrapper | Faithful to paper, adds production value existing implementations lack | — Pending |

---
*Last updated: 2026-03-24 after initialization*
