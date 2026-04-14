# credit-risk-ml-pipeline

A production ML system that detects **performative drift** — when a model's own predictions corrupt the future training distribution — and automatically closes the loop with drift-triggered retraining.

[![CI](https://github.com/Aniketh-74/credit-risk-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Aniketh-74/credit-risk-ml-pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LightGBM](https://img.shields.io/badge/model-LightGBM-orange)
![Airflow](https://img.shields.io/badge/orchestration-Airflow%203-red)
![MLflow](https://img.shields.io/badge/registry-MLflow%203-blueviolet)

---

## Architecture

```mermaid
graph LR
    A[Applicant] -->|POST /score| B[FastAPI\nsplit-path router]
    B -->|90%| C[LightGBM\n@champion]
    B -->|10%| D[CheckerBoard\nPredictor]
    C --> E[(PostgreSQL\npredictions)]
    D --> E
    E -->|label delay| F[(outcomes)]
    F --> G[Airflow DAG\ncredit_risk_daily]
    G -->|compute_drift| H{CB-PDD\nis_drift?}
    H -->|yes| I[retrain\nMLflow run]
    H -->|no| J[skip]
    I -->|AUC improved?| K[promote\n@champion]
    E --> L[Streamlit\ndashboard]
    F --> L
```

**Five services, one command:**
```bash
docker compose up
```

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/Aniketh-74/credit-risk-ml-pipeline
cd credit-risk-ml-pipeline
cp .env.example .env          # fill in passwords and keys

# 2. Start the full stack
docker compose up -d

# 3. Train the initial champion model
python src/training/train.py data/cs-training.csv

# 4. Verify drift detection works
pytest tests/drift/ -v

# 5. Open the dashboard
open http://localhost:8501
```

---

## What This Project Demonstrates

### The Problem: Performative Drift

Standard ML drift detection asks: *did the input distribution change?*

This project addresses a harder question: *did the model's own decisions cause the distribution to change?*

In credit risk, denied applicants don't disappear — they return with nudged feature values. After 30 days, the model that was trained on one population is being evaluated on a population it helped create. PSI sees distribution shift but cannot tell you *why* it shifted. CB-PDD can.

### The Solution: CB-PDD Algorithm

Ported faithfully from [arXiv 2412.10545](https://arxiv.org/abs/2412.10545), the CheckerBoard Performative Drift Detector works by:

1. Routing 10% of predictions through a checkerboard assignment — alternating approve/deny for similar applicants across τ-instance trial periods
2. Measuring the density change rate `a = correction_rate(last_w) − correction_rate(first_w)` within each trial
3. Running a Mann-Whitney U test on Group A vs Group B density distributions
4. Firing only after **2 consecutive windows** with p < α — eliminating single-trial false positives

**Empirical result:** On 30-day denial-loop simulation with τ=1000, CB-PDD first detects at Day 14 with 0 false positives on random data (null control experiment in `tests/drift/test_cb_pdd.py`).

### The Pipeline

| Phase | What was built | Key file |
|-------|---------------|---------|
| 1 | PostgreSQL schema + Docker Compose | `docker-compose.yml`, `db/models.py` |
| 2 | LightGBM + SMOTE, MLflow Registry | `src/training/train.py` |
| 3 | FastAPI scoring + split-path router | `src/api/main.py` |
| 4 | CB-PDD port + drift scorer | `src/drift/cb_pdd.py`, `src/drift/scorer.py` |
| 5 | Airflow daily detect-retrain DAG | `dags/credit_risk_daily.py` |
| 6 | Streamlit monitoring dashboard | `src/dashboard/app.py` |
| 7 | CI/CD + GCP Cloud Run deployment | `.github/workflows/` |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | LightGBM 4.x, SMOTE (imbalanced-learn), AUC > 0.85 |
| API | FastAPI 0.135, asyncpg, Pydantic v2 |
| Drift detection | CB-PDD (arXiv 2412.10545), PSI baseline, scipy Mann-Whitney U |
| Orchestration | Apache Airflow 3.x, LocalExecutor, TaskFlow API |
| Registry | MLflow 3.x, `@champion` alias |
| Database | PostgreSQL 16, SQLAlchemy 2.0, Alembic migrations |
| Dashboard | Streamlit, Plotly |
| Infrastructure | Docker Compose, GCP Cloud Run, GitHub Actions |

---

## Repository Layout

```
credit-risk-ml-pipeline/
├── src/
│   ├── api/           # FastAPI service with split-path router
│   ├── training/      # LightGBM pipeline, SMOTE, MLflow logging
│   ├── drift/         # CB-PDD algorithm + drift scorer
│   ├── simulators/    # Denial loop + score gaming simulators
│   └── dashboard/     # Streamlit monitoring app
├── dags/
│   └── credit_risk_daily.py   # Airflow DAG
├── db/
│   └── models.py      # SQLAlchemy 2.0 ORM models
├── tests/             # 80 tests, 0 failures
├── docker/            # Dockerfiles per service
├── docs/
│   ├── ADR-001.md     # Why CB-PDD over standard detectors
│   └── architecture.md
└── docker-compose.yml
```

---

## What I Learned

**Performative drift is qualitatively different from covariate shift.** A model that denies a loan application at 0.72 risk score is not passively observing the world — it is actively shaping the next observation. Applicants respond. The training distribution shifts not because the world changed, but because the model's predictions changed the world. Standard drift detectors treat this as ordinary distribution shift and either miss it or respond to the symptom rather than the cause.

**Algorithm implementation is a different skill from algorithm use.** Porting CB-PDD from the paper required understanding why the two-consecutive-window gate exists (single-trial false positive rate is exactly α by construction), why the density change rate needs both a first_w and last_w window within the same trial (not across trials), and why Group B uses sign-flipped values (applicants predicted positive in one checkerboard cell vs negative in another must be compared symmetrically). None of this is in a library README.

**τ calibration matters more than algorithm selection.** With τ=500, the detector fires on Day 6 — fast enough to alert, but early enough that the p-value is driven by 3 trials of 500 samples each. With τ=2000 it fires Day 28 — too slow for a 30-day simulation window. τ=1000 is the Goldilocks setting for n_per_day=1000, and the sensitivity chart in the dashboard makes this trade-off visible to anyone who opens it.

---

## Production Limitations

- **Label delay**: CB-PDD requires `outcome_received_at` to be non-null. Real loan defaults arrive 30–180 days late. At long delays, increase `DRIFT_WINDOW_DAYS` or accept slower detection.
- **τ at low volume**: τ=1000 was calibrated at 1000 predictions/day. At 200/day, the first trial period takes 5 days; first detection shifts to Day 70. Recalibrate τ proportionally.
- **Checkerboard routing cost**: 10% of lending decisions are made sub-optimally by design. This is the measurement tax. A live deployment would negotiate this rate with the business team.
- **Single-node Airflow**: LocalExecutor is sufficient for this simulation scale. Production would use CeleryExecutor with a dedicated worker pool.

---

## References

- CB-PDD algorithm: [arXiv 2412.10545](https://arxiv.org/abs/2412.10545) — "Detecting Performative Drift in Machine Learning Systems"
- Dataset: [Give Me Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — 150K loan applicants, 7% default rate
- MLflow 3.x alias API: [MLflow docs](https://mlflow.org/docs/latest/model-registry.html#registering-a-model)
