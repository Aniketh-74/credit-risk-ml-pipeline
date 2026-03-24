# Architecture Research

**Domain:** MLOps pipeline — performative drift detection for credit risk scoring
**Researched:** 2026-03-24
**Confidence:** HIGH (component integration patterns verified via official docs and multiple sources)

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION LAYER                             │
│                                                                         │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                    Airflow DAG (daily schedule)                  │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │  │
│   │  │  batch   │  │  drift   │  │ branch:  │  │  promote if    │  │  │
│   │  │  score   │→ │  check   │→ │  retrain │→ │  AUC improves  │  │  │
│   │  │  task    │  │  task    │  │  or skip │  │  (alias swap)  │  │  │
│   │  └──────────┘  └──────────┘  └──────────┘  └────────────────┘  │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                │                    │                    │
                ▼                    ▼                    ▼
┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│   SERVING LAYER      │  │  DRIFT DETECTION │  │  EXPERIMENT TRACKING │
│                      │  │  MODULE          │  │                      │
│  ┌────────────────┐  │  │                  │  │  ┌────────────────┐  │
│  │  FastAPI       │  │  │  drift_detector/ │  │  │  MLflow        │  │
│  │  /score        │  │  │  cb_pdd.py       │  │  │  Tracking      │  │
│  │  /outcome      │  │  │                  │  │  │  Server        │  │
│  │  /health       │  │  │  (standalone     │  │  │                │  │
│  └───────┬────────┘  │  │   Python module, │  │  │  Model         │  │
│          │           │  │   called by      │  │  │  Registry      │  │
│          │ log       │  │   Airflow task)  │  │  │  (aliases)     │  │
│          ▼           │  └────────┬─────────┘  │  └────────┬───────┘  │
│  ┌────────────────┐  │           │             │           │          │
│  │  BackgroundTask│  │           │ reads from  │           │ champion │
│  │  (async write) │  │           │ PostgreSQL  │           │  alias   │
│  └────────┬───────┘  │           │             │           │          │
└───────────┼──────────┘  └─────────┼─────────────┘  └─────────┼────────┘
            │                       │                           │
            ▼                       ▼                           │
┌─────────────────────────────────────────────────────┐         │
│                  PERSISTENCE LAYER                  │         │
│                                                     │         │
│  ┌─────────────────────────────────────────────┐   │         │
│  │  PostgreSQL                                 │   │         │
│  │                                             │   │         │
│  │  predictions (id, features, score,          │   │         │
│  │               decision, ts, model_version)  │   │         │
│  │                                             │   │         │
│  │  outcomes (prediction_id, actual_default,   │   │         │
│  │            observed_at)                     │   │         │
│  │                                             │   │         │
│  │  drift_scores (computed_at, drift_score,    │   │         │
│  │                threshold_crossed, window)   │   │         │
│  │                                             │   │         │
│  │  alerts (fired_at, drift_score,             │   │         │
│  │          retrain_run_id, promoted)           │   │         │
│  └─────────────────────────────────────────────┘   │         │
└─────────────────────────────────────────────────────┘         │
                                                                 │
┌────────────────────────────────────────────────────────────────┘
│              VISUALISATION LAYER
│
│  ┌─────────────────────────────────────────────────────────┐
│  │  Streamlit Dashboard                                    │
│  │                                                         │
│  │  • Drift score timeline (30-day window)                 │
│  │  • Model version history (reads MLflow REST API)        │
│  │  • Alert log (reads drift_scores + alerts tables)       │
│  └─────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| FastAPI service | Score loan applications; log predictions + simulated outcomes to PostgreSQL | Python, uvicorn, SQLAlchemy async |
| PostgreSQL | Authoritative store for all predictions, outcomes, drift scores, alerts | Docker container, tables per concern |
| CB-PDD detector | Implement arxiv 2412.10545 algorithm; compute drift score over a rolling window of predictions | Standalone Python module, no framework dependency |
| Airflow DAG | Orchestrate daily: batch score → drift check → conditional retrain → conditional promote | DAG with BranchPythonOperator for the retrain decision |
| MLflow Tracking | Log all training runs, metrics, artifacts | mlflow.start_run(), auto-logging |
| MLflow Registry | Version models; expose `champion` alias that FastAPI loads at startup | client.set_registered_model_alias() |
| Streamlit dashboard | Visualise drift score history, model versions, alert log | Reads PostgreSQL + MLflow REST API directly |
| Feedback loop simulator | Replay predictions as if they affect next day's applicant pool; two modes: denial loop + score gaming | Python script seeded by previous day's prediction table, called before batch scoring |

## Recommended Project Structure

```
credit-risk-ml-pipeline/
├── src/
│   ├── api/                        # FastAPI application
│   │   ├── main.py                 # App factory, router registration
│   │   ├── routes/
│   │   │   ├── score.py            # POST /score — returns probability + decision
│   │   │   └── outcome.py          # POST /outcome — records ground truth
│   │   ├── models.py               # Pydantic request/response schemas
│   │   └── db.py                   # SQLAlchemy async session, prediction logging
│   │
│   ├── drift/                      # Performative drift detection (no external deps)
│   │   ├── cb_pdd.py               # CB-PDD algorithm faithful to arxiv 2412.10545
│   │   ├── windows.py              # Sliding window helpers over prediction history
│   │   └── scorer.py               # Wraps CB-PDD; reads PostgreSQL; returns drift score
│   │
│   ├── training/                   # Model training + MLflow integration
│   │   ├── train.py                # XGBoost/GBM training on Give Me Credit
│   │   ├── evaluate.py             # AUC, precision, recall calculation
│   │   └── promote.py              # set_registered_model_alias("champion") logic
│   │
│   ├── simulation/                 # Feedback loop simulation
│   │   ├── denial_loop.py          # Denial feedback: denied applicants never default
│   │   └── score_gaming.py         # Gaming feedback: applicants shift features toward approval
│   │
│   └── dashboard/
│       └── app.py                  # Streamlit app
│
├── dags/
│   └── credit_risk_pipeline.py     # Airflow DAG definition
│
├── db/
│   └── migrations/                 # Alembic migrations (predictions, outcomes, drift_scores, alerts)
│
├── notebooks/
│   └── eda.ipynb                   # Exploratory analysis on Give Me Credit
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.airflow
│   └── Dockerfile.dashboard
│
├── docker-compose.yml              # Full local stack
├── .github/workflows/
│   ├── ci.yml                      # Lint, test, build on PR
│   └── deploy.yml                  # Push to Cloud Run on main merge
└── README.md
```

### Structure Rationale

- **src/drift/**: Isolated from the API and Airflow, making the CB-PDD implementation independently testable — important for paper-fidelity claims in interviews.
- **src/simulation/**: Separated from training and drift detection because it is the "fake data generator" — it never runs in a real deployment, only in the demo loop.
- **dags/**: Standard Airflow convention; keeping DAG definitions outside src prevents import confusion when Airflow scans for DAGs.
- **src/training/promote.py**: Isolated promotion logic prevents the Airflow task from knowing too much about MLflow internals — the task just calls `promote.maybe_promote(run_id)`.

## Architectural Patterns

### Pattern 1: Drift Detector as Airflow Task (not embedded in API)

**What:** The CB-PDD detector runs as a dedicated Airflow task that reads the `predictions` table, computes the drift score, writes to `drift_scores`, and returns a boolean for the branch operator.

**When to use:** This is the correct placement because: (a) drift detection over a rolling window of thousands of predictions is a batch operation, not a per-request concern; (b) embedding it in FastAPI would add latency and couple two unrelated concerns; (c) as a separate service it would add unnecessary network overhead and operational complexity for a demo project.

**Trade-offs:** The drift check is only as fresh as the DAG schedule (daily). For this project that is acceptable — the feedback loop simulation operates on daily windows anyway.

```python
# dags/credit_risk_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta

def check_drift(**context):
    from src.drift.scorer import compute_drift_score
    score = compute_drift_score(window_days=7)
    context['ti'].xcom_push(key='drift_score', value=score)
    return 'trigger_retrain' if score > DRIFT_THRESHOLD else 'skip_retrain'

with DAG('credit_risk_pipeline', schedule_interval='@daily', ...) as dag:
    run_batch_scoring = PythonOperator(task_id='batch_score', ...)
    drift_branch = BranchPythonOperator(task_id='drift_check', python_callable=check_drift)
    trigger_retrain = PythonOperator(task_id='trigger_retrain', ...)
    skip_retrain = PythonOperator(task_id='skip_retrain', python_callable=lambda: None)
    promote_model = PythonOperator(task_id='promote_model', trigger_rule='none_failed')

    run_batch_scoring >> drift_branch >> [trigger_retrain, skip_retrain] >> promote_model
```

### Pattern 2: MLflow Alias-Based Model Promotion

**What:** After retraining, the new run's AUC is compared to the current `champion` model's AUC. If the new model wins, `set_registered_model_alias("credit-risk-model", "champion", new_version)` is called. FastAPI loads `models:/credit-risk-model@champion` at startup (or polls for updates).

**When to use:** Always in MLflow 3.x — stages are deprecated as of MLflow 2.9.0. Aliases are the production pattern.

**Trade-offs:** Model is only reloaded at API restart unless an explicit reload endpoint is added. For demo purposes, a startup load is sufficient.

```python
# src/training/promote.py
import mlflow
from mlflow.tracking import MlflowClient

def maybe_promote(new_run_id: str, model_name: str = "credit-risk-model") -> bool:
    client = MlflowClient()
    new_run = client.get_run(new_run_id)
    new_auc = new_run.data.metrics["auc"]

    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
        champion_run = client.get_run(champion.run_id)
        champion_auc = champion_run.data.metrics["auc"]
    except Exception:
        champion_auc = 0.0  # No champion yet

    if new_auc > champion_auc:
        new_version = client.get_latest_versions(model_name)[0].version
        client.set_registered_model_alias(model_name, "champion", new_version)
        return True
    return False
```

### Pattern 3: Feedback Loop Simulation via Prediction Table Replay

**What:** The feedback loop simulation reads yesterday's prediction decisions from PostgreSQL and constructs tomorrow's synthetic applicant pool. Two mechanisms:

1. **Denial loop:** Applicants who were denied loans yesterday are removed from tomorrow's pool (they went elsewhere or gave up), so the model never sees them default — reinforcing the model's belief that this demographic is high-risk.
2. **Score gaming loop:** Applicants who were near the approval threshold adjust features (e.g., lower reported debt ratio) to cross the boundary — shifting the feature distribution.

The simulator produces a CSV/dataframe that becomes the next day's batch scoring input, passed to the FastAPI `/score` endpoint or directly to the batch scoring Airflow task.

**When to use:** This is the core novelty of the project. The simulator runs as the first step in the daily DAG, before batch scoring. It does NOT need to be stateless — it accumulates drift deliberately over 30 simulated days.

**Trade-offs:** The simulation is fake by design. Ground truth labels are also simulated (loans denied by the model are assigned "no default" because we never observed them). This is precisely what CB-PDD is designed to detect.

```python
# src/simulation/denial_loop.py
def generate_next_day_pool(db_session, simulation_day: int) -> pd.DataFrame:
    """
    Reads yesterday's predictions. Removes denied applicants.
    Returns synthetic applicant pool for today's batch scoring.
    """
    yesterday_predictions = db_session.query(Prediction).filter(
        Prediction.simulation_day == simulation_day - 1
    ).all()

    approved = [p for p in yesterday_predictions if p.decision == "approved"]
    # Denied applicants exit the pool — this is the feedback mechanism
    new_pool = build_pool_from_approved(approved)
    return new_pool
```

## Data Flow

### Prediction Logging Flow (online path)

```
POST /score (loan application features)
    |
    v
FastAPI loads model: mlflow.pyfunc.load_model("models:/credit-risk-model@champion")
    |
    v
Model returns probability score + decision (approve/deny)
    |
    v
BackgroundTask: async write to PostgreSQL predictions table
    |  (includes: features, score, decision, model_version, simulation_day)
    v
Response returned to caller
```

### Daily Orchestration Flow (batch path)

```
Airflow DAG fires (daily or per simulation day)
    |
    v
[Task 1] Feedback loop simulator
    reads: predictions WHERE simulation_day = N-1
    writes: synthetic applicant pool for day N (to temp table or CSV artifact)
    |
    v
[Task 2] Batch score day N applicant pool
    calls: FastAPI /score endpoint in batch OR directly invokes model
    writes: predictions table (simulation_day = N)
    |
    v
[Task 3] CB-PDD drift check
    reads: predictions table (rolling 7-day window)
    computes: Mann-Whitney U test on density groups (per arxiv 2412.10545)
    writes: drift_scores table (drift_score, threshold_crossed flag)
    returns: 'trigger_retrain' or 'skip_retrain' to BranchPythonOperator
    |
    v (branch)
[Task 4a] trigger_retrain (if drift_score > threshold)
    runs: mlflow.start_run() → XGBoost training on recent labeled data
    logs: AUC, feature importance, model artifact
    writes: alerts table (drift_score, retrain_run_id)
    |
    v (if threshold crossed only)
[Task 4b] promote_model
    compares: new AUC vs champion AUC via MLflow client
    if improved: client.set_registered_model_alias("credit-risk-model", "champion", new_version)
    writes: alerts table (promoted=True/False)
    |
    v (both branches join here, trigger_rule='none_failed')
[Task 5] Update dashboard data cache
    (optional: materialize a summary view in PostgreSQL for Streamlit performance)
```

### Dashboard Read Flow

```
Streamlit app (polling or on-demand)
    |
    +--> PostgreSQL: SELECT * FROM drift_scores ORDER BY computed_at
    |         → renders: drift score timeline chart
    |
    +--> MLflow REST API: GET /api/2.0/mlflow/registered-models/get-latest-versions
    |         → renders: model version history table
    |
    +--> PostgreSQL: SELECT * FROM alerts ORDER BY fired_at
              → renders: alert log with retrain outcomes
```

### Key Data Flows

1. **Prediction → Drift signal:** `predictions` table is the source of truth for CB-PDD. Every scored application feeds the detector's window. The simulation makes this table grow predictably with known feedback patterns.
2. **Drift score → Retrain trigger:** The drift check task writes to `drift_scores` and returns a task ID to the branch operator. The retraining decision is deterministic and logged — no side effects outside DAG state + PostgreSQL.
3. **Retrain → Promotion:** MLflow run ID flows via Airflow XCom from the retrain task to the promotion task. The promotion task reads AUC from MLflow, not from the database.
4. **Promotion → Serving:** FastAPI reads the `@champion` alias from MLflow Registry. A model reload is triggered on the next request after promotion (or by restarting the container — acceptable for demo scope).

## Suggested Build Order

Dependencies drive this order. Each layer depends on the previous.

| Order | Component | Depends On | Rationale |
|-------|-----------|------------|-----------|
| 1 | PostgreSQL schema + Alembic migrations | Nothing | All other components read/write here |
| 2 | Training script + MLflow logging | PostgreSQL (for Give Me Credit data path) | Must have a trained model before serving |
| 3 | Model promotion logic (promote.py) | MLflow Registry | Needed by Airflow promote task |
| 4 | FastAPI /score endpoint + prediction logging | Trained model in MLflow Registry | Need predictions in DB before drift detection |
| 5 | Feedback loop simulator | predictions table in PostgreSQL | Reads yesterday's predictions |
| 6 | CB-PDD drift detector (cb_pdd.py + scorer.py) | predictions table + paper algorithm | Core novelty; build standalone, unit-test against known inputs |
| 7 | Airflow DAG | All above tasks as importable functions | Orchestrates the full loop end-to-end |
| 8 | Streamlit dashboard | PostgreSQL drift_scores + alerts + MLflow REST API | Reads existing data; no writes |
| 9 | Docker Compose | All services defined | Local integration test |
| 10 | CI/CD + GCP Cloud Run | Docker images + secrets | Final deployment step |

## Where the Performative Drift Detector Lives

**Decision: Airflow task (standalone Python module, called by task function).**

Rationale:

- CB-PDD operates on a window of historical predictions — inherently a batch computation. It is wrong to run it per-request in FastAPI.
- A "separate microservice" approach would add a network boundary, a separate container, health checks, and deployment complexity that adds zero portfolio value and obscures the algorithm.
- As an Airflow task, the drift computation is: scheduled, logged (via XCom), observable (via Airflow UI), and wired to the retrain branch — which is exactly the production MLOps story this project tells.
- The Python module in `src/drift/` has no Airflow imports, so it can be unit-tested independently and described as a clean algorithm implementation during interviews.

The detector reads from PostgreSQL (not from a streaming queue), which matches the batch/daily cadence of the feedback loop simulation.

## Anti-Patterns

### Anti-Pattern 1: Embedding Drift Detection in FastAPI

**What people do:** Add a drift check inside the `/score` endpoint, computing drift after every N requests.

**Why it's wrong:** Drift detection on a prediction window is a batch operation (seconds to run on thousands of rows). Doing it per-request adds latency unpredictably. It also couples the serving concern (low latency response) with the monitoring concern (statistical test over history).

**Do this instead:** Run drift detection as a scheduled Airflow task. FastAPI's only job is to score and log.

### Anti-Pattern 2: Using MLflow Stages Instead of Aliases

**What people do:** `transition_model_version_stage(name, version, stage="Production")` — the old pattern.

**Why it's wrong:** MLflow 2.9.0+ deprecated stages. MLflow 3.x maps stages to aliases internally, but the stage API will be removed in a future major release.

**Do this instead:** Use `client.set_registered_model_alias(name, "champion", version)` and load with `models:/credit-risk-model@champion`.

### Anti-Pattern 3: Simulating Ground Truth Labels from Model Decisions

**What people do:** Using `prediction.score > 0.5` as the outcome label to feed the retraining loop.

**Why it's wrong:** This creates a perfectly self-confirming loop — the model retrains on its own outputs with no signal correction. This is not performative drift detection; it is label leakage.

**Do this instead:** The denial loop simulation assigns `actual_default = False` to denied applicants (they never had the chance to default, so we have no signal). CB-PDD is designed precisely to detect this absence-of-signal pattern. Do not fabricate default labels from scores.

### Anti-Pattern 4: One Fat Airflow Task

**What people do:** A single `run_pipeline` Python operator that does batch scoring + drift check + retrain + promote in sequence.

**Why it's wrong:** You lose Airflow's retry granularity, observability, and branching. If retraining fails, you cannot skip just that task on retry — you restart everything.

**Do this instead:** Each logical step is its own task. Use BranchPythonOperator for the retrain/skip decision. Use `trigger_rule='none_failed'` on the final promote task so it runs whether or not retraining occurred.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| MLflow Tracking Server | Python SDK (`mlflow.start_run()`, `mlflow.log_metrics()`) | Run as Docker container; expose port 5000 |
| MLflow Model Registry | `MlflowClient` API; load via `models:/name@alias` URI | Aliases replace stages in MLflow 3.x |
| PostgreSQL | SQLAlchemy (async for FastAPI, sync for Airflow tasks) | Single DB, separate schemas or tables per concern |
| Streamlit → MLflow | MLflow REST API (`GET /api/2.0/mlflow/...`) | No Python SDK needed in Streamlit; HTTP calls sufficient |
| Airflow → src/ | Python imports; DAG tasks call functions from `src/` | Keep `dags/` thin — business logic lives in `src/` |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| FastAPI API → PostgreSQL | SQLAlchemy async session; BackgroundTask for logging | Prediction write must be non-blocking |
| Airflow task → CB-PDD module | Direct Python function call (import) | No HTTP; drift module has no Airflow dependency |
| Airflow retrain task → MLflow | mlflow SDK inside task function | Airflow task is responsible for starting and ending the run |
| Airflow promote task → MLflow Registry | MlflowClient.set_registered_model_alias() | Receives run_id via XCom from retrain task |
| Streamlit → PostgreSQL | psycopg2 or SQLAlchemy sync | Read-only queries; cache with st.cache_data |
| Streamlit → MLflow | REST API calls (requests library) | Simpler than SDK for read-only dashboard use |

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Demo (30-day simulation, single user) | All services on Docker Compose; SQLite could work but PostgreSQL keeps it production-realistic |
| Portfolio demo live on Cloud Run | FastAPI on Cloud Run (stateless), PostgreSQL on Cloud SQL, MLflow on Cloud Run or persistent VM, Airflow on Cloud Composer (or skip — run locally) |
| Production at a fintech | FastAPI behind load balancer; PostgreSQL with read replicas; Airflow on managed service (MWAA/Cloud Composer); MLflow on dedicated server with S3 artifact store; Streamlit replaced by Grafana/custom BI |

### Scaling Priorities

1. **First bottleneck:** PostgreSQL write throughput under high prediction volume. Fix: async writes via BackgroundTask (already in the design); add connection pooling via pgbouncer.
2. **Second bottleneck:** Drift check runtime on large prediction windows. Fix: materialised PostgreSQL view over the drift window; CB-PDD operates on aggregated density counts, not raw rows.

## Sources

- [MLflow Model Registry (Official Docs)](https://mlflow.org/docs/latest/ml/model-registry/) — alias-based promotion confirmed, stages deprecated as of 2.9.0 [HIGH confidence]
- [MLflow Model Registry Workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow/) — `set_registered_model_alias` API confirmed [HIGH confidence]
- [Apache Airflow MLOps Use Cases](https://airflow.apache.org/use-cases/mlops/) — BranchPythonOperator for conditional retraining confirmed [HIGH confidence]
- [Astronomer Airflow MLOps Best Practices](https://www.astronomer.io/docs/learn/airflow-mlops) — atomic tasks, none_failed trigger rule pattern [MEDIUM confidence]
- [arxiv 2412.10545 — CB-PDD Algorithm](https://arxiv.org/abs/2412.10545) — Mann-Whitney U test on density windows, binary drift output [HIGH confidence — primary paper]
- [arxiv 2412.10545 HTML Full Text](https://arxiv.org/html/2412.10545) — algorithm inputs (f, τ, α), two-window density calculation confirmed [HIGH confidence]
- [Evidently AI + FastAPI Monitoring Pattern](https://www.evidentlyai.com/blog/fastapi-tutorial) — prediction logging to PostgreSQL via BackgroundTask [MEDIUM confidence]
- [MLflow End-to-End CI/CD Workflow](https://markaicode.com/mlflow-complete-workflow/) — champion/challenger alias pattern [MEDIUM confidence]

---
*Architecture research for: credit-risk-ml-pipeline MLOps — performative drift detection*
*Researched: 2026-03-24*
