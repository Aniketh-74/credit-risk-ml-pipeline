# Stack Research

**Domain:** MLOps pipeline — performative drift detection for credit risk scoring
**Researched:** 2026-03-24
**Confidence:** MEDIUM-HIGH (versions verified via PyPI/official sources; performative drift specifics LOW because no off-the-shelf library exists)

---

## Critical Finding: Performative Drift Requires Custom Implementation

**No general-purpose drift library (Evidently, Alibi-Detect, River, NannyML, Frouros) supports performative drift detection.** All standard tools detect covariate shift, concept drift, or data quality degradation — they measure distributional changes in features or labels. Performative drift is a causal mechanism (predictions cause future distribution shifts), not a distributional signal. The distinction matters:

- Standard drift detectors ask: "has the data distribution changed?"
- Performative drift detectors ask: "are the model's own predictions causally responsible for the distribution shift?"

The only published algorithm for this is **CB-PDD** from arxiv 2412.10545 (accepted at AAAI 2025). The authors released Python code at https://github.com/BrandonGower-Winter/CheckerboardDetection/releases/tag/v1.0. This code is experimental/research-grade and is the starting point for the custom production module — not a drop-in library.

**What this means for the stack:** Evidently AI is still used for supplementary standard drift monitoring (covariate shift, PSI, feature distributions). The CB-PDD algorithm is ported from the research repo into a production `drift/` module. These are two separate concerns.

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11.x | Runtime | 3.11 provides significant performance gains over 3.10 and is the most stable version for the scientific Python ecosystem (NumPy, scikit-learn, LightGBM all tested against it); 3.12 has minor compatibility rough edges with some MLflow internals as of early 2026 |
| LightGBM | 4.6.0 | Credit risk model | Consistently outperforms XGBoost on tabular credit data in 2025 benchmarks (lower latency, fewer cold-start issues at inference); native support for imbalanced datasets via `is_unbalance`; the Give Me Credit dataset has heavy class imbalance (~6.7% default rate) |
| FastAPI | 0.135.1 | Scoring API + prediction logging | ASGI native, Pydantic v2 first-class, async I/O without thread pool overhead; the correct choice for any ML serving API in Python that also writes to PostgreSQL |
| Pydantic | 2.12.5 | Request/response validation | v2 is now the only supported version; ~10x faster validation than v1 via Rust core; required by FastAPI 0.119+ |
| MLflow | 3.10.1 | Experiment tracking + model registry | MLflow 3.x (released 2026) is the current stable branch; model stages deprecated in 2.9 — use aliases (`@production`, `@staging`) instead; provides the model registry needed to auto-promote retrained models |
| Apache Airflow | 3.1.8 | Pipeline orchestration | Airflow 3.x is now the stable release (April 2025 GA); Airflow 2.x reaches EOL April 2026 — building new projects on 2.x is inadvisable; DAG versioning and event-driven scheduling in 3.x are exactly what a daily drift-detect-retrain loop needs |
| PostgreSQL | 16.x | Prediction log store | Mature, widely known to hiring managers, native JSON column support for flexible prediction metadata; Row-level security for future multi-tenant extension; preferred over SQLite (no concurrent writes) or MongoDB (unnecessary complexity) |
| Streamlit | 1.44.x (2026 branch) | Monitoring dashboard | Used by 90%+ of Fortune 50 for internal data apps; zero-JS Python-native; `st.tabs`, `st.metric`, `st.line_chart` sufficient for drift score evolution + model version history — no React needed |
| Docker / docker-compose | 27.x / 2.x | Local reproducibility | Canonical containerization; docker-compose v2 (now `docker compose`) is the standard for local multi-service setups |
| GCP Cloud Run | N/A (managed) | Production hosting | Serverless container hosting; zero infrastructure management; scales to zero (free when idle — correct for a portfolio demo); native GitHub Actions integration via Cloud Build |
| GitHub Actions | N/A | CI/CD | Standard for portfolio; free tier sufficient; proven GCP Cloud Run deployment workflows documented officially |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-learn | 1.6.x | Preprocessing, SMOTE, evaluation metrics | Feature encoding, train/test split, ROC AUC, confusion matrix; wraps around LightGBM via the sklearn API |
| imbalanced-learn | 0.12.x | SMOTEENN / SMOTE for class imbalance | Give Me Credit has ~6.7% default rate; apply SMOTE on training set only, never on test set |
| SQLAlchemy | 2.0.x | ORM for PostgreSQL | Async-compatible via `asyncpg`; use SQLAlchemy 2.0 style (not legacy 1.4 patterns); required for FastAPI prediction logging |
| asyncpg | 0.30.x | Async PostgreSQL driver | Fastest async Postgres driver in benchmarks; pair with SQLAlchemy 2.0 async engine |
| Alembic | 1.14.x | Database schema migrations | SQLAlchemy's migration companion; version-control the prediction_logs schema |
| Evidently | 0.7.17 | Supplementary standard drift monitoring | Use for covariate/data quality drift reporting in Streamlit dashboard alongside the custom CB-PDD score; do NOT use as the performative drift detector |
| scipy | 1.15.x | Statistical testing inside CB-PDD | Mann-Whitney U test used directly by the CB-PDD algorithm; already a transitive dep of scikit-learn |
| pandas | 2.2.x | Data manipulation | Required for Give Me Credit loading and feature engineering; use 2.x (Arrow-backed dtypes reduce memory) |
| numpy | 1.26.x | Numerical operations | Pinned to 1.26.x for compatibility; numpy 2.x has breaking C API changes that affect some LightGBM builds |
| pytest | 8.x | Testing | Standard; use with `pytest-asyncio` for FastAPI async endpoint tests |
| httpx | 0.27.x | Async HTTP client for testing | FastAPI's `TestClient` uses this internally; also used to call the scoring API from the Airflow DAG |
| gunicorn | 23.x | WSGI process manager for Cloud Run | Manages Uvicorn workers in production; formula: `(2 * CPU) + 1` workers |
| uvicorn | 0.34.x | ASGI server | Runs FastAPI; use `uvicorn[standard]` to include `uvloop` and `httptools` |
| python-dotenv | 1.0.x | Environment variable management | Load `.env` in local dev; Cloud Run uses env vars natively |
| google-cloud-storage | 2.x | GCS artifact storage for MLflow | MLflow artifact store backend on GCP; store model binaries in GCS, not in the container |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| uv | Fast Python package/env management | Faster than pip+venv; use for local dev; Docker still uses pip for reproducibility |
| ruff | Linting + formatting | Replaces flake8 + black + isort in one tool; run in GitHub Actions pre-commit |
| mypy | Static type checking | Catches Pydantic model mismatches early; configure `strict = false` for initial phases |
| pre-commit | Git hooks | Wire ruff + mypy; prevents committing untested code |
| docker-compose | Local multi-service stack | Airflow, FastAPI, PostgreSQL, MLflow all in one `compose up` |

---

## Performative Drift Implementation Strategy

This is the architecturally critical decision. Three options exist:

**Option A — Use CB-PDD research code directly (NOT recommended)**
The research repo at `CheckerboardDetection/releases/v1.0` is experiment code, not a production module. It likely has hardcoded paths, no tests, and no API surface.

**Option B — Port CB-PDD to a production module (RECOMMENDED)**
Implement the CB-PDD algorithm from scratch in a `drift/` package using the paper's pseudocode + the research repo as a reference implementation. The algorithm requires: a checkerboard classifier, sliding window density tracking, and a Mann-Whitney U test (scipy). This is ~200-400 lines of Python. This approach is the differentiator — it signals you read the paper, understood it, and engineered a production version.

**Option C — Wrap Evidently around a proxy metric for performative drift (NOT recommended)**
Some teams approximate performative drift by watching for denial-rate feedback (e.g., denied applicants disappearing from future cohorts). This is not CB-PDD and would misrepresent fidelity to arxiv 2412.10545.

**Decision: Option B.** Port CB-PDD into `src/drift/cb_pdd.py`. Use Evidently for supplementary data quality and covariate shift dashboards only.

---

## Installation

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Core runtime
pip install \
  fastapi[standard]==0.135.1 \
  pydantic==2.12.5 \
  uvicorn[standard]==0.34.0 \
  gunicorn==23.0.0 \
  mlflow==3.10.1 \
  apache-airflow==3.1.8 \
  lightgbm==4.6.0 \
  scikit-learn==1.6.1 \
  imbalanced-learn==0.12.4 \
  pandas==2.2.3 \
  numpy==1.26.4 \
  scipy==1.15.2 \
  sqlalchemy==2.0.39 \
  asyncpg==0.30.0 \
  alembic==1.14.1 \
  evidently==0.7.17 \
  streamlit==1.44.0 \
  httpx==0.27.2 \
  python-dotenv==1.0.1 \
  google-cloud-storage==2.19.0

# Dev dependencies
pip install \
  pytest==8.3.5 \
  pytest-asyncio==0.25.3 \
  ruff==0.9.9 \
  mypy==1.14.1 \
  pre-commit==4.1.0
```

**Note on Airflow:** Airflow has its own complex dependency tree. Install in a separate venv or use the official Docker image (`apache/airflow:3.1.8`). Do not install Airflow alongside FastAPI in the same container.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Airflow 3.1.8 | Airflow 2.11.x | Only if your organization is already on 2.x and can't migrate; new greenfield projects should start on 3.x since 2.x hits EOL April 2026 |
| Airflow 3.1.8 | Prefect 3.x | If you're a solo/small team that wants faster iteration; Prefect has less setup overhead but less name recognition with enterprise hiring managers |
| Airflow 3.1.8 | Dagster 1.x | If asset-centric observability is a priority; Dagster has better DX but Airflow is the industry standard signal for MLOps roles |
| LightGBM 4.6.0 | XGBoost 2.x | XGBoost performs comparably but has higher inference latency; LightGBM wins on the Give Me Credit dataset in most benchmarks |
| LightGBM 4.6.0 | CatBoost | CatBoost wins on high-cardinality categorical data; Give Me Credit is mostly numerical |
| Custom CB-PDD module | Evidently (alone) | Never — Evidently does not detect performative drift; only use Evidently as a supplement for standard drift metrics |
| Custom CB-PDD module | Alibi-Detect | Alibi-Detect has excellent kernel-based tests (MMD, LSDD) for covariate shift but no causal/feedback-loop mechanism |
| Custom CB-PDD module | River | River is an online learning library; its ADWIN/KSWIN detectors are concept drift detectors, not performative drift detectors |
| PostgreSQL 16 | SQLite | SQLite lacks concurrent write support; when the FastAPI service and Airflow DAG both write predictions, you need a proper RDBMS |
| PostgreSQL 16 | Redis | Redis is for cache/queue; not suitable as the primary prediction log store |
| asyncpg | psycopg2 | psycopg2 is synchronous; acceptable for Airflow DAGs (which are synchronous) but wrong for FastAPI async endpoints |
| MLflow 3.x | Weights & Biases | W&B is excellent but closed-source SaaS; MLflow self-hosted is more aligned with "production infrastructure" portfolio signal |
| GCP Cloud Run | AWS Lambda | Cloud Run handles long-running containers better; Lambda cold starts are severe for ML model loading |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Airflow 2.x for new projects | Hits EOL April 2026; XCom pickling removed in 3.x creates migration pain if you start on 2.x | Airflow 3.1.8 |
| MLflow Model Stages (Staging/Production/Archived) | Deprecated since MLflow 2.9.0, will be removed in future major | MLflow model aliases: `@production`, `@challenger` |
| numpy 2.x | Breaking C API changes affect LightGBM GPU builds and some scikit-learn extensions; not worth the risk for this project | numpy 1.26.x (pin explicitly) |
| psycopg2 in FastAPI async handlers | Blocks the event loop; causes latency spikes under load | asyncpg with SQLAlchemy 2.0 async engine |
| River (online ML library) | Its drift detectors (ADWIN, KSWIN, PageHinkley) detect concept drift in streaming data — fundamentally different mechanism from CB-PDD | Custom CB-PDD module |
| Evidently as the performative drift detector | Evidently detects covariate/label shift; it has no mechanism for causal feedback loops | Custom CB-PDD module (use Evidently only for supplementary standard drift metrics) |
| Alibi-Detect as the performative drift detector | MMD and LSDD tests detect distribution mismatch; they cannot distinguish whether the model caused the shift | Custom CB-PDD module |
| Flask instead of FastAPI | No async support, no automatic OpenAPI docs, no Pydantic integration; wrong choice for ML serving in 2026 | FastAPI |
| Jupyter notebooks in production | Non-reproducible, not testable, not deployable; use for EDA only, then move logic to `.py` modules | Python modules with pytest |
| Pickle for model serialization | Security vulnerabilities, not portable across Python versions | MLflow's `log_model()` with LightGBM flavor (saves in native LightGBM binary format) |

---

## Stack Patterns by Variant

**For the Airflow DAG (daily pipeline):**
- Use the synchronous psycopg2/SQLAlchemy pattern (not async) — Airflow operators are synchronous
- Use `apache-airflow-providers-standard` for PythonOperator (now a separate package in Airflow 3.x)
- Use `apache-airflow-providers-http` to call the FastAPI scoring endpoint from a DAG task

**For the FastAPI scoring service:**
- Use async SQLAlchemy + asyncpg for all database writes
- Use `BackgroundTasks` for non-blocking prediction logging (don't block the response)
- Load LightGBM model from MLflow registry at startup, not per-request

**For Cloud Run deployment:**
- One container per service: FastAPI (scoring), Streamlit (dashboard)
- Airflow runs in docker-compose locally; for Cloud Run, use Cloud Composer if needed (but local Airflow is sufficient for a portfolio demo)
- MLflow tracking server: deploy as a separate Cloud Run service with GCS artifact store + Cloud SQL (PostgreSQL) backend

**For the CB-PDD module:**
- Keep it as a standalone Python package in `src/drift/cb_pdd.py`
- Accept a `pd.DataFrame` of prediction-outcome pairs as input
- Return a `DriftResult` dataclass with: `is_drift: bool`, `p_value: float`, `score: float`, `trial_window: int`
- Write unit tests against synthetic feedback loop data before integrating into the Airflow DAG

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| fastapi==0.135.1 | pydantic>=2.7.0 | Pydantic v1 is deprecated as of FastAPI 0.119+; do not mix |
| lightgbm==4.6.0 | numpy==1.26.x | numpy 2.x causes build issues with LightGBM on some platforms; pin to 1.26.x |
| mlflow==3.10.1 | Python 3.11, sqlalchemy>=2.0 | MLflow 3.x requires Python 3.9+; use 3.11 for best performance |
| apache-airflow==3.1.8 | Python 3.11, sqlalchemy>=2.0 | Airflow 3.x requires Python 3.9+; PythonOperator now in `apache-airflow-providers-standard` |
| sqlalchemy==2.0.x | asyncpg==0.30.x | Use `create_async_engine("postgresql+asyncpg://...")` syntax |
| evidently==0.7.17 | pandas==2.2.x, scikit-learn>=1.0 | Evidently 0.7.x API changed from pre-0.6; use `Report` and `Test Suite` classes |
| scikit-learn==1.6.x | imbalanced-learn==0.12.x | imblearn tracks sklearn releases; 0.12.x is compatible with sklearn 1.6.x |

---

## Sources

- arxiv 2412.10545 (HTML) — CB-PDD algorithm structure, components, parameter list — MEDIUM confidence
- https://github.com/BrandonGower-Winter/CheckerboardDetection/releases/tag/v1.0 — Reference Python implementation exists, Python 98.8% of codebase — MEDIUM confidence
- https://ojs.aaai.org/index.php/AAAI/article/view/33276 — Paper accepted at AAAI 2025, confirming algorithm validity — HIGH confidence
- https://mlflow.org/releases — MLflow 3.10.1 confirmed as latest (March 2026) — HIGH confidence
- https://pypi.org/project/mlflow/ — Version confirmation — HIGH confidence
- https://airflow.apache.org/docs/apache-airflow/stable/release_notes.html — Airflow 3.1.8 confirmed latest stable — HIGH confidence
- https://pypi.org/project/fastapi/ — FastAPI 0.135.1 confirmed latest — HIGH confidence
- https://docs.pydantic.dev/latest/ — Pydantic 2.12.5 confirmed latest — HIGH confidence
- https://lightgbm.readthedocs.io/en/latest/ — LightGBM 4.6.0 confirmed latest stable — HIGH confidence
- https://arxiv.org/pdf/2404.18673 — Open-Source Drift Detection Tools comparison (Evidently vs Alibi-Detect) — MEDIUM confidence
- https://www.fuzzylabs.ai/blog-post/evidently-vs-alibi-detect-comparing-model-monitoring-tools — Confirmed Evidently more performant for production use — MEDIUM confidence
- WebSearch: Airflow 3.x vs 2.x migration — breaking changes confirmed with official Airflow docs — HIGH confidence
- WebSearch: LightGBM vs XGBoost on credit data (2025 research) — LightGBM recommended — MEDIUM confidence

---

## Open Questions (Needs Phase-Specific Research)

1. **CB-PDD parameter tuning (τ, f, w, α):** The paper specifies these but optimal values for Give Me Credit's denial-loop simulation require experimentation. Flag Phase 2 for this.

2. **MLflow 3.x artifact store on GCP:** Verify GCS bucket permissions pattern for Cloud Run → GCS artifact writes. The OSS MLflow docs show AWS S3 examples more prominently; GCS is supported but setup is less documented.

3. **Airflow 3.x `apache-airflow-providers-standard` packaging:** PythonOperator moved to a separate package. Verify the correct `pip install apache-airflow-providers-standard` version that ships with Airflow 3.1.8 before writing DAGs.

4. **Streamlit + Airflow network topology in docker-compose:** Streamlit needs to query PostgreSQL for drift scores and MLflow for model history. Verify the service network layout before Phase 5.

---
*Stack research for: credit-risk-ml-pipeline (performative drift detection MLOps)*
*Researched: 2026-03-24*
