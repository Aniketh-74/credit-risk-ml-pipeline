# Pitfalls Research

**Domain:** MLOps credit risk pipeline with performative drift detection (arxiv 2412.10545)
**Researched:** 2026-03-24
**Confidence:** MEDIUM-HIGH — paper pitfalls from direct HTML fetch (HIGH); MLOps pitfalls from multiple sources (MEDIUM); portfolio pitfalls from practitioner articles (MEDIUM)

---

## Critical Pitfalls

### Pitfall 1: CB-PDD Requires Deliberately Misclassifying Live Predictions

**What goes wrong:**
The CB-PDD algorithm is an intervention testing method. It works by assigning some predictions via the CheckerBoard logic (deliberately wrong labels) rather than the actual model, then measuring whether density shifts correlate with those interventions. In a simulated portfolio project, you might implement the model scoring loop without the intervention layer and only realize when writing the drift detector that it expects a fraction of predictions to be CheckerBoard-assigned, not model-assigned.

**Why it happens:**
The paper's Algorithm 1 assumes intervention is always possible. The mix parameter (probability an instance goes to actual predictor vs CheckerBoard) is described in experiments but the paper gives no guidance for real-world values. Developers read the paper, implement the model, then discover the drift detector is architecturally separate from the scorer in a non-trivial way.

**How to avoid:**
Design the prediction service from day one with a two-path architecture: a `production_path` (real model) and an `intervention_path` (CheckerBoard predictor). The scoring endpoint must route a configurable fraction of requests through the CheckerBoard. Wire this in Phase 1 before the drift detector exists. Set `mix=0.1` (10% intervention) as a starting default — the paper's experiments show detection degrades above mix=0.25 with strong PD models.

**Warning signs:**
- You build a FastAPI `/score` endpoint that only calls `model.predict()` with no routing logic
- The drift detector module has no way to access which predictions were intervention-assigned vs model-assigned
- You realize you need to retrofit a prediction router into an already-built inference pipeline

**Phase to address:**
Phase 1 (infrastructure) — the two-path prediction router must be in the initial FastAPI design, not added later.

---

### Pitfall 2: Label Availability Assumption — CB-PDD Assumes Immediate True Labels

**What goes wrong:**
The CB-PDD algorithm requires true labels (actual default/no-default outcomes) to compute correctness sets `corr₁` and `corr₂` in Stage 2-3. In credit lending, true labels arrive 30-90 days after loan origination (you don't know if someone defaulted until they miss payments). The paper never discusses label delay — it assumes immediate availability. Implementing drift detection on a 30-day simulation will silently give meaningless results if the label timing is not explicitly modeled.

**Why it happens:**
Academic papers use controlled datasets where labels are always available. The paper's "stream learning" framing implies a real-time stream where labels arrive continuously, but gives no guidance on delay tolerance or windowing strategy.

**How to avoid:**
Build the simulation with explicit label delay. In the Give Me Credit replay, treat each simulated "day" as a batch where outcomes from N days ago become available. Log predictions with a `prediction_timestamp` and labels with an `outcome_timestamp`. The drift detector should only consume instances where both are present. Document this as a known production limitation in the README.

**Warning signs:**
- Your simulation immediately pairs predictions with labels from the same timestep
- The Airflow DAG runs drift detection on the same batch that just generated predictions, with no delay offset
- Your PostgreSQL schema has no `outcome_received_at` column separate from `predicted_at`

**Phase to address:**
Phase 2 (simulation design) — label delay must be modeled when designing the feedback loop replay logic.

---

### Pitfall 3: Conflating Performative Drift with Ordinary Concept Drift

**What goes wrong:**
The project's entire value proposition is that CB-PDD distinguishes performative drift (caused by the model's own predictions) from intrinsic drift (caused by external factors like economic cycles). If the simulation produces both types mixed together, and the detector cannot isolate the performative component, the dashboard will show "drift detected" without proving the paper's claim. The hiring manager cannot tell if you actually implemented the paper's distinguishing insight.

**Why it happens:**
The Give Me Credit dataset has static labels from 2005-2007. To simulate performative drift, you must deliberately modify future-period feature distributions based on what the model predicted. It is tempting to just add random noise to the features to simulate drift — that is intrinsic drift, not performative. The two are mathematically distinct.

**How to avoid:**
Implement two explicit simulation modes: (a) `denial_loop` — denied applicants are removed from future batches (model prediction → selection pressure → distribution shift) and (b) `score_gaming` — approved applicants shift feature values toward approval thresholds in subsequent rounds. Each mode produces performative drift. Use a separate "control" simulation with random noise to produce intrinsic drift. Run CB-PDD on both; document that it fires on (a) and (b) but not the control. This is the empirical proof the paper claims.

**Warning signs:**
- Your simulation has a single `simulate_drift()` function that adds Gaussian noise to features
- No causal mechanism connects the model's output to how the next batch's features are generated
- The drift score rises monotonically regardless of whether the model is deterministic or random

**Phase to address:**
Phase 2 (simulation design) — the causal feedback mechanism is the core scientific contribution; it must be correct.

---

### Pitfall 4: CB-PDD Fails on Class-Imbalanced Datasets (Acknowledged Paper Limitation)

**What goes wrong:**
The paper explicitly acknowledges that "CB-PDD falters when applied to imbalanced datasets," citing the Credit Card dataset (97% majority class) as a failure case. The Give Me Credit dataset has ~7% default rate (13:1 imbalance). If you train on raw data without handling imbalance and then apply CB-PDD, the detector will not work reliably — and you will spend days debugging before finding this is a known paper limitation.

**Why it happens:**
CB-PDD uses density-based correctness counting per feature partition. With heavy class imbalance, the minority class (defaults) has too few instances per partition for the Mann-Whitney U test to reach significance. The paper tested this on Credit Card with 0.17% minority class and found failure; at 7% (Give Me Credit) the results are marginal.

**How to avoid:**
Apply SMOTE or class weighting during model training. More critically, ensure the simulation produces balanced-enough batches for the CheckerBoard regions to have detectable minority class membership. Use stratified sampling when replaying dataset batches. Consider evaluating CB-PDD sensitivity with your specific imbalance ratio before building the full pipeline around it — run the paper's experiments on a subsample of Give Me Credit first (1-2 hours of work) to verify the detector fires.

**Warning signs:**
- Model training uses raw `class_weight='balanced'=None` (sklearn default)
- CB-PDD drift score never crosses threshold despite clearly simulated drift
- All CheckerBoard partition correctness counts are near-zero for the minority class

**Phase to address:**
Phase 1 (baseline model) — handle imbalance in training; Phase 2 — validate CB-PDD fires on your specific dataset before building the orchestration.

---

### Pitfall 5: MLflow Registry Stages Are Deprecated — Auto-Promotion Code Will Break

**What goes wrong:**
The project requires automated model promotion: "MLflow auto-retraining run fired on drift alert, new model promoted if AUC improves." Many tutorials and blog posts use `transition_model_version_stage(name, version, stage="Production")`. This API is deprecated since MLflow 2.9 and marked for removal in a future major version. Code written against the stages API will produce deprecation warnings now and break in future MLflow versions — a red flag in a portfolio project.

**Why it happens:**
Most MLflow tutorials online (including ones from 2023-2024) use the stages API because it was the primary interface for years. The replacement (aliases) was introduced in MLflow 2.9 and the migration is not yet universal in community content.

**How to avoid:**
Use model aliases from day one. Instead of promoting to "Production" stage, set an alias: `client.set_registered_model_alias(name, "champion", version)`. Load the champion model via `mlflow.pyfunc.load_model("models:/credit-risk-scorer@champion")`. This is the current recommended pattern and works in MLflow 2.9+. Never write `stage="Production"` in new code.

**Warning signs:**
- Your Airflow DAG imports `transition_model_version_stage`
- MLflow logs `FutureWarning: ``transition_model_version_stage`` is deprecated`
- You copied a tutorial from 2022-2023 that references staging/production stage names

**Phase to address:**
Phase 3 (MLflow integration) — use aliases in the initial implementation, not stages.

---

### Pitfall 6: The CheckerBoard Parameter τ Has No Production Calibration Guidance

**What goes wrong:**
The trial length τ (default 1000 in the paper) determines how many instances CB-PDD observes per trial window. In the Give Me Credit simulation with 150k rows replayed over 30 simulated days, you have roughly 5,000 instances per day. If τ=1000, you get 5 trials per day, which may or may not match the drift's temporal scale. Setting τ too high means the detector is slow to fire; too low means false positives from noise. The paper says "τ sensitivity to actual drift intensity remains unclear in practice" — there is no principled way to set it.

**Why it happens:**
All paper experiments use synthetic data generators with known drift parameters. The paper cannot advise on real datasets because "the greatest issue is the lack of real-world data with known performative drift."

**How to avoid:**
Treat τ as an explicit hyperparameter of the detection system and expose it as a configurable parameter in the DAG (via Airflow Variables or a config table in PostgreSQL). Document τ selection reasoning in the README: "set to X based on batch size of Y instances/day, targeting detection within Z days." Run sensitivity analysis during simulation: plot drift detection latency vs τ across {500, 1000, 2000} and include this analysis in the dashboard. This turns a limitation into a demonstrated depth of understanding.

**Warning signs:**
- τ is hardcoded in the CB-PDD implementation
- No sensitivity analysis exists in the codebase
- The README does not explain why τ was chosen

**Phase to address:**
Phase 2 (algorithm implementation) — expose τ as config; Phase 4 (monitoring) — add sensitivity analysis to dashboard.

---

### Pitfall 7: Airflow + MLflow + Postgres in Docker-Compose — Port and Credential Collisions

**What goes wrong:**
When all three services (Airflow metadata DB, MLflow tracking backend, and application data) run in the same docker-compose stack, they commonly collide on: the default Postgres port (5432), database names (`mlflow` vs `airflow`), and the `mlflow` container not having `psycopg2` installed by default. The official `mlflow/mlflow` Docker image lacks the postgres driver, causing `ModuleNotFoundError: No module named 'psycopg2'` when you set `--backend-store-uri postgresql://...`.

**Why it happens:**
Most tutorials use SQLite for MLflow tracking (the default). Switching to Postgres for a "production-grade" stack is correct but requires a custom MLflow Dockerfile. This is not obvious from the MLflow docs, and the official image issue has been open on GitHub since 2022.

**How to avoid:**
Create a custom `Dockerfile.mlflow` that extends `ghcr.io/mlflow/mlflow` and adds `pip install psycopg2-binary`. Use separate Postgres databases: `airflow_db` for Airflow metadata, `mlflow_db` for MLflow tracking, `app_db` for application data. Use different ports or a single Postgres with multiple databases. Document all connection strings in a single `.env.example`. Set `depends_on` with `condition: service_healthy` (not just `service_started`) to prevent startup race conditions.

**Warning signs:**
- MLflow container exits with `ModuleNotFoundError` on first `docker-compose up`
- Airflow and MLflow both try to create a `public` schema in the same Postgres database
- `docker-compose up` succeeds but MLflow shows no experiments because it fell back to SQLite

**Phase to address:**
Phase 1 (infrastructure) — resolve all docker-compose configuration before building any ML code.

---

### Pitfall 8: Simulating Feedback Loops on a Static Dataset Produces Circular Evidence

**What goes wrong:**
You simulate the denial loop by removing denied applicants from future batches. But the "future batch" is just the same Give Me Credit CSV re-sampled with exclusions. After several rounds, the remaining pool is increasingly biased toward low-risk applicants, the model achieves high AUC on that restricted population, and drift detection may fire — but you have not proved the algorithm works, only that exclusion sampling changes the distribution. A skeptical hiring manager or technical interviewer will ask: "How do you know this is performative drift and not just selection bias from your sampling procedure?"

**Why it happens:**
Static dataset replay inherently cannot produce ground-truth performative effects. The paper addresses this with a synthetic data generator (centroids with σ-weighted shifts). Using a real dataset for the simulation is methodologically cleaner for a portfolio but requires careful documentation.

**How to avoid:**
Document the simulation mechanism explicitly in the README with a diagram: "Round N: model predicts → denied applicants removed → Round N+1 draws from remaining pool." State clearly that this is an approximation of performative effects, not a ground-truth simulation. Include a null hypothesis test: run the same pipeline with a random predictor and verify CB-PDD does NOT fire (because a random predictor cannot cause systematic performative drift). This is a strong proof of concept.

**Warning signs:**
- Your README does not explain how feedback is simulated mechanically
- There is no negative control (random predictor) in your experiments
- The drift score increases at the same rate regardless of model quality

**Phase to address:**
Phase 2 (simulation) — design the null control experiment upfront; include in dashboard.

---

### Pitfall 9: FastAPI Model Loading on Cloud Run — Cold Starts Kill Demo Latency

**What goes wrong:**
Cloud Run is stateless. Each cold start loads the entire scikit-learn/XGBoost model from MLflow artifacts (network fetch + deserialization). For a gradient boosting model on Give Me Credit, this can take 10-30 seconds per cold start. A hiring manager clicking the live demo URL after it has been idle gets a timeout. The demo dies on its most important test.

**Why it happens:**
Local development always has the model loaded in memory. Cloud Run's scale-to-zero behavior is invisible locally. The default Cloud Run timeout is 300 seconds but the request that triggers the cold start may timeout before the model is ready.

**How to avoid:**
Load the model at application startup, not per-request, using FastAPI's lifespan context manager. Set Cloud Run minimum instances to 1 (prevents scale-to-zero; ~$5/month for the demo period). Use `python:3.11-slim` base image to minimize container size and cold start time. Pre-download the model artifact into the container image at build time (bake it in) rather than fetching from MLflow on startup. For the portfolio, document the min-instances setting in the README so reviewers understand the cost tradeoff.

**Warning signs:**
- Model is loaded inside the request handler function, not at app startup
- No Cloud Run minimum instances configuration in the deployment script
- First request to the deployed API takes >10 seconds

**Phase to address:**
Phase 5 (deployment) — address at Cloud Run deployment time, not after.

---

### Pitfall 10: Airflow DAG Duplicate IDs and Silent Parsing Failures

**What goes wrong:**
Airflow silently drops DAGs with duplicate `dag_id` values — no error, no log, the DAG simply disappears from the UI. This is especially dangerous when refactoring DAG files or when two developers create DAGs with similar names. Additionally, Airflow parses all DAG files at import time; a DAG file with a Python import error will silently fail to load, and the scheduler will not run it.

**Why it happens:**
Airflow's DAG discovery mechanism scans all `.py` files in the dags folder. Duplicate IDs or import-time errors produce no visible failures in the UI — the DAG just does not appear. This is described in the Astronomer "7 common errors" documentation.

**How to avoid:**
Use a DAG naming convention that includes the project prefix: `credit_risk_daily_drift_detection`, `credit_risk_weekly_retrain`. Add a DAG validation test that imports all DAG files and checks for parse errors (pytest can do this in <1 second). Set `catchup=False` on all DAGs to prevent historical backfill on first deployment. Use `schedule_interval=None` for manual-trigger-only DAGs during development to avoid accidental runs.

**Warning signs:**
- A DAG you created does not appear in the Airflow UI
- No error is visible in the scheduler logs at a glance
- Two DAG files in the dags/ directory share a dag_id

**Phase to address:**
Phase 3 (orchestration) — include DAG validation in CI from the start.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| SQLite for MLflow tracking | Zero config, works immediately | Cannot share between Airflow workers, breaks multi-process access | Never — Postgres is required from day one given Airflow also runs in Docker |
| Hardcoded τ in CB-PDD | Simpler code | Cannot demonstrate parameter sensitivity, looks naive to technical reviewers | Never — it is a key algorithm parameter |
| Loading model per-request in FastAPI | Simpler code | Cold start kills live demo on Cloud Run | Never — lifespan loading costs 5 lines of code |
| Using MLflow stages instead of aliases | Follows old tutorials | Deprecated API, deprecation warnings in logs, breaks in future MLflow | Never — aliases are the current standard |
| Single Postgres database for all services | Simpler docker-compose | Airflow and MLflow schema conflicts, harder to debug data isolation | Only in a single-developer prototype with a plan to separate later |
| Skipping the null control experiment | Faster to implement | Cannot prove CB-PDD distinguishes performative from intrinsic drift | Never — the null control is the scientific proof |
| catchup=True in Airflow DAGs | Default behavior | Triggers historical backfills on first deploy, fills logs with failed runs | Never for this project |

---

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| MLflow + Postgres | Using official `mlflow/mlflow` image with `--backend-store-uri postgresql://` | Build custom image: `FROM ghcr.io/mlflow/mlflow && pip install psycopg2-binary` |
| Airflow + MLflow | Calling `mlflow.set_tracking_uri()` inside DAG body (parsed at import) | Set `MLFLOW_TRACKING_URI` as Airflow environment variable; use Connection objects |
| FastAPI + MLflow model | Fetching model from tracking server on every request | Load at startup with `lifespan` context; bake model artifact into Docker image for Cloud Run |
| Streamlit + Postgres | Direct Postgres connection from Streamlit (no connection pooling) | Use a thin REST API layer or SQLAlchemy connection pool; Streamlit re-runs on every user interaction |
| GCP Cloud Run + MLflow | Cloud Run can't reach local MLflow tracking server | Use Cloud SQL (Postgres) for MLflow backend in GCP; or bake model artifact into image |
| GitHub Actions + GCP | Passing GCP credentials as plaintext in workflow YAML | Use Workload Identity Federation or store credentials as encrypted GitHub secrets |
| Airflow + Docker-compose | `depends_on: postgres` without health check | Use `depends_on: postgres: condition: service_healthy` with a postgres healthcheck |

---

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Loading full Give Me Credit CSV on every simulation run | Airflow task takes 30s+ to start | Load once, store processed batches in Postgres | At 30 simulation rounds (150 * 30 = 4,500k row reads) |
| Re-computing CB-PDD from scratch on every Airflow run | Detection task slow; no incremental state | Persist CB-PDD window state in Postgres between runs | After ~10 simulation rounds when window state grows |
| Logging every prediction to Postgres synchronously in FastAPI | API latency increases under load | Use async writes or a write-ahead buffer | At >100 simultaneous scoring requests |
| Pulling all MLflow run history to Streamlit on each dashboard refresh | Dashboard refresh >10s | Cache run history with `@st.cache_data(ttl=300)` | When MLflow has >100 runs logged |
| Docker-compose Airflow with LocalExecutor | Sequential task execution | Acceptable for portfolio scale; document the limitation | At >20 concurrent DAG runs |

---

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Committing `.env` file with GCP credentials to GitHub | Credential exposure, GCP account compromise | Use `.gitignore` for `.env`; use GitHub Secrets; rotate keys immediately if exposed |
| Exposing MLflow tracking server publicly on Cloud Run | Unauthorized model uploads, experiment pollution | MLflow behind authenticated proxy or VPC-only; for portfolio, document that auth is out of scope |
| Using `SeriousDelinquency` as a direct feature in scoring | Target leakage — the label is in the features | Explicitly exclude target column in feature list; add assertion in preprocessing pipeline |
| Storing raw credit features without noting PII implications | Regulatory risk in real systems | Add README note: "Would require PII handling, differential privacy, and audit logs in production" — signals awareness |
| Airflow webserver with default admin/admin credentials in docker-compose | Trivial takeover of orchestration | Change default credentials in docker-compose env vars; document in README |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **CB-PDD implementation:** Often missing the intervention routing layer — verify the prediction endpoint has a `mix` parameter and routes a fraction through CheckerBoard logic
- [ ] **Drift detection:** Often shows "drift detected" without a negative control — verify a random predictor does NOT trigger drift detection
- [ ] **Auto-retraining:** Often triggers retraining but promotes unconditionally — verify AUC comparison gates promotion (new_auc > current_champion_auc - epsilon)
- [ ] **Airflow DAG:** Often shows as "success" in UI but tasks ran 0 times due to catchup=False and future start_date — verify DAG ran the intended number of times in logs
- [ ] **MLflow model loading:** Often works locally but fails on Cloud Run — verify cold start latency under 5 seconds by checking Cloud Run metrics tab
- [ ] **Simulation:** Often produces drift score that always increases — verify score DECREASES after successful retraining with uncontaminated data
- [ ] **Dashboard:** Often shows static charts with no update mechanism — verify Streamlit charts refresh when new simulation data is written to Postgres
- [ ] **Feedback loop types:** Often implements denial_loop only — verify score_gaming scenario also triggers CB-PDD (tests both paper causal mechanisms)
- [ ] **Give Me Credit preprocessing:** Verify MonthlyIncome NaNs (19.8% missing) are handled — using `fillna(0)` introduces systematic bias toward median imputation

---

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| CB-PDD intervention layer missing from scorer | HIGH | Refactor prediction endpoint to split-path routing; update all tests; re-run simulation from scratch |
| Label delay not modeled in simulation | MEDIUM | Add `outcome_received_at` column to prediction log; rebuild simulation replay with N-day offset; re-run drift detection |
| MLflow stages API used throughout | LOW | Search-replace `transition_model_version_stage` → `set_registered_model_alias`; update alias-based loading URIs |
| docker-compose Postgres conflicts | LOW | Rename databases in env vars; delete volumes (`docker-compose down -v`); restart |
| CB-PDD never fires due to class imbalance | MEDIUM | Apply SMOTE in training; re-run simulation with stratified batches; validate τ is appropriate for batch size |
| Cloud Run cold starts killing demo | LOW | Set `--min-instances 1` in Cloud Run deployment config; bake model into image |
| Simulation does not produce distinct drift types | HIGH | Redesign simulation with explicit causal mechanism functions; requires re-running all 30-day simulation data |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| CB-PDD intervention layer missing | Phase 1 (FastAPI infrastructure) | Split-path routing in initial endpoint design |
| Label delay not modeled | Phase 2 (simulation design) | `outcome_received_at` column present in schema from day one |
| Performative vs intrinsic drift conflated | Phase 2 (simulation design) | Null control experiment exists and CB-PDD does NOT fire for random predictor |
| Class imbalance breaks CB-PDD | Phase 1 (baseline model) + Phase 2 validation | CB-PDD fires on simulated denial loop before building orchestration |
| MLflow stages deprecated | Phase 3 (MLflow integration) | No `transition_model_version_stage` in codebase; aliases used throughout |
| τ not calibrated | Phase 2 (algorithm impl) + Phase 4 (monitoring) | τ exposed as config; sensitivity chart in dashboard |
| Docker-compose collisions | Phase 1 (infrastructure) | Full stack `docker-compose up` runs without errors on first attempt |
| Static dataset circular evidence | Phase 2 (simulation design) | Null control experiment documented in README |
| Cloud Run cold starts | Phase 5 (deployment) | Cold start latency <5s verified in Cloud Run metrics |
| Airflow silent DAG failures | Phase 3 (orchestration) | DAG validation test in CI pipeline |

---

## Sources

- [arxiv 2412.10545 full HTML — CB-PDD algorithm details, limitations, parameter guidance](https://arxiv.org/html/2412.10545) — HIGH confidence (primary source)
- [AAAI 2025 published version of CB-PDD paper](https://ojs.aaai.org/index.php/AAAI/article/view/33276/35431) — HIGH confidence
- [MLflow Model Registry — alias documentation](https://mlflow.org/docs/latest/model-registry/) — HIGH confidence (official docs)
- [RFC: deprecating MLflow model registry stages — Issue #10336](https://github.com/mlflow/mlflow/issues/10336) — HIGH confidence (official GitHub)
- [Astronomer: 7 Common Airflow Debugging Errors](https://www.astronomer.io/blog/7-common-errors-to-check-when-debugging-airflow-dag/) — MEDIUM confidence
- [Official mlflow/mlflow image lacks psycopg2 — Issue #9513](https://github.com/mlflow/mlflow/issues/9513) — HIGH confidence (confirmed bug)
- [Valohai: MLflow+Airflow+Kubernetes integration problems](https://valohai.com/blog/the-mlflow-airflow-kubernets-makeshift-monster/) — MEDIUM confidence
- [Give Me Some Credit dataset statistics — 19.8% MonthlyIncome NaN rate](https://medium.com/@chirag.sharma0378/give-me-some-credit-machine-learning-case-study-7178edef0d47) — MEDIUM confidence (confirmed across multiple Kaggle notebooks)
- [Feedback loops and bias in automated decision-making systems — ACM FAccT](https://dl.acm.org/doi/fullHtml/10.1145/3617694.3623227) — HIGH confidence (peer-reviewed)
- [Performative Prediction: Past and Future — survey of foundational issues](https://arxiv.org/pdf/2310.16608) — HIGH confidence
- [ZenML: From research papers to production AI — gaps and assumptions](https://www.zenml.io/blog/from-research-papers-to-production-ai) — MEDIUM confidence
- [GCP Cloud Run cold start mitigation strategies](https://medium.com/google-cloud/3-solutions-to-mitigate-the-cold-starts-on-cloud-run-8c60f0ae7894) — MEDIUM confidence

---
*Pitfalls research for: Credit Risk MLOps Pipeline with Performative Drift Detection*
*Researched: 2026-03-24*
