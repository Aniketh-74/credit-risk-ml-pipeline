# Feature Research

**Domain:** Production-grade MLOps pipeline — credit risk scoring with performative drift detection
**Researched:** 2026-03-24
**Confidence:** MEDIUM-HIGH (paper algorithm: HIGH from primary source; MLOps table stakes: HIGH from multiple verified sources; portfolio signal claims: MEDIUM from industry surveys)

---

## Feature Landscape

### Table Stakes (Hiring Managers Expect These)

Features every serious MLOps portfolio project must have. Missing any of these signals incomplete understanding of production ML.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Baseline model on real dataset | Proves you can train a non-trivial model, not just hello-world | LOW | XGBoost/LightGBM on Give Me Credit (150k rows, 10 features, class imbalance ~6.7%). AUC ~0.86 is achievable baseline. |
| MLflow experiment tracking | Standard industry tool; recruiters check whether you log params, metrics, artifacts | LOW | Log: hyperparams, AUC, precision-recall curve, feature importances per run. Use MLflow autolog where possible. |
| MLflow model registry with staging/production | Model lifecycle management is table stakes for MLOps roles | LOW | Register models, transition staging → production on AUC improvement. Version history tells the drift story. |
| FastAPI prediction endpoint | Deployed model serving is mandatory for portfolio credibility | LOW-MEDIUM | POST /predict accepting JSON loan application, returning score + model version. Health endpoint. |
| Docker + docker-compose | Containerisation is assumed; any project without it looks like 2018 | LOW | All services (API, MLflow, Airflow, Postgres, Streamlit) in compose. Single `docker-compose up` must work. |
| PostgreSQL for prediction + outcome logging | Closing the feedback loop requires persisted predictions tied to outcomes | MEDIUM | Schema: predictions table (application_id, features, score, model_version, timestamp) + outcomes table. Enables drift score computation. |
| Airflow DAG for pipeline orchestration | Orchestration is explicitly on the skills checklist for MLE/Data Engineer roles | MEDIUM | DAG: ingest data → score batch → compute drift → conditional retrain → promote model. Daily schedule. |
| Data drift detection (standard) | Evidently/PSI-based drift is now boilerplate — must be present as baseline | LOW | Population Stability Index (PSI) on score distribution. Threshold: warning at 0.1, critical at 0.25. Complements the novel algorithm. |
| Automated retraining trigger | Closed-loop MLOps (detect → retrain → promote) is what separates "deployed" from "production" | MEDIUM | Trigger on drift alert. New model only promoted if AUC improves over current production model. |
| CI/CD with GitHub Actions | Automated testing and deployment pipeline signals engineering discipline | LOW-MEDIUM | On push: run tests, build Docker image, push to GCR, deploy to Cloud Run. |
| Model performance monitoring | Track AUC/precision/recall over time as outcomes accumulate | LOW | Computed in daily Airflow DAG. Stored in MLflow. Surfaced in dashboard. |
| Recruiter-facing README | Recruiters spend <90 seconds; README must explain value, show architecture diagram, link live demo | LOW | Architecture diagram (draw.io or Mermaid), "what I learned" section, live demo URL. This is non-negotiable for portfolio. |

---

### Differentiators (What Makes This Project Stand Out)

These are the features that separate this project from the 99% of ML portfolios using static datasets with no feedback loop modelling. The performative drift story is the entire differentiation thesis.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| CB-PDD algorithm (faithful implementation of arxiv 2412.10545) | No production implementation exists. Implementing a December 2024 AAAI paper signals research depth that most engineers lack. | HIGH | See "Performative Drift Algorithm: Minimal vs Full" section below. Core module: ~200-300 lines Python. |
| Denial loop simulation | Models the self-defeating feedback loop specific to credit lending: denied applicants never default → model learns denied group = high risk | MEDIUM | Simulate by replaying predictions into future population: applicants scored below threshold are removed from next training batch, causing distributional shift in observed outcomes. |
| Score gaming loop simulation | Models the self-fulfilling feedback loop: applicants who learn scoring criteria game their features, shifting the population | MEDIUM | Simulate by applying feature perturbations to applicants near the decision boundary, representing applicants who optimise their presentation. |
| Performative drift score time series in dashboard | Visual narrative showing drift score rising over 30 simulated days, alert firing, model retraining, score dropping back — this is the interview story | MEDIUM | Streamlit dashboard: drift score line chart with threshold line, model version markers on retraining events, alert log table. |
| Dual drift type classification | Detecting and labelling which feedback loop type is occurring (self-fulfilling vs self-defeating) is a research contribution beyond standard drift detection | MEDIUM-HIGH | CB-PDD distinguishes these via the direction of density change (Group A vs Group B). Surface this distinction in the dashboard and API response. |
| Bias-corrected retraining | After detecting performative drift, the retraining strategy adjusts for the selection bias introduced by the feedback loop | HIGH | For denial loop: inverse propensity score weighting or reject inference to recover censored outcomes. This is what production credit teams actually do and rarely appear in portfolios. |
| GCP Cloud Run deployment | Public live demo URL is what converts "interesting" to "let's schedule a call" | MEDIUM | Stateless FastAPI on Cloud Run + Cloud SQL for Postgres. GitHub Actions deploys on main merge. |
| 30-day drift evolution simulation | Demonstrates the pipeline over time, not just a point-in-time prediction — this proves understanding of temporal model degradation | MEDIUM | Replay Give Me Credit data in time-ordered batches with drift injected. Precomputed simulation data stored in Postgres. |

---

### Anti-Features (Do Not Build These)

Features that seem like good additions but will hurt portfolio signal by diluting focus or adding scope that exceeds the 30-day timeline.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Real-time Kafka streaming | Looks impressive on paper | Adds 1-2 weeks of infrastructure setup that obscures the drift story. The causal mechanism is visible in batch. Kafka is a separate portfolio signal. | Scheduled batch pipeline (daily Airflow DAG) is sufficient and closer to how most credit lenders actually operate. |
| Multi-model ensemble | More models = better accuracy | Complicates the drift narrative. The story is about feedback loops, not model selection. Ensemble adds complexity without improving the research story. | Single gradient boosting model (XGBoost or LightGBM). One model version history is cleaner for the dashboard story. |
| Auth / multi-tenant API | Looks production-ready | Adds 3-5 days of auth plumbing (JWT, user management) that is invisible to hiring managers and wastes time. | Single-user API with API key in env var. Comment in README: "auth would be added for multi-tenant deployment." |
| Kubernetes deployment | Signals infrastructure depth | Overkill for a 30-day project. K8s setup alone consumes a week. Cloud Run provides the same "deployed in the cloud" signal without the overhead. | GCP Cloud Run. Containerised, scalable, public URL, no K8s overhead. |
| Evidently full monitoring suite | Feature-rich monitoring | Evidently is table stakes (see above) but building a comprehensive Evidently setup competes with the CB-PDD algorithm as the narrative focus. | Use Evidently for PSI only, as a baseline comparison to CB-PDD. Do not build custom Evidently dashboards. |
| SHAP / explainability deep dive | Important for credit decisions | Interesting but a separate portfolio dimension. SHAP adds 3-4 days and creates a second story that dilutes the drift narrative. | Log feature importances from MLflow. One SHAP summary plot in the README is sufficient as a mention. |
| Online learning / streaming retraining | Cutting-edge | Online learning changes the retraining architecture fundamentally and makes the drift detection story harder to explain. CB-PDD was designed for batch scenarios. | Batch retraining triggered by drift alert. Simpler, more explainable, faithful to the paper's design. |
| Mobile or frontend beyond Streamlit | Broadens audience | JavaScript frontend adds a week and changes the project category from MLOps to full-stack. Hiring managers for MLE roles do not value this. | Streamlit. Fast to build, Python-native, visually adequate for portfolio demonstration. |
| Hyperparameter tuning (Optuna/Ray) | Optimised models look serious | Fine-tuning to squeeze 0.5% AUC improvement is not the story. The story is drift detection. Optuna integration takes 2 days for marginal gain. | Fixed hyperparameters from a single grid search. Document the choice. Move on. |
| Real credit bureau data / PII | Regulatory realism | Legal complexity, data access barriers, PII handling requirements. Give Me Credit is publicly available, well-known to hiring managers, and sufficient. | Give Me Credit (Kaggle). The simulation layer on top demonstrates understanding of real production challenges. |

---

## Performative Drift Algorithm: Minimal Viable vs Full Paper

This section specifically addresses the quality gate requirement: what is minimal faithful implementation vs full production system.

### What CB-PDD Actually Detects

CB-PDD (CheckerBoard Performative Drift Detection, arxiv 2412.10545, AAAI 2025) detects when a deployed model's predictions cause the future data distribution to shift. It distinguishes:

- **Self-fulfilling loop**: Predictions cause the predicted outcome to become more likely (e.g., model approves applicants → approved applicants build credit history → model validates itself)
- **Self-defeating loop**: Predictions cause the predicted outcome to become less likely (e.g., model denies loans → denied group cannot default → model sees them as lower risk than they are — the denial loop)

The key insight: standard drift detectors cannot distinguish whether drift was caused by the model or by external factors. CB-PDD can, because it uses a controlled A/B testing design (the checkerboard) that observes whether changing predictions changes the distribution.

### Minimal Faithful Implementation

A minimal implementation that faithfully represents the paper requires:

**Core module: `performative_drift/cb_pdd.py`**

1. `CheckerBoardPredictor` class
   - `__init__(f, tau, alpha, w)` — four parameters only
   - `predict(instance)` — assigns group label based on feature value and current period parity
   - `flip()` — called every tau instances to alternate labels

2. `DensityChangeTracker` class
   - `add_instance(instance, prediction, label)` — streams instances in
   - `compute_window_density(window)` — proportion of instances per group in a window of size w
   - `get_trial_result()` — returns (group_A_density_change, group_B_density_change) at trial end

3. `PerformativeDriftDetector` class
   - `update(instance, label)` — main entry point
   - `_run_statistical_test()` — Mann-Whitney U test (scipy.stats.mannwhitneyu) on accumulated trial results
   - `is_drift_detected()` — returns bool
   - `drift_type()` — returns "self_fulfilling" | "self_defeating" | None based on direction of density change

**Parameters for Give Me Credit scenario:**
- `tau = 1000` (trial length in instances — matches paper defaults)
- `w = 100` (window size — matches paper defaults)
- `alpha = 0.01` (significance threshold)
- `f = 1.0` (feature split — use the full feature range, normalised)

**Dependencies:** `numpy`, `scipy` only. No ML framework needed for the detector itself.

**Estimated lines of code:** 200-300 Python. This is achievable in 2-3 focused sessions.

### What a Minimal Implementation Deliberately Omits

The paper contains additional material not required for a faithful production implementation:

- Synthetic dataset generation for benchmarking (paper section 4.1) — not needed in production
- Comparison with ADWIN, DDM, EDDM baselines (paper section 4.3) — interesting for research, not required for production
- Multi-feature checkerboard patterns beyond binary (paper appendix) — single-feature split is sufficient
- Statistical power analysis and parameter sensitivity experiments (paper section 4.4) — document parameters used, skip the sensitivity sweep

### Full Production Wrapper (What This Project Adds Beyond the Paper)

The paper provides the detection algorithm. This project wraps it in:

- Streaming prediction logging to PostgreSQL
- Airflow DAG calling the detector on daily batches
- Alert generation and storage on threshold crossing
- MLflow retraining run triggered by alert
- Model promotion logic (AUC comparison)
- Dashboard visualization of drift score time series

This is the production value that does not exist in the paper and does not exist as an open-source implementation. This is the interview story.

---

## Feature Dependencies

```
[Give Me Credit baseline model + MLflow tracking]
    └──requires──> [PostgreSQL prediction/outcome schema]
                       └──requires──> [FastAPI prediction endpoint]
                                          └──requires──> [Docker compose setup]

[CB-PDD algorithm module]
    └──requires──> [Prediction logging in PostgreSQL]
    └──requires──> [Drift simulation layer (denial loop + gaming loop)]

[Airflow DAG: drift detection + retrain]
    └──requires──> [CB-PDD algorithm module]
    └──requires──> [MLflow model registry]
    └──requires──> [Automated retraining trigger]

[Streamlit dashboard]
    └──requires──> [Drift score time series in PostgreSQL]
    └──requires──> [MLflow model version history]
    └──requires──> [Alert log in PostgreSQL]

[GCP Cloud Run deployment]
    └──requires──> [GitHub Actions CI/CD]
    └──requires──> [Docker compose → production Dockerfile]

[Bias-corrected retraining]
    └──requires──> [Denial loop simulation]
    └──enhances──> [Automated retraining trigger]

[Standard PSI drift detection]
    └──enhances──> [CB-PDD algorithm module] (baseline comparison)
    └──conflicts──> [Building full Evidently dashboard] (anti-feature: dilutes focus)
```

### Dependency Notes

- **Prediction logging requires PostgreSQL schema first**: The schema must be defined before the API can log. Build schema in Phase 1 before API in Phase 2.
- **CB-PDD requires simulation data**: The algorithm needs a data stream with drift injected. Build simulation layer before wiring up the detector.
- **Dashboard requires all upstream components**: Streamlit is the last piece. Build it only after drift scores are flowing.
- **Bias-corrected retraining enhances the denial loop story**: Not strictly required for MVP, but elevates the research depth significantly if time allows.

---

## MVP Definition

### Launch With (v1) — 30-day target

- [ ] Give Me Credit baseline model (XGBoost, AUC > 0.85) with MLflow tracking — establishes the "before drift" baseline
- [ ] PostgreSQL schema for predictions + outcomes — enables everything downstream
- [ ] FastAPI /predict endpoint + Docker compose — makes it deployable and testable
- [ ] CB-PDD algorithm module (faithful paper implementation) — the core differentiator; must be present
- [ ] Denial loop simulation (self-defeating feedback loop) — more compelling interview story than gaming loop; build this first
- [ ] Airflow DAG: daily score → detect drift → retrain if needed → promote model — demonstrates the closed loop
- [ ] Streamlit dashboard: drift score time series + alert log + model version history — the visual story for README and interviews
- [ ] GCP Cloud Run deployment with GitHub Actions CI/CD — live demo URL converts interest to interviews
- [ ] Recruiter-facing README with architecture diagram — mandatory for portfolio

### Add After Validation (v1.x) — if time allows in 30-day window

- [ ] Score gaming loop simulation (self-fulfilling loop) — adds second drift type, deepens the research story; add after denial loop is working
- [ ] Bias-corrected retraining (inverse propensity weighting) — production-grade addition that very few portfolios have; add if Day 20+ has slack
- [ ] Standard PSI drift detection as baseline comparison — makes CB-PDD results more credible; low effort addition

### Future Consideration (v2+) — out of scope for portfolio phase

- [ ] SHAP explainability — separate portfolio signal; build as a separate project or add-on after v1 ships
- [ ] Kafka streaming layer — separate portfolio signal demonstrating streaming infrastructure
- [ ] Multi-model comparison — only relevant if the project pivots toward model selection research

---

## Feature Prioritization Matrix

| Feature | Portfolio Value | Implementation Cost | Priority |
|---------|-----------------|---------------------|----------|
| CB-PDD algorithm (faithful) | HIGH | MEDIUM | P1 |
| Give Me Credit baseline model + MLflow | HIGH | LOW | P1 |
| FastAPI prediction endpoint | HIGH | LOW | P1 |
| Docker + docker-compose | HIGH | LOW | P1 |
| PostgreSQL prediction logging | HIGH | LOW | P1 |
| Airflow DAG (full loop) | HIGH | MEDIUM | P1 |
| Denial loop simulation | HIGH | MEDIUM | P1 |
| Streamlit dashboard | HIGH | MEDIUM | P1 |
| GitHub Actions CI/CD | HIGH | LOW | P1 |
| GCP Cloud Run deployment | HIGH | MEDIUM | P1 |
| Recruiter README + arch diagram | HIGH | LOW | P1 |
| Score gaming loop simulation | MEDIUM | MEDIUM | P2 |
| Standard PSI drift (baseline) | MEDIUM | LOW | P2 |
| Bias-corrected retraining | HIGH | HIGH | P2 |
| SHAP explainability | LOW | MEDIUM | P3 |
| Kafka streaming | LOW | HIGH | P3 |
| Full Evidently dashboard | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for launch — project is incomplete without it
- P2: Should have — adds interview depth, build if Day 20+ has slack
- P3: Nice to have — separate project or v2

---

## Competitor Feature Analysis

Comparing against other credit risk / MLOps GitHub portfolios and what this project does differently.

| Feature | Typical Credit Risk Portfolio | JakobLS/mlops-credit-risk (comparable GitHub project) | This Project |
|---------|-------------------------------|-------------------------------------------------------|--------------|
| Dataset | Give Me Credit or similar | Give Me Credit | Give Me Credit (same, but with simulation layer on top) |
| Drift detection | PSI or Evidently standard drift | Evidently data drift reports | CB-PDD (novel AAAI 2025 algorithm) + PSI as baseline |
| Feedback loop modelling | None | None | Explicit simulation of two causal loop types |
| Retraining | Manual or scheduled | Scheduled (Prefect weekly) | Alert-triggered (closed loop) |
| Algorithm novelty | None | None | Faithful implementation of paper with no existing production version |
| Dashboard | Grafana or none | None described | Streamlit showing drift evolution + alert timeline |
| Cloud deployment | Rarely | GCS bucket | GCP Cloud Run with live demo URL |
| Interview story | "I built a credit scoring model" | "I deployed a model with monitoring" | "I implemented a 2024 paper on feedback loops in lending and showed how a model causes its own degradation" |

---

## Sources

- arxiv 2412.10545 HTML version: [Identifying Predictions That Influence the Future: Detecting Performative Concept Drift in Data Streams](https://arxiv.org/html/2412.10545v1) — HIGH confidence, primary source
- AAAI 2025 proceedings: [CB-PDD paper](https://ojs.aaai.org/index.php/AAAI/article/view/33276) — HIGH confidence
- MLOps production pipeline requirements: [MLOps Principles](https://ml-ops.org/content/mlops-principles) — HIGH confidence
- Credit risk MLOps specifics: [How to Build a Risk-Scoring Engine: MLOps for Financial Services](https://www.appitsoftware.com/blog/how-to-build-risk-scoring-engine-mlops-financial-services) — MEDIUM confidence
- Portfolio hiring expectations: [ML Engineer Portfolio Projects That Will Get You Hired in 2025](http://www.interviewnode.com/post/ml-engineer-portfolio-projects-that-will-get-you-hired-in-2025) — MEDIUM confidence (industry survey, not official)
- Comparable GitHub project: [JakobLS/mlops-credit-risk](https://github.com/JakobLS/mlops-credit-risk) — MEDIUM confidence (single example)
- MLOps monitoring patterns: [MLOps Best Practices 2025](https://www.dataa.dev/2025/03/17/mlops-best-practices-production-ml-pipelines-2025/) — MEDIUM confidence
- Performative prediction research context: [When Predictions Shape Reality](https://arxiv.org/abs/2601.04447) — HIGH confidence (peer reviewed)

---

*Feature research for: production-grade MLOps credit risk pipeline with performative drift detection*
*Researched: 2026-03-24*
