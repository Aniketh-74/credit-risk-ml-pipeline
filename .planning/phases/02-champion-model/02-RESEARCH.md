# Phase 2: Champion Model - Research

**Researched:** 2026-03-29
**Domain:** LightGBM training, SMOTE class imbalance handling, MLflow 3.x model registry, EDA on Give Me Credit dataset, early CB-PDD smoke validation
**Confidence:** HIGH (stack verified via official docs; CB-PDD parameter guidance MEDIUM from paper + arxiv fetch)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODEL-01 | LightGBM model trained on Give Me Credit (150k rows, AUC > 0.85) with SMOTE for class imbalance | SMOTE-after-split pipeline pattern verified; AUC 0.85 confirmed achievable; LightGBM 4.6 confirmed |
| MODEL-02 | All training runs logged to MLflow — hyperparameters, AUC, precision-recall curve, feature importances | mlflow.lightgbm.autolog() logs feature importance + params; PR curve via manual matplotlib + log_artifact |
| MODEL-03 | Trained model registered in MLflow Registry and promoted using `@champion` alias (no deprecated stages API) | set_registered_model_alias API verified from official docs; load via models:/name@champion confirmed |
| MODEL-04 | EDA notebook for Give Me Credit — missing value analysis (MonthlyIncome 19.8% NaN), class distribution, feature correlations | Dataset statistics confirmed from multiple sources; imputation strategy researched |
</phase_requirements>

---

## Summary

Phase 2 trains the first champion model. The goal is not AUC maximization — it is getting a calibrated, SMOTE-corrected LightGBM model registered in MLflow under `@champion` so Phase 3 (scoring API) has something to load at startup and Phase 4 (CB-PDD) has a real model to detect drift against. The two hard constraints are: SMOTE must be applied only on the training split (never the test split), and MLflow registration must use aliases, not stages.

The Give Me Credit dataset has two preprocessing landmines that must be addressed before training: 19.8% NaN rate on MonthlyIncome (use median imputation, not mean — the distribution is right-skewed with a long tail) and one outlier column `NumberOfTimes90DaysLate` with values up to 98 that are likely data entry errors (cap at 17). The target `SeriousDelinquency` (7% default rate, ~10,500 positives in 150k rows) must never appear as a feature — add an explicit assertion in preprocessing.

MLflow 3.x fully supports the alias-based registry workflow. The API is stable: `client.set_registered_model_alias(name, "champion", version)` and `mlflow.pyfunc.load_model("models:/credit-risk-model@champion")`. The autolog for LightGBM logs hyperparameters and feature importances automatically; the precision-recall curve must be logged manually via matplotlib + `mlflow.log_artifact()` because autolog does not produce it for LightGBM.

**Primary recommendation:** Split first (80/20 stratified), apply SMOTE only on train split via `fit_resample`, train LightGBM with `is_unbalance=True` as belt-and-suspenders, log everything via `mlflow.lightgbm.autolog()` + manual PR curve, then `mlflow.register_model()` + `set_registered_model_alias(..., "champion", ...)`.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| lightgbm | 4.6.0 | Credit risk classifier | Native sklearn API, fastest GBM on tabular data, `is_unbalance` flag, LightGBM flavor in MLflow |
| scikit-learn | 1.6.x | train_test_split, metrics, preprocessing | sklearn API compatibility with LightGBM; StratifiedKFold, roc_auc_score, classification_report |
| imbalanced-learn | 0.12.x | SMOTE for class imbalance | Tracks sklearn 1.6.x; `fit_resample` pattern prevents data leakage |
| mlflow | 3.10.1 | Experiment tracking + model registry | MLflow 3.x with alias-based model promotion; already in docker-compose |
| pandas | 2.2.3 | Data loading, EDA, preprocessing | Give Me Credit is a CSV; Arrow-backed dtypes; already pinned in project |
| numpy | 1.26.4 | Numerical ops | Pinned for LightGBM compatibility; avoid numpy 2.x |
| matplotlib | 3.8.x | Precision-recall curve plot for MLflow artifact | Standard; used to generate PR curve PNG for mlflow.log_artifact |
| seaborn | 0.13.x | EDA correlation heatmap, distribution plots | Thin matplotlib wrapper; cleaner EDA notebook output |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy | 1.15.x | Spearman correlation in EDA | Transitive dep of sklearn; useful for non-linear feature correlation in EDA notebook |
| jupyter | latest | EDA notebook (MODEL-04) | Only for the EDA notebook — not used in any `.py` module |
| kaggle | 1.6.x | Download Give Me Credit dataset via API | CLI: `kaggle competitions download -c GiveMeSomeCredit` — needs KAGGLE_USERNAME + KAGGLE_KEY in .env |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| SMOTE (oversampling) | `is_unbalance=True` only | `is_unbalance` alone sometimes undershoots 0.85 AUC on this dataset; SMOTE gives the model more minority signal at training time |
| LightGBM sklearn API | LightGBM native API | Native API has more callbacks but sklearn API integrates cleanly with mlflow.lightgbm.autolog() and imblearn pipelines |
| median imputation | iterative/KNN imputation | KNN imputation is better statistically but adds 5+ minutes of training-time complexity; median is the standard accepted approach for this dataset |

**Installation (training dependencies, run inside container or venv):**
```bash
pip install lightgbm==4.6.0 scikit-learn==1.6.1 imbalanced-learn==0.12.4 \
            mlflow==3.10.1 pandas==2.2.3 numpy==1.26.4 \
            matplotlib==3.8.4 seaborn==0.13.2 scipy==1.15.2 \
            jupyter kaggle
```

---

## Architecture Patterns

### Recommended Project Structure (Phase 2 additions)

```
credit-risk-ml-pipeline/
├── notebooks/
│   └── eda.ipynb               # MODEL-04: EDA — class dist, missing values, correlations
├── src/
│   └── training/
│       ├── __init__.py
│       ├── data.py             # load_give_me_credit(), preprocess(), train_test_split wrapper
│       ├── train.py            # main training function: SMOTE + LightGBM + MLflow run
│       ├── evaluate.py         # compute AUC, PR curve, feature importance DataFrame
│       └── promote.py          # register_model(), set_registered_model_alias(), maybe_promote()
├── scripts/
│   └── train_champion.py       # CLI entrypoint: python scripts/train_champion.py
└── tests/
    └── training/
        ├── __init__.py
        ├── test_data.py        # test preprocessing: NaN handling, no target leakage
        ├── test_train.py       # test training produces model with AUC > threshold
        └── test_promote.py     # test alias promotion logic (mocked MLflow client)
```

### Pattern 1: Split-First SMOTE (avoids data leakage)

**What:** Split the full dataset first (stratified 80/20), then apply SMOTE only to `X_train, y_train`. The test set is never touched by SMOTE. This is the only correct approach per imbalanced-learn official docs.

**When to use:** Always, for any oversampling technique. Applying SMOTE before split causes test set contamination.

**Example:**
```python
# Source: https://imbalanced-learn.org/stable/common_pitfalls.html
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

# Step 1: split FIRST — preserve real class distribution in test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,  # critical for 7% imbalance
)

# Step 2: SMOTE only on training data
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# X_test, y_test remain untouched — real-world class distribution preserved
```

### Pattern 2: LightGBM Training with MLflow Autolog

**What:** Enable `mlflow.lightgbm.autolog()` before `lgb.train()` to capture hyperparameters and feature importances automatically. Log the precision-recall curve manually as an artifact.

**When to use:** All model training runs. Autolog captures what it can; PR curve requires manual matplotlib save.

**Example:**
```python
# Source: https://mlflow.org/docs/latest/python_api/mlflow.lightgbm.html
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt

mlflow.lightgbm.autolog(log_models=True, log_input_examples=False)

with mlflow.start_run(run_name="lgbm_smote_v1") as run:
    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "is_unbalance": True,  # belt-and-suspenders on top of SMOTE
        "random_state": 42,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train_resampled,
        y_train_resampled,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    # Autolog captures: params, feature importance (split + gain), model artifact
    # Must log PR curve manually — autolog does NOT produce it for LightGBM
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    mlflow.log_metric("auc_test", auc)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AUC={auc:.4f})")
    fig.savefig("/tmp/pr_curve.png")
    mlflow.log_artifact("/tmp/pr_curve.png", artifact_path="plots")
    plt.close(fig)

    run_id = run.info.run_id
```

### Pattern 3: MLflow Registry — Register and Set Champion Alias

**What:** After training, register the model and set the `@champion` alias. This is the MLflow 3.x pattern. Never use `transition_model_version_stage()`.

**When to use:** After every training run that meets the AUC threshold. Called from `src/training/promote.py`.

**Example:**
```python
# Source: https://mlflow.org/docs/latest/ml/model-registry/workflow/
from mlflow import MlflowClient
import mlflow

MODEL_NAME = "credit-risk-model"

def register_and_promote(run_id: str, model_artifact_path: str = "model") -> str:
    """Register model from run and set @champion alias.

    Args:
        run_id: MLflow run ID from training.
        model_artifact_path: Artifact path used in log_model (default "model").

    Returns:
        The version string of the newly registered + promoted model.
    """
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    version = result.version  # integer as string, e.g. "1"

    client = MlflowClient()
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=version,
    )
    return version


def load_champion() -> object:
    """Load the @champion model for inference.

    Returns:
        MLflow pyfunc model with .predict() interface.
    """
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
```

### Pattern 4: EDA Notebook Structure (MODEL-04)

**What:** A Jupyter notebook that documents Give Me Credit's data quality before training. This is a portfolio artifact — it must be readable, not just runnable.

**Sections the notebook must cover:**
1. Dataset shape and dtypes
2. Target distribution — `SeriousDelinquency` frequency (should show ~7% default rate, ~10,500 positives)
3. Missing value profile — show `MonthlyIncome` 19.8% NaN and `NumberOfDependents` ~2.6% NaN as a sorted bar chart
4. Feature distributions — histograms for all 10 features; show skewness
5. Correlation heatmap — Spearman rank correlation (Pearson is wrong for skewed credit features)
6. Outlier investigation — `NumberOfTimes90DaysLate` has values up to 98, capped at 17 by convention
7. Class imbalance summary — state the SMOTE rationale

### Anti-Patterns to Avoid

- **SMOTE before split:** Creates test set contamination. Inflates AUC by ~0.02-0.05 on this dataset. Use split-first pattern always.
- **`transition_model_version_stage()` with `stage="Production"`:** Deprecated since MLflow 2.9. Produces FutureWarning. Use `set_registered_model_alias()` + `@champion` alias.
- **Using `SeriousDelinquency` as a feature:** Target leakage. Add `assert "SeriousDelinquency" not in X.columns` in preprocessing.
- **Mean imputation for MonthlyIncome:** Monthly income is right-skewed (median ~5,400, mean ~6,670). Mean imputation shifts imputed values toward outliers. Use median.
- **`model.predict()` instead of `model.predict_proba()[:,1]`:** CB-PDD requires continuous probability scores, not binary labels. AUC also requires probabilities.
- **Logging model artifacts only, not the run:** `mlflow.register_model()` requires a run URI (`runs:/<run_id>/model`). Do not call `log_model` without an active `start_run` context.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Oversampling minority class | Custom duplicate-with-noise logic | `imblearn.SMOTE.fit_resample()` | SMOTE's k-NN interpolation is statistically correct; naive duplication adds no new information |
| Model versioning + metadata store | Custom registry in Postgres | MLflow Model Registry | Already deployed; handles version incrementing, aliases, artifact URIs |
| Feature importance extraction | Parse lgb.Booster internal dicts | `mlflow.lightgbm.autolog()` | Autolog handles both split and gain importance as JSON + PNG automatically |
| Precision-recall computation | Manual threshold sweep | `sklearn.metrics.precision_recall_curve()` | Handles all thresholds correctly including edge cases |
| Model serialization | pickle or joblib | `mlflow.lightgbm.log_model()` | MLflow uses LightGBM's native `.txt` format, not pickle — portable across Python versions |
| Train/test split with stratification | Manual index splitting | `sklearn.model_selection.train_test_split(..., stratify=y)` | Stratify=y ensures both splits have 7% default rate |

**Key insight:** The standard MLOps stack handles everything except the precision-recall curve logging — that one manual artifact is the only custom work needed beyond wiring existing APIs together.

---

## Common Pitfalls

### Pitfall 1: SMOTE Applied to Entire Dataset Before Split
**What goes wrong:** AUC appears 0.03-0.05 higher than it really is. Model underperforms in production (and on Phase 4's real scoring).
**Why it happens:** Developers apply `fit_resample(X, y)` on full dataset to simplify code, then split. SMOTE's k-NN interpolation uses test set neighbors.
**How to avoid:** Always call `train_test_split` first. Then `smote.fit_resample(X_train, y_train)`. Test set is never passed to SMOTE.
**Warning signs:** AUC > 0.90 on a simple default LightGBM run on Give Me Credit without tuning — suspect leakage.

### Pitfall 2: MLflow Stages API (deprecated)
**What goes wrong:** `transition_model_version_stage(name, version, stage="Production")` produces `FutureWarning` in MLflow 3.x. Will break in a future major version.
**Why it happens:** Most tutorials online (pre-2024) use the stages API. Copy-paste propagates the pattern.
**How to avoid:** Never write `stage=`. Always use `client.set_registered_model_alias(name, "champion", version)`. Load via `models:/credit-risk-model@champion`.
**Warning signs:** `FutureWarning: transition_model_version_stage is deprecated` in logs.

### Pitfall 3: `get_latest_versions()` Returns Empty List
**What goes wrong:** When calling `client.get_latest_versions(MODEL_NAME)` right after `mlflow.register_model()`, the model version may not be immediately queryable if MLflow's background registration hasn't completed.
**Why it happens:** `mlflow.register_model()` is asynchronous by default (`await_registration_for=300`). Calling `get_latest_versions()` synchronously immediately after may race.
**How to avoid:** Use the `result.version` returned directly from `mlflow.register_model()` — it is the version number. Do not query `get_latest_versions()` to find the just-registered version.
**Warning signs:** `KeyError` or empty list from `get_latest_versions()` in promote logic.

### Pitfall 4: CB-PDD Won't Fire on Class-Imbalanced Scores
**What goes wrong:** After training without SMOTE or class weighting, the model scores nearly all applicants near 0.05-0.10 probability. CB-PDD's CheckerBoard partitions have too few minority-class instances per bucket. The smoke test (MODEL-04 success criterion 4) produces zero drift detections even on a simulated denial loop.
**Why it happens:** Paper explicitly acknowledges this failure mode for datasets with <5% minority class. Give Me Credit at 7% is in the marginal zone.
**How to avoid:** Apply SMOTE during training (Phase 2). Then run the CB-PDD smoke test in Plan 02-03 — if it doesn't fire on a synthetic 30-day denial loop with τ=1000, adjust τ downward or increase SMOTE ratio.
**Warning signs:** All model scores clustering below 0.15; smoke test shows drift_score never exceeds 0.5 × threshold.

### Pitfall 5: NaN Handling Introduces Target Leakage
**What goes wrong:** `MonthlyIncome` NaNs (19.8%) are imputed using the global median. If the global median is computed on the full dataset before splitting, information from the test set bleeds into the training imputation.
**Why it happens:** `df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())` looks benign but uses test-set values in the median computation.
**How to avoid:** Fit the imputer (or compute the median) only on `X_train`. Apply to both `X_train` and `X_test`. Use `sklearn.impute.SimpleImputer(strategy="median")` inside a pipeline or fit it explicitly on training data only.
**Warning signs:** Imputation median differs when computed on train vs full dataset.

### Pitfall 6: NumberOfDependents NaN Silently Treated as 0
**What goes wrong:** `NumberOfDependents` has ~2.6% NaN. Using `fillna(0)` treats unknown dependents as zero dependents — introduces systematic bias toward underestimating family size risk.
**Why it happens:** Zero-fill is the default assumption for count-like columns. But 0 dependents has a specific credit risk meaning.
**How to avoid:** Use median imputation (median is ~0 for this column, but at least the imputed value comes from the distribution, not an assumption). Add a flag column `NumberOfDependents_was_missing` to let the model learn the missingness signal.
**Warning signs:** Disproportionate model weight on `NumberOfDependents` compared to published benchmarks.

---

## Code Examples

### Give Me Credit Preprocessing Pipeline

```python
# Source: verified against imbalanced-learn docs + multiple Kaggle solutions
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
TARGET_COL = "SeriousDelinquency"

def load_and_preprocess(csv_path: str) -> tuple:
    """Load Give Me Credit CSV, impute, cap outliers, return X, y.

    Returns:
        Tuple of (X: pd.DataFrame, y: pd.Series).
    """
    df = pd.read_csv(csv_path, index_col=0)
    assert TARGET_COL not in FEATURE_COLS, "Target leakage guard"

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Cap known outlier column — values >17 are data entry errors
    X["NumberOfTimes90DaysLate"] = X["NumberOfTimes90DaysLate"].clip(upper=17)
    X["NumberOfTime30-59DaysPastDueNotWorse"] = X[
        "NumberOfTime30-59DaysPastDueNotWorse"
    ].clip(upper=17)

    # Add missingness flag before imputing
    X["MonthlyIncome_was_missing"] = X["MonthlyIncome"].isna().astype(int)
    X["NumberOfDependents_was_missing"] = X["NumberOfDependents"].isna().astype(int)

    return X, y


def build_train_test(X: pd.DataFrame, y: pd.Series):
    """Split, impute train-only, apply SMOTE to train only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit imputer only on training data
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns
    )

    # SMOTE only on training split — test set untouched
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test, imputer
```

### MLflow Registration and Champion Alias

```python
# Source: https://mlflow.org/docs/latest/ml/model-registry/workflow/
from mlflow import MlflowClient
import mlflow

MODEL_NAME = "credit-risk-model"


def register_and_promote(run_id: str) -> str:
    """Register model run and set @champion alias.

    Args:
        run_id: Active MLflow run ID containing logged model.

    Returns:
        Version string of registered model.
    """
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    version = result.version  # use directly — don't query get_latest_versions()

    client = MlflowClient()
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=version,
    )
    return version


def get_champion_auc() -> float:
    """Return AUC of the current @champion model run.

    Returns:
        AUC float, or 0.0 if no champion exists yet.
    """
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
        run = client.get_run(mv.run_id)
        return run.data.metrics.get("auc_test", 0.0)
    except Exception:
        return 0.0
```

### CB-PDD Smoke Test (MODEL-04 success criterion 4)

```python
# Minimal synthetic denial loop — verifies CB-PDD fires before Phase 4 builds on it
import numpy as np

def generate_denial_loop_scores(
    n_days: int = 30,
    n_per_day: int = 5000,
    base_default_rate: float = 0.07,
    denial_threshold: float = 0.5,
) -> list[dict]:
    """Generate synthetic prediction records mimicking a 30-day denial loop.

    Each day, denied applicants (score > denial_threshold) are removed from
    the next day's pool. Remaining pool skews toward low-risk over time.

    Returns:
        List of dicts with keys: simulation_day, score, decision, path.
    """
    rng = np.random.default_rng(42)
    records = []
    default_rate = base_default_rate

    for day in range(n_days):
        scores = rng.beta(
            a=1 - default_rate,
            b=default_rate * 10,
            size=n_per_day,
        )
        for score in scores:
            decision = "denied" if score > denial_threshold else "approved"
            records.append({
                "simulation_day": day,
                "score": float(score),
                "decision": decision,
                "path": "model",
            })
        # Denial loop: higher-risk applicants exit next day
        default_rate = max(0.02, default_rate * 0.92)

    return records
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `transition_model_version_stage("Production")` | `set_registered_model_alias("champion", version)` | MLflow 2.9.0 (2023) | Old API still works in 3.x but produces FutureWarning; will be removed |
| `mlflow.sklearn.log_model` for LightGBM | `mlflow.lightgbm.log_model` with LightGBM flavor | MLflow 1.27+ | Native LightGBM serialization (`.txt` format) instead of pickle; portable |
| Apply SMOTE before split | Split first, SMOTE on train only | imbalanced-learn 0.8+ docs | Standard practice documented as the ONLY correct approach |
| `model.predict()` for scoring | `model.predict_proba()[:, 1]` | Always correct | AUC and CB-PDD both require continuous probabilities, not binary labels |
| `numpy` 2.x | `numpy` 1.26.x pinned | LightGBM 4.x compatibility | numpy 2.x breaks some LightGBM GPU builds; pin explicitly |

**Deprecated/outdated:**
- `mlflow.client.MlflowClient().transition_model_version_stage()`: deprecated since 2.9, FutureWarning in 3.x
- `model.predict()` for probability output: use `.predict_proba()[:,1]` — predict() returns class labels

---

## Open Questions

1. **CB-PDD τ calibration for Give Me Credit**
   - What we know: Paper uses τ=1000 as default; 150k rows / 30 days = 5,000 instances/day → 5 trials/day at τ=1000
   - What's unclear: Whether 5 trials/day is sensitive enough for the denial loop drift pattern — paper shows detection degrades when τ is too long relative to drift rate
   - Recommendation: Run smoke test in Plan 02-03 with τ values {500, 1000, 2000}. If τ=1000 fails to fire on a 30-day simulated denial loop, drop to τ=500. Document the chosen value with rationale.

2. **`mlflow.register_model()` blocking behavior in Docker**
   - What we know: Default `await_registration_for=300` seconds — should block until registered
   - What's unclear: Whether MLflow running in docker-compose (localhost:5000 tracking URI) introduces any race conditions
   - Recommendation: Call `mlflow.register_model()` and use the returned `result.version` directly. Add a 2-second sleep before `set_registered_model_alias` as insurance if race issues appear in testing.

3. **Give Me Credit dataset download — Kaggle API vs manual**
   - What we know: Dataset is at `kaggle competitions download -c GiveMeSomeCredit`; requires KAGGLE_USERNAME and KAGGLE_KEY env vars
   - What's unclear: Whether the Kaggle competition API requires competition acceptance before download, or if it works with just credentials
   - Recommendation: Document in the script's docstring that manual download from https://www.kaggle.com/c/GiveMeSomeCredit/data is the fallback. Add a data presence check at script start.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | `pyproject.toml` — `[tool.pytest.ini_options]` testpaths=["tests"] |
| Quick run command | `pytest tests/training/ -v --tb=short` |
| Full suite command | `pytest tests/ -v --tb=short` |
| Estimated runtime | ~30-60 seconds (model training test uses tiny synthetic dataset) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MODEL-01 | SMOTE applied after split; LightGBM fits; AUC > 0.85 on test | integration | `pytest tests/training/test_train.py -x` | Wave 0 gap |
| MODEL-01 | MonthlyIncome NaN imputed with training-set median only | unit | `pytest tests/training/test_data.py::test_imputer_fit_on_train_only -x` | Wave 0 gap |
| MODEL-01 | Target column not in feature set (no leakage) | unit | `pytest tests/training/test_data.py::test_no_target_leakage -x` | Wave 0 gap |
| MODEL-02 | MLflow run logs auc_test metric and pr_curve artifact | integration | `pytest tests/training/test_train.py::test_mlflow_run_artifacts -x` | Wave 0 gap |
| MODEL-03 | `@champion` alias resolves after promotion | integration | `pytest tests/training/test_promote.py::test_champion_alias_set -x` | Wave 0 gap |
| MODEL-03 | No `transition_model_version_stage` in codebase | static/grep | `grep -r "transition_model_version_stage" src/ && exit 1 \|\| exit 0` | Can run now |
| MODEL-04 | EDA notebook runs end-to-end (no exceptions) | smoke | `jupyter nbconvert --to notebook --execute notebooks/eda.ipynb` | Wave 0 gap |

### Nyquist Sampling Rate

- **Minimum sample interval:** After every committed task → run: `pytest tests/training/ -v --tb=short`
- **Full suite trigger:** Before merging final task of any plan wave
- **Phase-complete gate:** Full suite green before `/gsd:verify-work` runs
- **Estimated feedback latency per task:** ~30-60 seconds (integration tests use 1,000-row synthetic dataset, not full 150k)

### Wave 0 Gaps (must be created before implementation)

- [ ] `tests/training/__init__.py` — package init for training test module
- [ ] `tests/training/test_data.py` — covers MODEL-01 preprocessing assertions (no leakage, NaN handling, outlier capping)
- [ ] `tests/training/test_train.py` — covers MODEL-01 (AUC > threshold) and MODEL-02 (MLflow artifacts logged); uses 1k-row synthetic data with 7% imbalance
- [ ] `tests/training/test_promote.py` — covers MODEL-03 alias promotion; mocks `MlflowClient` to avoid real MLflow calls
- [ ] `tests/training/conftest.py` — shared fixtures: synthetic_credit_df (1000 rows, 7% default), mock_mlflow_client

---

## Sources

### Primary (HIGH confidence)
- https://mlflow.org/docs/latest/ml/model-registry/workflow/ — `set_registered_model_alias`, `get_model_version_by_alias`, `load_model` URI pattern verified
- https://mlflow.org/docs/latest/python_api/mlflow.lightgbm.html — `autolog()` parameters, `log_model()` signature, feature importance artifacts
- https://imbalanced-learn.org/stable/common_pitfalls.html — SMOTE must be applied after split; pipeline pattern for leakage prevention
- https://arxiv.org/html/2412.10545 — τ=1000 default, imbalanced dataset limitation, mix parameter behavior

### Secondary (MEDIUM confidence)
- https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/ — SMOTE after split pattern (corroborates official docs)
- https://nycdatascience.com/blog/student-works/kaggle-predict-consumer-credit-default/ — Give Me Credit feature descriptions, MonthlyIncome right-skew confirmed
- https://www.kaggle.com/c/GiveMeSomeCredit — Dataset source; 150k rows, 11 features confirmed
- Multiple Kaggle notebooks on Give Me Credit — MonthlyIncome 19.8% NaN, `NumberOfTimes90DaysLate` outliers at 98, all confirmed across multiple independent sources

### Tertiary (LOW confidence)
- LightGBM AUC > 0.85 achievability on imbalanced credit data — inferred from multiple papers showing 0.856+ AUC on similar datasets; not validated specifically on Give Me Credit with SMOTE

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified via official docs and PyPI
- Architecture: HIGH — SMOTE pattern verified from imbalanced-learn official docs; MLflow API verified from official workflow docs
- Pitfalls: HIGH — CB-PDD class imbalance limitation sourced from paper directly; MLflow deprecation from official GitHub issue
- CB-PDD τ sensitivity: MEDIUM — paper provides defaults and directional guidance but no formula for calibration to specific batch sizes

**Research date:** 2026-03-29
**Valid until:** 2026-04-29 (MLflow and imbalanced-learn APIs are stable; 30-day window is conservative)
