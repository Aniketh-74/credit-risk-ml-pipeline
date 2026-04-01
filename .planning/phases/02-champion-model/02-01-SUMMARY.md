---
phase: 02-champion-model
plan: "01"
subsystem: data-analysis
tags: [jupyter, pandas, seaborn, matplotlib, scipy, eda, smote, class-imbalance, credit-risk]

# Dependency graph
requires:
  - phase: 01-solid-ground
    provides: project scaffold, Docker infra, PostgreSQL schema, FastAPI prediction endpoint
provides:
  - notebooks/eda.ipynb — documented data quality analysis justifying all preprocessing decisions
  - clip(upper=17) decision for NumberOfTimes90DaysLate and NumberOfTime30-59DaysPastDueNotWorse
  - median imputation + missingness flag decision for MonthlyIncome
  - SMOTE post-split rationale documented
affects: [02-champion-model plan 02 (training pipeline), src/training/data.py preprocessing logic]

# Tech tracking
tech-stack:
  added: [jupyter, seaborn==0.13.2, scipy (1.17.1 installed), matplotlib (3.10.8 installed)]
  patterns: [EDA-before-training — all preprocessing decisions trace to documented notebook findings]

key-files:
  created:
    - notebooks/eda.ipynb
  modified:
    - pyproject.toml (added [project.optional-dependencies] dev group)

key-decisions:
  - "clip(upper=17) for NumberOfTimes90DaysLate and NumberOfTime30-59DaysPastDueNotWorse — values of 96/98 are sentinel codes, not real delinquency counts; 17 is the plausible quarterly-cycle max"
  - "Median imputation for MonthlyIncome (19.8% NaN) — right-skewed distribution makes mean unrepresentative; fitted on train split only to prevent leakage"
  - "MonthlyIncome_was_missing binary flag to be added — missingness may correlate with income type (self-employed, unemployed) and carries predictive signal"
  - "SMOTE applied after train/test split only — pre-split SMOTE allows synthetic test-set samples into training data, inflating eval metrics"
  - "Spearman correlation chosen over Pearson — credit features are heavily right-skewed and Pearson's linearity + outlier assumptions are both violated"
  - "Retain all three late-payment count features despite ~0.98 pairwise correlation — LightGBM handles multicollinearity and each feature captures different timing signal"

patterns-established:
  - "EDA-first: all preprocessing choices in training code must have a corresponding documented finding in notebooks/eda.ipynb"

requirements-completed: [MODEL-04]

# Metrics
duration: 20min
completed: 2026-04-01
---

# Phase 02 Plan 01: EDA Notebook Summary

**7-section Jupyter EDA of Give Me Credit confirming 6.68% default rate, 19.8% MonthlyIncome NaN, and 96/98 sentinel outliers — documenting clip, median imputation, and SMOTE rationale before any training code is written**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-04-01T00:00:00Z
- **Completed:** 2026-04-01T00:20:00Z
- **Tasks:** 1 (Task 0 pre-completed — dataset already present)
- **Files modified:** 2

## Accomplishments

- Built a 7-section portfolio-grade EDA notebook that executes end-to-end via `jupyter nbconvert --execute` without exceptions
- Confirmed dataset statistics: 150,000 rows, 6.68% default rate (~10,020 positives), MonthlyIncome 19.8% NaN, NumberOfDependents 2.6% NaN
- Documented outlier finding: NumberOfTimes90DaysLate contains values of 96 and 98 (~220 rows each), established `clip(upper=17)` as the preprocessing decision
- Spearman correlation heatmap shows the three late-payment features (30-59, 60-89, 90DaysLate) are highly correlated (~0.98), consistent with expected borrower behavior
- Final section ties all three challenges (imbalance, missing values, outliers) to their training pipeline solutions with explicit ordering constraint (SMOTE after split)

## Task Commits

1. **Task 1: Build EDA notebook** - `08f2175` (feat)

## Files Created/Modified

- `notebooks/eda.ipynb` — 7-section EDA notebook; loads cs-training.csv, generates all plots and markdown analysis
- `pyproject.toml` — added `[project.optional-dependencies] dev` group with jupyter, seaborn, scipy, matplotlib

## Decisions Made

- **clip(upper=17):** Values of 96 and 98 in NumberOfTimes90DaysLate are almost certainly sentinel codes. Cap at 17 matches multiple top Kaggle solutions and is consistent with quarterly credit reporting cycles.
- **Median imputation for MonthlyIncome:** Right-skewed income distribution makes mean unrepresentative. Imputation fitted on training fold only.
- **MonthlyIncome_was_missing flag:** Missingness is non-random (likely correlated with employment type) — preserving it as a binary feature avoids information loss.
- **Spearman not Pearson:** Credit features heavily violate Pearson's linearity and normality assumptions. Spearman rank correlation is robust to skewed distributions and outliers.
- **SMOTE after split:** Applying SMOTE before splitting allows synthetic samples derived from test-set minority instances to leak into training data.

## Deviations from Plan

None — plan executed exactly as written. Notebook structure, section content, and verification command all matched specification. Installed packages were already present in the environment (seaborn 0.13.2, scipy 1.17.1, matplotlib 3.10.8, jupyter/nbconvert).

## Issues Encountered

None. The notebook executed cleanly on first run. The scipy version in the environment (1.17.1) is newer than the pinned requirement (1.15.2) — this is compatible and no downgrade was needed. The dev dependency in pyproject.toml retains the plan-specified pin for reproducibility.

## User Setup Required

None — notebook reads from `data/raw/cs-training.csv` (already present, gitignored). No external services or environment variables needed.

## Next Phase Readiness

- `notebooks/eda.ipynb` provides documented justification for all preprocessing decisions needed in Plan 02-02 (training pipeline)
- Key preprocessing decisions to implement in `src/training/data.py`:
  - `clip(upper=17)` on NumberOfTimes90DaysLate and NumberOfTime30-59DaysPastDueNotWorse
  - Median imputation for MonthlyIncome (train-fit only) + MonthlyIncome_was_missing flag
  - Median imputation for NumberOfDependents
  - SMOTE applied after train/test split inside the pipeline

---
*Phase: 02-champion-model*
*Completed: 2026-04-01*
