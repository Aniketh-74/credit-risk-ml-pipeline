---
phase: 04-proving-the-feedback-loop
plan: "01"
subsystem: simulation
tags: [sqlalchemy, httpx, sqlite, postgresql, pytest, simulation, drift]

# Dependency graph
requires:
  - phase: 03-the-scoring-api
    provides: /score endpoint, Prediction and Outcome ORM models, predictions/outcomes tables
  - phase: 01-solid-ground
    provides: PostgreSQL schema with simulation_day column on predictions table
provides:
  - DenialLoopSimulator: calls /score API per applicant, nudges denied applicants back into pool
  - ScoreGamingSimulator: direct-to-DB writes with injectable score_fn and daily feature ramp
  - Smoke test suite in tests/simulators/ covering simulation_day, label delay, dual-table writes
affects: [05-airflow-orchestration, 04-02, 04-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Injectable score_fn for ScoreGamingSimulator — test-time mock, Phase 5 real model
    - Bulk insert via SQLAlchemy insert() per simulation day — avoids ORM overhead on 1000-row batches
    - Explicit UUID generation in Python (str(uuid.uuid4())) — avoids gen_random_uuid() PostgreSQL-only dependency in SQLite tests

key-files:
  created:
    - src/simulators/__init__.py
    - src/simulators/denial_loop.py
    - src/simulators/score_gaming.py
    - tests/simulators/__init__.py
    - tests/simulators/test_simulators.py
  modified: []

key-decisions:
  - "Explicit predicted_at written to Outcome rows — Outcome.predicted_at is NOT NULL with no server default; omitting it causes IntegrityError on both SQLite and PostgreSQL"
  - "UUID generated in Python for all rows — gen_random_uuid() is PostgreSQL-only; Python uuid4() works for both engines"
  - "ScoreGamingSimulator uses injectable score_fn — enables test isolation without API or real model, Phase 5 will inject champion model callable"
  - "DenialLoopSimulator mocks httpx.Client in tests — avoids dependency on live /score API while exercising full DB write path"

patterns-established:
  - "Simulator pattern: inject score_fn for testability, patch create_engine to redirect writes to SQLite"
  - "Label delay: outcome_received_at = predicted_at + random(1-7) days, written atomically with prediction row"

requirements-completed: []

# Metrics
duration: 3min
completed: 2026-04-05
---

# Phase 4 Plan 01: Denial Loop and Score Gaming Simulators Summary

**Two feedback loop simulators writing labeled prediction + outcome rows to PostgreSQL with simulation_day and outcome_received_at populated, tested against in-memory SQLite without a live API**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-05T19:46:04Z
- **Completed:** 2026-04-05T19:49:13Z
- **Tasks:** 3
- **Files modified:** 5 created

## Accomplishments
- DenialLoopSimulator calls the real /score API, nudges denied applicants toward approval each day, and bulk-writes prediction + outcome rows to PostgreSQL
- ScoreGamingSimulator writes directly to PostgreSQL using an injectable score function, creating a gradual feature ramp CB-PDD can detect
- 4 smoke tests pass using in-memory SQLite, mocked httpx.Client, and patched create_engine — no live API or database required

## Task Commits

Each task was committed atomically:

1. **Task 1: Denial Loop Simulator** - `16dc711` (feat)
2. **Task 2: Score Gaming Simulator** - `2d0518b` (feat)
3. **Task 3: Simulator smoke tests** - `2cfbaa6` (test)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `src/simulators/__init__.py` - Package marker
- `src/simulators/denial_loop.py` - DenialLoopSimulator with /score API calls, nudge logic, bulk DB writes
- `src/simulators/score_gaming.py` - ScoreGamingSimulator with injectable score_fn, daily feature nudge ramp
- `tests/simulators/__init__.py` - Package marker
- `tests/simulators/test_simulators.py` - 4 smoke tests: simulation_day population, label delay, dual-table writes

## Decisions Made

**Explicit predicted_at in Outcome rows:** The plan's code omitted `predicted_at` from `outcome_rows` dicts, but `Outcome.predicted_at` is `nullable=False` with no server default. This causes `IntegrityError` on both SQLite (for tests) and PostgreSQL. Fixed by copying `predicted_at.isoformat()` into every outcome row.

**Python UUID generation:** `Prediction.id` uses `server_default=text("gen_random_uuid()")` — PostgreSQL-only. SQLite tests fail with `no such function: gen_random_uuid` if `id` is omitted. Both simulators generate `id=str(uuid.uuid4())` explicitly, matching the plan's documented rationale.

**Injectable score_fn:** ScoreGamingSimulator accepts `score_fn=None` — default is a linear heuristic for standalone runs. Tests inject a deterministic mock that controls approval rates without touching the real model or API.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added missing predicted_at to Outcome rows in both simulators**
- **Found during:** Task 1 (reading Outcome model before implementation)
- **Issue:** Plan's outcome_rows dicts only had id, prediction_id, actual_default, outcome_received_at. Outcome.predicted_at is NOT NULL with no server default — inserting without it raises IntegrityError on SQLite (test engine) and PostgreSQL
- **Fix:** Added `"predicted_at": predicted_at.isoformat()` to outcome_rows in both denial_loop.py and score_gaming.py
- **Files modified:** src/simulators/denial_loop.py, src/simulators/score_gaming.py
- **Verification:** All 4 pytest tests pass including the JOIN query that reads both predicted_at and outcome_received_at
- **Committed in:** 16dc711 (Task 1), 2d0518b (Task 2)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Fix was required for any row insert to succeed. No scope creep.

## Issues Encountered
None beyond the auto-fixed Outcome.predicted_at issue above.

## User Setup Required
None - no external service configuration required. Tests run fully offline with SQLite.

## Next Phase Readiness
- Both simulators ready for Phase 5 Airflow task integration
- `run_denial_loop()` and `run_score_gaming()` accept `db_url` and `start_day` for DAG parameterization
- `score_fn` injection point in ScoreGamingSimulator ready for champion model callable from MLflow
- Tests confirm simulation_day is always non-null and outcome_received_at > predicted_at on every row

---
*Phase: 04-proving-the-feedback-loop*
*Completed: 2026-04-05*
