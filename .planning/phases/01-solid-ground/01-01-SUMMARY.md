---
phase: 01-solid-ground
plan: "01"
subsystem: database
tags: [sqlalchemy, alembic, postgresql, migrations, schema]

# Dependency graph
requires: []
provides:
  - SQLAlchemy 2.0 ORM models for Prediction, Outcome, DriftScore, Alert
  - Alembic migration 0001 creating all four tables with full column spec
  - PostgreSQL multi-DB init script for airflow_db, mlflow_db, app_db
  - Pinned infra requirements (alembic, sqlalchemy, asyncpg, psycopg2-binary, python-dotenv)
affects:
  - 01-02 (Docker Compose mounts init-multiple-dbs.sh as postgres initdb script)
  - 01-03 (smoke test runs alembic upgrade head against live app_db)
  - phase-03 (CB-PDD router reads predictions.path to distinguish model vs checkerboard)
  - phase-04 (SIM-04 reads outcomes.outcome_received_at for label delay simulation)
  - phase-04 (DRIFT-05 reads drift_scores.psi_score for PSI baseline comparison)

# Tech tracking
tech-stack:
  added:
    - sqlalchemy==2.0.48
    - alembic==1.14.1
    - asyncpg==0.30.0
    - psycopg2-binary==2.9.9
    - python-dotenv==1.0.1
  patterns:
    - SQLAlchemy 2.0 DeclarativeBase + mapped_column (no legacy Column() or declarative_base())
    - Alembic env.py reads APP_DB_URL from environment — no hardcoded connection strings
    - asyncpg:// URL auto-rewritten to psycopg2:// for synchronous Alembic migrations
    - Google-style docstrings on all ORM model classes

key-files:
  created:
    - db/models.py
    - db/__init__.py
    - db/migrations/alembic.ini
    - db/migrations/env.py
    - db/migrations/script.py.mako
    - db/migrations/versions/0001_initial_schema.py
    - db/migrations/versions/__init__.py
    - scripts/init-multiple-dbs.sh
    - requirements-infra.txt
  modified: []

key-decisions:
  - "sqlalchemy bumped from 2.0.39 to 2.0.48: Python 3.14 has a regression in typing.Union.__getitem__ that breaks SQLAlchemy 2.0.39 Mapped type resolution — 2.0.48 is the first release with Python 3.14 support"
  - "predictions.path VARCHAR(20) NOT NULL with CHECK IN ('model','checkerboard') — present from migration 0001 because CB-PDD requires per-row routing label and retroactive addition would leave NULLs in existing rows"
  - "outcomes.outcome_received_at as a separate TIMESTAMPTZ column from predicted_at — captures real label delay, not just prediction time; Phase 4 SIM-04 reads this gap"
  - "drift_scores.psi_score nullable — PSI requires a reference distribution baseline which is not available during warm-up; nullable allows rows before baseline is established"

patterns-established:
  - "Schema contract pattern: column names in models.py match migration column names exactly — no renames without a migration"
  - "Environment-driven config: all DB URLs read from APP_DB_URL env var — nothing hardcoded in committed files"
  - "Alembic URL rewriting: asyncpg:// driver prefix swapped to psycopg2:// in env.py get_url() for migration compatibility"

requirements-completed: [INFRA-01]

# Metrics
duration: 5min
completed: 2026-03-28
---

# Phase 01 Plan 01: SQLAlchemy Schema and Alembic Migration Summary

**SQLAlchemy 2.0 ORM schema contract for four PostgreSQL tables (predictions, outcomes, drift_scores, alerts) with Alembic migration 0001 and multi-DB PostgreSQL init script**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-28T15:38:26Z
- **Completed:** 2026-03-28T15:43:10Z
- **Tasks:** 2
- **Files modified:** 9 created, 0 modified

## Accomplishments

- Four SQLAlchemy 2.0 ORM models using `DeclarativeBase` and `mapped_column` — zero legacy `Column()` or `declarative_base()` patterns
- `predictions.path` VARCHAR(20) NOT NULL with CHECK constraint for `'model'|'checkerboard'` — required from day one by Phase 3 CB-PDD router
- `outcomes.outcome_received_at` TIMESTAMPTZ NOT NULL as a separate column from `predicted_at` — captures the label delay that Phase 4 SIM-04 reads
- Alembic migration 0001 creates all four tables with matching column specs and drops them cleanly in reverse FK order on downgrade
- `scripts/init-multiple-dbs.sh` creates `airflow_db`, `mlflow_db`, `app_db` with `GRANT ALL ON SCHEMA public` for PostgreSQL 15+ compatibility

## Task Commits

Each task was committed atomically:

1. **Task 1: SQLAlchemy 2.0 declarative models and requirements file** - `9e9c08e` (feat)
2. **Task 2: Alembic setup, first migration, and multi-DB init script** - `23c8232` (feat)

## Files Created/Modified

- `db/__init__.py` - Empty package marker for db module
- `db/models.py` - Four ORM models (Prediction, Outcome, DriftScore, Alert) using SQLAlchemy 2.0 style
- `db/migrations/alembic.ini` - Alembic config; URL set via env.py override, not hardcoded
- `db/migrations/env.py` - Imports Base.metadata; asyncpg→psycopg2 URL rewrite; online + offline mode support
- `db/migrations/script.py.mako` - Standard Alembic migration template
- `db/migrations/versions/__init__.py` - Empty package marker for versions directory
- `db/migrations/versions/0001_initial_schema.py` - Creates all four tables; revision ID `0001`
- `scripts/init-multiple-dbs.sh` - Creates airflow_db, mlflow_db, app_db with correct PostgreSQL 15+ grants
- `requirements-infra.txt` - Pinned: alembic 1.14.1, sqlalchemy 2.0.48, asyncpg 0.30.0, psycopg2-binary 2.9.9, python-dotenv 1.0.1

## Decisions Made

- **sqlalchemy 2.0.48 instead of 2.0.39:** The dev machine runs Python 3.14.2. SQLAlchemy 2.0.39 has a `TypeError` in `make_union_type()` when resolving `Optional[T]` in `Mapped` annotations on Python 3.14 — `typing.Union.__getitem__` signature changed. Version 2.0.48 is the first SQLAlchemy release with Python 3.14 support. Updated `requirements-infra.txt` accordingly.
- **`path` column from day one:** CB-PDD router in Phase 3 requires per-row routing labels. Adding this column retroactively after rows exist would leave NULLs in historical predictions, breaking the algorithm. The column must exist before any prediction is written.
- **`outcome_received_at` as separate column:** Kept distinct from `predicted_at` to preserve the label delay — the time between making a prediction and observing the actual outcome. Phase 4 reads this gap to simulate realistic feedback lag.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Upgraded sqlalchemy from 2.0.39 to 2.0.48**
- **Found during:** Task 1 (model verification)
- **Issue:** `python -c "from db.models import ..."` raised `TypeError: descriptor '__getitem__' requires a 'typing.Union' object but received a 'tuple'` — SQLAlchemy 2.0.39 incompatible with Python 3.14's changed `typing.Union` internals
- **Fix:** Upgraded to `sqlalchemy==2.0.48` (first release with Python 3.14 support); updated `requirements-infra.txt`
- **Files modified:** `requirements-infra.txt`
- **Verification:** `from db.models import Base, Prediction, Outcome, DriftScore, Alert` imports cleanly; all four table names printed
- **Committed in:** `9e9c08e` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking — version incompatibility)
**Impact on plan:** Required for correctness on Python 3.14. The pinned version is still within SQLAlchemy 2.0.x series — same API, same migration behavior.

## Issues Encountered

- Python 3.14 compatibility: `from __future__ import annotations` combined with SQLAlchemy 2.0.39's Union type introspection caused a `TypeError`. Removing the future import and switching to explicit `Optional[T]` syntax partially helped, but the root issue was SQLAlchemy 2.0.39 itself. Upgrading to 2.0.48 resolved it cleanly.

## User Setup Required

None — no external service configuration required for this plan. The init script and migration will be exercised in Plan 03 (smoke test with a live Docker Postgres).

## Next Phase Readiness

- Schema contract is locked — all four tables defined, migration ready to run
- `scripts/init-multiple-dbs.sh` ready to mount in docker-compose.yml postgres service
- `db/migrations/env.py` reads `APP_DB_URL` — Plan 02 docker-compose.yml sets this variable for the app service
- Alembic `upgrade head` / `downgrade base` verified to be structurally correct (live DB test deferred to Plan 03 smoke test)

---
*Phase: 01-solid-ground*
*Completed: 2026-03-28*
