---
phase: 01-solid-ground
plan: "03"
subsystem: infra
tags: [docker, alembic, postgresql, env-config, pyproject, ruff, black, pytest]

# Dependency graph
requires:
  - phase: 01-01
    provides: SQLAlchemy models, Alembic migration 0001, init-multiple-dbs.sh
  - phase: 01-02
    provides: docker-compose.yml, all seven services, FastAPI health stub

provides:
  - ".env.example contract documenting all environment variables with generation instructions"
  - ".gitignore protecting secrets, data files, Python artifacts, and IDE files"
  - "pyproject.toml with ruff (google docstrings), black (88 chars), and pytest (testpaths=tests) config"
  - "tests/ directory with __init__.py for pytest discovery"
  - "[PENDING] Verified full stack boot — alembic upgrade head, health checks, MLflow PostgreSQL backend"

affects:
  - phase-02 (EDA pipeline inherits pyproject tooling config)
  - phase-03 (API scoring must pass /health check established here)
  - phase-04 (test suite uses tests/ directory created here)
  - all-phases (.env.example is the env var contract for the entire project)

# Tech tracking
tech-stack:
  added:
    - "ruff (linter, configured in pyproject.toml)"
    - "black (formatter, configured in pyproject.toml)"
    - "pytest (test runner, configured in pyproject.toml)"
  patterns:
    - ".env.example as living documentation — every var has a comment explaining source or generation command"
    - "Placeholder syntax: <your-value> for user-supplied secrets, <generate-see-above> for generated keys"
    - "pyproject.toml as single source of truth for tooling config — no separate .ruff.toml, setup.cfg, or pytest.ini"

key-files:
  created:
    - .env.example
    - .gitignore
    - pyproject.toml
    - tests/__init__.py
  modified: []

key-decisions:
  - "All secret placeholders use angle-bracket syntax (<your-secure-password>) so grep/diff reveals any accidental literal value"
  - "APP_DB_URL in .env.example uses asyncpg driver (for FastAPI) with a comment noting Alembic rewrites to psycopg2 automatically"
  - "pyproject.toml uses [tool.ruff] select list with E501 ignored — black handles line length, ruff handles everything else"

patterns-established:
  - ".env.example pattern: every var has inline comment; secrets have generation command above them"
  - "Google docstring convention locked in pyproject.toml — applies to all Python files in the project"

requirements-completed: [INFRA-04]

# Metrics
duration: 5min
completed: 2026-03-28
status: checkpoint-pending
---

# Phase 01 Plan 03: Environment Configuration and Stack Verification Summary

**Environment variable contract (.env.example), Python tooling config (pyproject.toml), and .gitignore established — stack verification checkpoint pending**

## Status: CHECKPOINT PENDING

Task 1 is complete. Task 2 (human verification of the live stack) is awaiting user confirmation.

## Performance

- **Duration:** ~5 min (Task 1 only — Task 2 deferred to checkpoint)
- **Started:** 2026-03-28T15:46:04Z
- **Completed (Task 1):** 2026-03-28T15:51:00Z
- **Tasks:** 1 of 2 complete
- **Files created:** 4

## Accomplishments

- `.env.example` documents every environment variable with inline comments; secrets use `<generate-see-above>` placeholders with exact generation commands
- `.gitignore` covers Python cache, virtual envs, data files (Give Me Credit CSVs), IDE artifacts, and Docker volumes — `.env` is protected
- `pyproject.toml` locks ruff + black + pytest config for the project; google-style docstrings enforced via `[tool.ruff.lint.pydocstyle]`
- `tests/__init__.py` creates the pytest-discoverable test package used from Phase 4 onward

## Task Commits

1. **Task 1: .env.example, .gitignore, pyproject.toml, and tests scaffold** - `ca49a92` (chore)
2. **Task 2: Stack verification** - PENDING (awaiting checkpoint approval)

## Files Created/Modified

- `.env.example` - Full environment variable documentation with placeholder markers for all secrets
- `.gitignore` - Comprehensive Python/Docker/data gitignore; `.env` excluded from git tracking
- `pyproject.toml` - Project metadata + ruff (py311, google docstrings, E501 ignored) + black (88 chars) + pytest (testpaths=tests)
- `tests/__init__.py` - Empty package marker for pytest discovery

## Decisions Made

- **Placeholder syntax:** Used `<your-secure-password>` for user-chosen values and `<generate-see-above>` for generated keys — makes accidental literal values detectable by grep
- **asyncpg in APP_DB_URL:** FastAPI uses asyncpg; comment in `.env.example` notes that Alembic env.py rewrites to psycopg2 automatically (established in Plan 01-01)
- **E501 ignored in ruff:** Black enforces line length at 88; having ruff also flag E501 creates duplicate warnings with no benefit

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected verification script assertion**
- **Found during:** Task 1 verification
- **Issue:** The plan's verification script asserted `'.env' not in first_line` — the first line `# .env.example — copy to .env and fill in your values` legitimately contains `.env` as part of the comment text, causing a false positive failure
- **Fix:** Changed assertion to `first_line.startswith('#')` which correctly verifies the first line is a comment (the actual intent)
- **Files modified:** None (verification script only, not committed)
- **Verification:** All checks passed after correction

---

**Total deviations:** 1 auto-fixed (1 false-positive bug in test assertion)
**Impact on plan:** Zero scope change. The files created match the plan spec exactly.

## Issues Encountered

None — files were created as specified. Automated verification passed after fixing a false-positive assertion in the plan's inline test.

## User Setup Required

Before running the stack:
1. Copy `.env.example` to `.env`: `cp .env.example .env`
2. Set `POSTGRES_PASSWORD` to any secure password
3. Generate and set `AIRFLOW__CORE__FERNET_KEY`: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
4. Generate and set `AIRFLOW__WEBSERVER__SECRET_KEY`: `python -c "import secrets; print(secrets.token_hex(32))"`
5. Update `APP_DB_URL`, `MLFLOW_BACKEND_STORE_URI`, `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` with your chosen password

## Next Phase Readiness

- `.env.example` contract is locked — future phases know exactly which variables are required
- `pyproject.toml` is in place — `ruff check .` and `black .` can be run immediately
- Stack verification (Task 2) must pass before Phase 1 is declared complete
- Once Task 2 passes: all of Phase 1 is complete and Phase 2 (EDA + data pipeline) can begin

---
*Phase: 01-solid-ground*
*Status: Checkpoint pending — Task 2 awaiting user verification*
*Completed: 2026-03-28 (partial)*
