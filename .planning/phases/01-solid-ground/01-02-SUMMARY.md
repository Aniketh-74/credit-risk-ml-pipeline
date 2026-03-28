---
phase: 01-solid-ground
plan: "02"
subsystem: infra
tags: [docker, docker-compose, fastapi, mlflow, airflow, postgres, psycopg2]

requires: []

provides:
  - "Custom MLflow Docker image (ghcr.io/mlflow/mlflow:v3.10.1 + psycopg2-binary==2.9.9)"
  - "docker-compose.yml with seven services and correct startup dependency ordering"
  - "FastAPI health stub (GET /health returning {status: ok, version: stub})"
  - "Dockerfile.api and Dockerfile.dashboard for local container builds"
  - "dags/ directory placeholder for Phase 5 Airflow DAG files"

affects:
  - 01-solid-ground
  - 01-03-PLAN
  - 01-04-PLAN
  - Phase 3 (API scoring endpoints mount to this container)
  - Phase 5 (Airflow DAGs mount into the dags/ volume)

tech-stack:
  added:
    - "ghcr.io/mlflow/mlflow:v3.10.1 (base, extended with psycopg2-binary)"
    - "psycopg2-binary==2.9.9"
    - "python:3.11-slim (api and dashboard containers)"
    - "fastapi==0.135.1"
    - "uvicorn[standard]==0.34.0"
    - "streamlit==1.44.0"
    - "apache/airflow:3.1.8"
    - "postgres:16"
  patterns:
    - "Docker Compose v2 format: no version: key at top of file"
    - "All depends_on use condition: service_healthy or condition: service_completed_successfully — no bare depends_on"
    - "All secrets use ${VAR} interpolation — no literal credentials anywhere"
    - "airflow-init as one-shot init container, other Airflow services wait for service_completed_successfully"
    - "FastAPI lifespan context manager pattern for startup/shutdown hooks"

key-files:
  created:
    - docker/Dockerfile.mlflow
    - docker/Dockerfile.api
    - docker/Dockerfile.dashboard
    - docker-compose.yml
    - src/__init__.py
    - src/api/__init__.py
    - src/api/main.py
    - src/dashboard/__init__.py
    - dags/.gitkeep
  modified: []

key-decisions:
  - "No version: key in docker-compose.yml — Docker Compose v2 plugin does not require it"
  - "start_period: 30s on postgres healthcheck — allows init-multiple-dbs.sh to finish creating airflow_db, mlflow_db, app_db before dependents start health probing"
  - "MLflow artifact root points to named volume /mlflow/artifacts — decouples artifacts from container filesystem and enables future GCS swap in Phase 7"
  - "Dockerfile.mlflow extends official MLflow image rather than building from scratch — minimizes image size while fixing psycopg2 SQLite fallback (GitHub issue #9513)"
  - "airflow-api-server (Airflow 3.x) replaces deprecated webserver command — Airflow 3.x splits web UI into separate api-server"
  - "src/dashboard/app.py NOT created here — full dashboard built in Plan 01-04 using frontend-design skill"

patterns-established:
  - "Service dependency ordering: postgres -> mlflow, airflow-init -> airflow-scheduler, airflow-api-server"
  - "Healthcheck-gated startup prevents race conditions on postgres init"
  - "FastAPI async lifespan context manager reserved for Phase 3 model loading"

requirements-completed:
  - INFRA-02
  - INFRA-03

duration: 5min
completed: "2026-03-28"
---

# Phase 01 Plan 02: Docker Infrastructure Summary

**Seven-service Docker Compose stack with healthcheck-gated dependency ordering, custom MLflow image with psycopg2-binary, and FastAPI health stub**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-28T15:38:43Z
- **Completed:** 2026-03-28T15:43:00Z
- **Tasks:** 2
- **Files created:** 9

## Accomplishments

- Custom `docker/Dockerfile.mlflow` extends `ghcr.io/mlflow/mlflow:v3.10.1` with `psycopg2-binary==2.9.9`, preventing the silent SQLite fallback documented in MLflow GitHub issue #9513
- `docker-compose.yml` defines all seven services with correct startup ordering — postgres healthcheck guards four dependent services, airflow-init guards scheduler and api-server, eliminating race conditions on first `docker compose up`
- `src/api/main.py` provides a minimal FastAPI stub with async lifespan context manager and GET /health endpoint, matching the structure Phase 3 will extend with model loading and scoring routes

## Task Commits

1. **Task 1: Custom Dockerfiles and FastAPI health stub** - `ed319c0` (feat)
2. **Task 2: docker-compose.yml with health checks and dependency ordering** - `76b7e69` (feat)

## Files Created/Modified

- `docker/Dockerfile.mlflow` - Custom MLflow image with psycopg2-binary; mlflow server command comes from docker-compose.yml
- `docker/Dockerfile.api` - python:3.11-slim FastAPI container; serves src.api.main:app via uvicorn
- `docker/Dockerfile.dashboard` - python:3.11-slim Streamlit container; app.py added in Plan 01-04
- `docker-compose.yml` - Full seven-service stack with named volumes, health checks, ${VAR} secret interpolation
- `src/api/main.py` - FastAPI app with lifespan context and /health returning {status: ok, version: stub}
- `src/__init__.py`, `src/api/__init__.py`, `src/dashboard/__init__.py` - Python package markers
- `dags/.gitkeep` - Ensures dags/ directory is committed for Airflow volume mount in Phase 5

## Decisions Made

- **No `version:` key** — Docker Compose v2 plugin treats it as ignored/deprecated; omitting it avoids the deprecation warning and is the current standard.
- **`start_period: 30s` on postgres** — `init-multiple-dbs.sh` (Plan 01-01) creates three databases on first start. Without start_period, pg_isready passes before the init script finishes, causing mlflow and api to connect before `mlflow_db` and `app_db` exist.
- **Named volume for MLflow artifacts** — `/mlflow/artifacts` on a named volume keeps artifacts independent of the container lifecycle. This same volume can be replaced with a GCS bucket mount in Phase 7 without changing the server command.
- **airflow-api-server instead of webserver** — Airflow 3.x separates the REST API from the web UI. The `api-server` command is the correct entrypoint for the Airflow REST API used in Phase 5.
- **`src/dashboard/app.py` deferred** — Dashboard implementation is scoped to Plan 01-04 with the frontend-design skill. Creating a placeholder here would require Plan 01-04 to overwrite it, creating unnecessary noise in git history.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. A `.env` file with the variables referenced via `${VAR}` in docker-compose.yml will be required before running `docker compose up`. That `.env` is created in Plan 01-03 smoke test setup.

## Next Phase Readiness

- Plan 01-03 can now run `docker compose up` to verify the full stack starts and all healthchecks pass
- postgres depends on `scripts/init-multiple-dbs.sh` from Plan 01-01 — that script must exist before running `docker compose up`
- src/dashboard/app.py is expected by Dockerfile.dashboard's CMD but is not created yet — dashboard container will fail to start until Plan 01-04 is complete (expected behavior for Phase 1)

---
*Phase: 01-solid-ground*
*Completed: 2026-03-28*
