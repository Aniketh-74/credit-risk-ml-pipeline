---
status: testing
phase: 01-solid-ground
source: 01-01-SUMMARY.md, 01-02-SUMMARY.md, 01-03-SUMMARY.md, 01-04-SUMMARY.md
started: 2026-03-29T08:00:00Z
updated: 2026-03-29T08:00:00Z
---

## Current Test

number: 2
name: Alembic migration creates all 4 tables
expected: |
  alembic upgrade head prints "Running upgrade -> 0001, initial_schema" with no errors.
  \dt in app_db lists predictions, outcomes, drift_scores, alerts.
awaiting: user response

## Tests

### 1. Docker stack starts healthy
expected: docker compose ps shows postgres/mlflow/api as (healthy), airflow-init as Exited (0), scheduler and api-server as Up
result: pass

### 2. Alembic migration creates all 4 tables
expected: Running `APP_DB_URL=postgresql+psycopg2://admin:<pw>@localhost:5433/app_db alembic upgrade head` prints "Running upgrade -> 0001, initial_schema" with no errors. `\dt` in app_db lists predictions, outcomes, drift_scores, alerts.
result: [pending]

### 3. Critical schema columns exist
expected: `\d predictions` shows `path character varying(20)`. `\d outcomes` shows `outcome_received_at timestamp with time zone`.
result: [pending]

### 4. MLflow uses PostgreSQL backend
expected: MLflow UI at http://localhost:5001 loads the Experiments view. Running `docker exec $(docker compose ps -q mlflow) ls /` does NOT show a mlflow.db file.
result: [pending]

### 5. FastAPI health endpoint
expected: `curl http://localhost:8001/health` returns `{"status":"ok","version":"stub"}` with HTTP 200.
result: [pending]

### 6. Three separate databases exist
expected: `docker exec $(docker compose ps -q postgres) psql -U admin -c "\l"` lists airflow_db, mlflow_db, and app_db as separate databases.
result: [pending]

### 7. .env is protected from git
expected: `git status` does not show .env as untracked or modified. .env.example IS tracked and committed.
result: [pending]

### 8. Streamlit dashboard renders
expected: `docker exec $(docker compose ps -q dashboard) streamlit run src/dashboard/app.py --server.headless true` starts without import errors. Dashboard at http://localhost:8501 shows 4 KPI metric cards and a drift chart.
result: [pending]

### 9. Dashboard design quality
expected: The dashboard has a dark navy sidebar (not default Streamlit gray), a Plotly line chart with a red dashed threshold line at PSI=1.0, and red triangle markers at alert days 22+.
result: [pending]

## Summary

total: 9
passed: 0
issues: 0
pending: 9
skipped: 0

## Gaps

[none yet]
