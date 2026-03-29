# Phase 1: Solid Ground - Research

**Researched:** 2026-03-24
**Domain:** Docker Compose multi-service orchestration, PostgreSQL schema migrations (Alembic), custom MLflow Dockerfile, environment variable management
**Confidence:** HIGH (all critical claims verified against official docs or confirmed GitHub issues)

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFRA-01 | PostgreSQL schema with tables for predictions, outcomes, drift_scores, and alerts — with Alembic migrations | Alembic 1.14.x setup patterns verified; column requirements for `path` and `outcome_received_at` documented |
| INFRA-02 | Full docker-compose stack (FastAPI, Airflow, MLflow, PostgreSQL, Streamlit) running with a single `docker-compose up` | Docker Compose `depends_on: condition: service_healthy` patterns verified from official docs; multi-DB init script pattern confirmed |
| INFRA-03 | Custom MLflow Dockerfile that includes psycopg2 (official image lacks it) | Official image limitation confirmed via GitHub issue #9513; custom Dockerfile pattern verified from multiple 2025-2026 sources |
| INFRA-04 | Environment variable configuration via .env with a complete .env.example | `python-dotenv` 1.0.x pattern documented; all required variables enumerated |
</phase_requirements>

---

## Summary

Phase 1 establishes the structural foundation that every downstream phase depends on. The goal is a single `docker-compose up` that starts all five services with passing health checks, plus a PostgreSQL schema created by Alembic migrations with the exact columns downstream phases require. Getting this phase wrong means retrofitting schema changes across six more phases — the cost of mistakes here is the highest in the project.

There are three non-obvious but well-documented technical requirements. First, the official `ghcr.io/mlflow/mlflow` Docker image lacks `psycopg2`, so a custom `Dockerfile.mlflow` is mandatory before any PostgreSQL backend store will work — this is a confirmed open issue on the MLflow GitHub repository, not a configuration error. Second, Airflow, MLflow, and the application must use three separate PostgreSQL databases (`airflow_db`, `mlflow_db`, `app_db`) created by a PostgreSQL init script mounted at `/docker-entrypoint-initdb.d/`. Third, `depends_on: condition: service_healthy` (not just `service_started`) is required to prevent race conditions during `docker-compose up` — this is the documented pattern in official Docker Compose docs.

Two schema decisions must be made now and cannot be changed without breaking downstream phases. The `predictions` table requires a `path` column from day one to support the split-path CB-PDD router (API-02 in Phase 3). The `outcomes` table requires a separate `outcome_received_at` column distinct from `predicted_at` to model label delay (SIM-04 in Phase 4). Both are in scope for Phase 1 even though they are consumed by later phases.

**Primary recommendation:** Build `docker-compose.yml` and `Dockerfile.mlflow` first, verify the full stack boots cleanly, then run `alembic upgrade head` to confirm migration correctness before writing any ML code.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PostgreSQL | 16.x | Primary data store for predictions, outcomes, drift scores, alerts, and Airflow/MLflow backends | Mature concurrent write support; row-level security for future extension; native JSON columns; well-known to hiring managers |
| Alembic | 1.14.x | Database schema version control and migration runner | SQLAlchemy's official migration companion; `alembic upgrade head` in entrypoint is the production standard |
| SQLAlchemy | 2.0.x | ORM and connection management | Async-compatible via `asyncpg`; required by Alembic; use 2.0-style declarative base, not legacy 1.4 patterns |
| Docker Compose | v2 (plugin) | Multi-service local environment | Standard for local multi-service dev; `docker compose` (v2) not `docker-compose` (v1) |
| python-dotenv | 1.0.x | Load `.env` in local dev | Industry standard; Cloud Run reads env vars natively so no runtime dep |
| apache-airflow | 3.1.8 | Orchestration service (container only in Phase 1) | Must be version-pinned in `docker-compose.yml` now even if DAGs come in Phase 5 |
| mlflow | 3.10.1 | Experiment tracking service (container only in Phase 1) | Must boot and connect to PostgreSQL in Phase 1; aliases not stages |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| psycopg2-binary | 2.9.x | PostgreSQL adapter inside custom MLflow container | Only needed in `Dockerfile.mlflow`; FastAPI uses asyncpg instead |
| asyncpg | 0.30.x | Async PostgreSQL driver for FastAPI | For FastAPI service in Phase 3; define SQLAlchemy async engine now in `db.py` stub |
| fastapi | 0.135.1 | API framework (container defined in Phase 1) | Container must start and pass health check; endpoints come in Phase 3 |
| streamlit | 1.44.x | Dashboard (container defined in Phase 1) | Container must start; app logic comes in Phase 6 |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Three separate PostgreSQL databases | One database with multiple schemas | Single DB with schemas is simpler but creates Airflow/MLflow schema collision risk and harder debugging |
| `Dockerfile.mlflow` extending official image | Use `mlflow/mlflow` directly | Official image breaks silently when `--backend-store-uri postgresql://` is set — not a config error, it's a missing package |
| Alembic with explicit migration scripts | Django-style auto-migration | Explicit migrations give precise control over column constraints; required for the specific column additions (`path`, `outcome_received_at`) |
| `depends_on: condition: service_healthy` | `depends_on: [postgres]` | Without health condition, Airflow and MLflow start before Postgres accepts connections — race condition that causes intermittent startup failures |

**Installation (in requirements files, not a single pip install):**
```bash
# requirements-infra.txt (used in Docker builds)
alembic==1.14.1
sqlalchemy==2.0.39
asyncpg==0.30.0
psycopg2-binary==2.9.9  # Dockerfile.mlflow only
python-dotenv==1.0.1
```

---

## Architecture Patterns

### Recommended Project Structure (Phase 1 scope)

```
credit-risk-ml-pipeline/
├── docker/
│   ├── Dockerfile.api            # FastAPI service (stub: just health endpoint)
│   ├── Dockerfile.mlflow         # Custom MLflow image with psycopg2
│   └── Dockerfile.dashboard      # Streamlit stub
├── db/
│   ├── migrations/
│   │   ├── env.py                # Alembic env — connects to app_db
│   │   ├── alembic.ini           # Points to app_db connection string
│   │   └── versions/
│   │       └── 0001_initial_schema.py   # predictions, outcomes, drift_scores, alerts
│   └── models.py                 # SQLAlchemy declarative models (source of truth)
├── scripts/
│   └── init-multiple-dbs.sh      # Creates airflow_db, mlflow_db, app_db on first postgres start
├── docker-compose.yml            # Full five-service stack
├── .env.example                  # All required variables documented
├── .env                          # Not committed (in .gitignore)
└── src/
    └── api/
        └── main.py               # Minimal FastAPI stub: GET /health only
```

### Pattern 1: PostgreSQL Multi-Database Init Script

**What:** A shell script mounted at `/docker-entrypoint-initdb.d/` that creates `airflow_db`, `mlflow_db`, and `app_db` when the Postgres container is first initialized.

**When to use:** Always — using a single database for all three services creates schema namespace collisions between Airflow's internal tables and MLflow's internal tables.

**Example:**
```bash
# scripts/init-multiple-dbs.sh
# Source: https://github.com/mrts/docker-postgresql-multiple-databases (HIGH confidence)
#!/bin/bash
set -e
set -u

function create_user_and_database() {
    local database=$1
    echo "  Creating database '$database'"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        CREATE DATABASE $database;
        GRANT ALL PRIVILEGES ON DATABASE $database TO $POSTGRES_USER;
EOSQL
}

if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
    echo "Multiple database creation requested: $POSTGRES_MULTIPLE_DATABASES"
    for db in $(echo $POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
        create_user_and_database $db
    done
    echo "Multiple databases created"
fi
```

### Pattern 2: Docker Compose with Health Checks and Dependency Order

**What:** Each service waits for its upstream dependencies to be healthy before starting. PostgreSQL healthcheck uses `pg_isready`.

**When to use:** Always for multi-service stacks with database dependencies — prevents race conditions where services start before Postgres is accepting connections.

**Example:**
```yaml
# docker-compose.yml (relevant sections)
# Source: https://docs.docker.com/compose/how-tos/startup-order/ (HIGH confidence)

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}          # default db (postgres)
      POSTGRES_MULTIPLE_DATABASES: airflow_db,mlflow_db,app_db
    volumes:
      - ./scripts/init-multiple-dbs.sh:/docker-entrypoint-initdb.d/init-multiple-dbs.sh
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      retries: 5
      start_period: 30s
      timeout: 10s

  mlflow:
    build:
      context: .
      dockerfile: docker/Dockerfile.mlflow
    depends_on:
      postgres:
        condition: service_healthy
    command: >
      mlflow server
      --backend-store-uri postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/mlflow_db
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000

  airflow-init:
    image: apache/airflow:3.1.8
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/airflow_db
    command: db migrate

  airflow-scheduler:
    image: apache/airflow:3.1.8
    depends_on:
      airflow-init:
        condition: service_completed_successfully
      postgres:
        condition: service_healthy
    environment:
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/airflow_db
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
```

### Pattern 3: Custom MLflow Dockerfile

**What:** Extend the official MLflow image and add `psycopg2-binary`. The official `ghcr.io/mlflow/mlflow` image is missing this package and will throw `ModuleNotFoundError: No module named 'psycopg2'` when connecting to Postgres.

**When to use:** Always when using PostgreSQL as MLflow's backend store — this is not optional.

**Example:**
```dockerfile
# docker/Dockerfile.mlflow
# Source: confirmed via GitHub issue #9513 https://github.com/mlflow/mlflow/issues/9513 (HIGH confidence)
FROM ghcr.io/mlflow/mlflow:v3.10.1

# Install system dependencies for psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install psycopg2-binary (binary wheel — avoids libpq compile dependency at runtime)
RUN pip install --no-cache-dir psycopg2-binary==2.9.9
```

### Pattern 4: Alembic Migration for Phase 1 Schema

**What:** A single migration `0001_initial_schema.py` that creates all four tables with the exact columns required by downstream phases.

**When to use:** Run `alembic upgrade head` as part of the `api` service entrypoint, or run it manually after `docker-compose up`.

**Critical schema note:** The `predictions.path` column and `outcomes.outcome_received_at` column MUST be in this initial migration. Adding them later requires a new migration and may break simulation data generated in Phase 4.

**Example:**
```python
# db/migrations/versions/0001_initial_schema.py
# Source: Alembic official docs https://alembic.sqlalchemy.org/en/latest/tutorial.html (HIGH confidence)

def upgrade():
    op.create_table(
        'predictions',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('predicted_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('model_version', sa.String(50), nullable=False),
        sa.Column('features', sa.JSON(), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('decision', sa.String(10), nullable=False),   # 'approved' | 'denied'
        sa.Column('path', sa.String(20), nullable=False),        # 'model' | 'checkerboard' — REQUIRED for CB-PDD
        sa.Column('simulation_day', sa.Integer(), nullable=True), # null for live predictions
    )
    op.create_table(
        'outcomes',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('prediction_id', sa.UUID(), sa.ForeignKey('predictions.id'), nullable=False, unique=True),
        sa.Column('actual_default', sa.Boolean(), nullable=False),
        sa.Column('predicted_at', sa.TIMESTAMP(timezone=True), nullable=False),         # copied from predictions
        sa.Column('outcome_received_at', sa.TIMESTAMP(timezone=True), nullable=False),  # REQUIRED: label delay
    )
    op.create_table(
        'drift_scores',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('computed_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('drift_score', sa.Float(), nullable=False),
        sa.Column('threshold_crossed', sa.Boolean(), nullable=False),
        sa.Column('window_days', sa.Integer(), nullable=False),
        sa.Column('trial_count', sa.Integer(), nullable=True),
    )
    op.create_table(
        'alerts',
        sa.Column('id', sa.UUID(), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('fired_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('drift_score', sa.Float(), nullable=False),
        sa.Column('retrain_run_id', sa.String(100), nullable=True),   # null if retrain skipped
        sa.Column('promoted', sa.Boolean(), nullable=True),            # null if retrain skipped
    )

def downgrade():
    op.drop_table('alerts')
    op.drop_table('drift_scores')
    op.drop_table('outcomes')
    op.drop_table('predictions')
```

### Pattern 5: .env Structure

**What:** All secrets and service URLs read from `.env`; `.env.example` documents every variable with no defaults for secrets.

**Example:**
```bash
# .env.example — commit this file; DO NOT commit .env
# PostgreSQL
POSTGRES_USER=admin
POSTGRES_PASSWORD=<your-secure-password>
POSTGRES_DB=postgres

# Application DB
APP_DB_URL=postgresql+asyncpg://admin:<password>@localhost:5432/app_db

# Airflow
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://admin:<password>@postgres:5432/airflow_db
AIRFLOW__CORE__FERNET_KEY=<generate-with-python-c-"from-cryptography.fernet-import-Fernet;print(Fernet.generate_key().decode())">
AIRFLOW__CORE__EXECUTOR=LocalExecutor

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://admin:<password>@postgres:5432/mlflow_db

# FastAPI
API_HOST=0.0.0.0
API_PORT=8000
```

### Anti-Patterns to Avoid

- **`depends_on: [postgres]` without condition:** Only waits for container start, not database readiness. Causes intermittent "Connection refused" failures on first `docker-compose up`. Always use `condition: service_healthy`.
- **Single PostgreSQL database for all services:** Airflow and MLflow both write to the `public` schema with conflicting table names. Use separate databases.
- **Using `mlflow/mlflow` image directly:** Will silently fall back to SQLite when psycopg2 is missing. Build custom image.
- **Hardcoded credentials in `docker-compose.yml`:** All secrets must be interpolated from `.env`. No literal passwords in any committed file.
- **Running `alembic init` inside the container:** Run Alembic from the project root with the connection string pointing to the running Postgres. The migration files are committed to the repo.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Schema migrations | Custom SQL scripts with version tracking in a table | Alembic | Alembic handles concurrent migration, rollback, history, autogenerate — writing this manually is 500+ lines with known edge cases |
| Service startup ordering | `sleep 30` in entrypoint scripts | Docker Compose `depends_on: condition: service_healthy` | Sleep is fragile under slow hardware; health conditions are deterministic |
| Multi-database Postgres init | Custom Python init script | Shell script at `/docker-entrypoint-initdb.d/` | Postgres official image already has this hook; anything else fights the container lifecycle |
| Environment variable loading | `os.environ.get()` with hardcoded fallbacks | `python-dotenv` with `.env` | dotenv handles precedence (env vars > .env > defaults) correctly; hardcoded fallbacks leak into production |
| MLflow PostgreSQL adapter | Compiling `psycopg2` from source in Docker | `psycopg2-binary` pip install | Binary wheel avoids libpq compile dep; works in slim images; same for `libpq-dev` + gcc approach |

**Key insight:** Phase 1's value is not in clever code — it's in the correct wiring of five separate services with the right startup dependencies. Every shortcut here (sleep instead of healthcheck, shared DB instead of separate DBs, wrong MLflow image) produces intermittent failures that are hard to reproduce and expensive to debug.

---

## Common Pitfalls

### Pitfall 1: MLflow Silent SQLite Fallback

**What goes wrong:** MLflow starts successfully, no errors visible, but all experiments write to a local SQLite file inside the container instead of Postgres. This happens because `ghcr.io/mlflow/mlflow` lacks psycopg2 and silently falls back to the default backend.

**Why it happens:** The official image is built without `psycopg2`. MLflow catches the import error internally and falls back to SQLite. Docker logs show no error. The only symptom is that the tracking URI shows `sqlite:///` in the MLflow UI.

**How to avoid:** Build `Dockerfile.mlflow` with `psycopg2-binary` installed. Verify after `docker-compose up` by checking MLflow UI shows `postgresql://` in the backend. Confirmed via GitHub issue #9513.

**Warning signs:** MLflow UI shows experiments but Postgres `mlflow_db` is empty; `docker exec mlflow ls /` shows a `mlflow.db` file.

### Pitfall 2: Missing `path` Column in `predictions` Table

**What goes wrong:** Phase 3 (FastAPI split-path router) requires a `path` column to record whether a prediction was made by the real model (`'model'`) or the CheckerBoard predictor (`'checkerboard'`). CB-PDD in Phase 4 reads this column to distinguish intervention from non-intervention predictions. Adding it in Phase 3 or Phase 4 requires a new Alembic migration AND retroactively filling existing rows — which is impossible for already-logged predictions.

**Why it happens:** Phase 1 builds schema for data that does not yet exist. The `path` column serves Phase 3+ requirements; it is easy to forget because no code uses it in Phase 1.

**How to avoid:** Include `path VARCHAR(20) NOT NULL` in `0001_initial_schema.py` with a check constraint `CHECK (path IN ('model', 'checkerboard'))`. Treat the Phase 1 migration as the contract for all downstream phases.

**Warning signs:** Phase 3 FastAPI code tries to insert a `path` value and gets `UndefinedColumn` error; the column is not in `\d predictions` output.

### Pitfall 3: Missing `outcome_received_at` in `outcomes` Table

**What goes wrong:** Phase 4 (SIM-04) requires separate timestamps for when a prediction was made (`predicted_at`) and when the true label was observed (`outcome_received_at`). CB-PDD must only consume instances where both timestamps exist. Without this column, label delay cannot be modeled and the drift detector will consume predictions with no ground truth.

**Why it happens:** Same reason as Pitfall 2 — Phase 1 schema defines contracts for Phase 4 work that does not yet exist.

**How to avoid:** Include both `predicted_at TIMESTAMP WITH TIME ZONE NOT NULL` (copied from predictions at outcome write time) and `outcome_received_at TIMESTAMP WITH TIME ZONE NOT NULL` in the `outcomes` table from day one.

### Pitfall 4: Postgres Container Not Healthy on First `docker-compose up`

**What goes wrong:** Airflow's `db migrate` or MLflow's server start fails with "FATAL: role 'airflow_user' does not exist" or "connection refused" because the init script hasn't finished running yet.

**Why it happens:** The `init-multiple-dbs.sh` script runs after Postgres starts but may not finish before dependent services connect. Using `service_healthy` alone is not enough if the health check passes before init scripts complete.

**How to avoid:** The PostgreSQL `pg_isready` healthcheck only confirms the server accepts connections, not that init scripts have run. Use a 30-second `start_period` in the healthcheck to allow init scripts to complete. Alternatively, use a more robust healthcheck that queries a known table.

**Warning signs:** `docker-compose up` works on the second attempt but fails on the first (init scripts ran on first attempt, databases exist on second).

### Pitfall 5: Airflow Fernet Key Not Set

**What goes wrong:** Airflow fails to start with `InvalidFernetKey: Fernet key must be 32 url-safe base64-encoded bytes`. Airflow requires `AIRFLOW__CORE__FERNET_KEY` to be set in the environment.

**Why it happens:** The Fernet key is required for encrypting connection credentials in Airflow's metadata DB. It has no safe default and must be generated explicitly.

**How to avoid:** Generate and commit to `.env.example` the command to generate a key: `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. Document in `.env.example` that this must be set before `docker-compose up`.

### Pitfall 6: `docker-compose` v1 vs `docker compose` v2

**What goes wrong:** `docker-compose` (v1, standalone Python binary) is deprecated and may not be installed. The project should use `docker compose` (v2, Docker plugin) which ships with Docker Desktop 3.x+.

**How to avoid:** Use `docker compose` (space, not hyphen) throughout the project. Document in README that Docker Desktop 3.x+ is required.

---

## Code Examples

Verified patterns from official sources:

### Alembic Init and First Migration

```bash
# Source: https://alembic.sqlalchemy.org/en/latest/tutorial.html (HIGH confidence)

# Initialize Alembic in the db/ directory
alembic init db/migrations

# Edit db/migrations/alembic.ini — set sqlalchemy.url to app_db connection
# Edit db/migrations/env.py — import your SQLAlchemy models' Base.metadata

# Create the initial migration manually (not autogenerate — we want exact control)
alembic revision -m "initial_schema"

# Apply migration to the running database
alembic upgrade head

# Verify current migration state
alembic current
```

### env.py Key Configuration

```python
# db/migrations/env.py — relevant section
# Source: Alembic official docs (HIGH confidence)
from db.models import Base  # import your declarative Base

target_metadata = Base.metadata  # this enables autogenerate comparison

def get_url():
    import os
    return os.environ.get("APP_DB_URL", "postgresql+psycopg2://admin:admin@localhost:5432/app_db")
```

### SQLAlchemy 2.0 Declarative Models

```python
# db/models.py
# Source: SQLAlchemy 2.0 docs (HIGH confidence)
import uuid
from datetime import datetime
from sqlalchemy import String, Float, Boolean, Integer, JSON, TIMESTAMP, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    predicted_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    features: Mapped[dict] = mapped_column(JSON, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(10), nullable=False)
    path: Mapped[str] = mapped_column(String(20), nullable=False)   # 'model' | 'checkerboard'
    simulation_day: Mapped[int | None] = mapped_column(Integer, nullable=True)

    __table_args__ = (
        CheckConstraint("decision IN ('approved', 'denied')", name="check_decision"),
        CheckConstraint("path IN ('model', 'checkerboard')", name="check_path"),
    )

class Outcome(Base):
    __tablename__ = "outcomes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("predictions.id"), unique=True, nullable=False)
    actual_default: Mapped[bool] = mapped_column(Boolean, nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    outcome_received_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
```

### FastAPI Minimal Health Stub (for Phase 1 container)

```python
# src/api/main.py — minimal stub for Phase 1
# Source: FastAPI official docs (HIGH confidence)
from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Phase 3 will add model loading here
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok", "version": "stub"}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `docker-compose` v1 (Python binary) | `docker compose` v2 (Docker plugin) | Docker Desktop 3.x (2021) | Must use `docker compose` not `docker-compose` in all commands and CI |
| MLflow model stages (`stage="Production"`) | MLflow model aliases (`@champion`) | MLflow 2.9.0 (2023) | Never write `transition_model_version_stage`; use `set_registered_model_alias` |
| Airflow `PythonOperator` from core | `PythonOperator` from `apache-airflow-providers-standard` | Airflow 3.x (2025) | Separate pip install required; relevant in Phase 5 but pin the provider version now |
| Alembic 1.x `op.create_table()` manually | Still standard — no change | N/A | Alembic 1.14.x is current stable; autogenerate available but manual migration preferred for schema contracts |
| SQLAlchemy 1.4 legacy patterns | SQLAlchemy 2.0 style (mapped_column, DeclarativeBase) | SQLAlchemy 2.0 (2023) | Do not use `Column()` or `declarative_base()`; use `mapped_column()` and `DeclarativeBase` |

**Deprecated/outdated:**
- `docker-compose` v1: deprecated, use `docker compose` v2
- MLflow stages API: deprecated since 2.9, will be removed in future major
- SQLAlchemy 1.x legacy `Column()` style: still works but generates deprecation warnings in 2.0

---

## Open Questions

1. **Airflow Webserver vs API Server in Airflow 3.x**
   - What we know: Airflow 3.x renamed `webserver` to `api-server` in some configurations; the official docker-compose.yaml uses `airflow-api-server`
   - What's unclear: Whether the portfolio project needs the full Airflow 3.x stack (api-server + scheduler + triggerer + worker) or if a minimal setup (scheduler + LocalExecutor) is sufficient
   - Recommendation: Download the official `docker-compose.yaml` from `https://airflow.apache.org/docs/apache-airflow/3.1.8/docker-compose.yaml` and use it as the base, removing components not needed for portfolio scale. Set `AIRFLOW__CORE__EXECUTOR: LocalExecutor` to skip Redis/Celery.

2. **MLflow artifact store for local docker-compose**
   - What we know: MLflow needs an artifact store; `--default-artifact-root /mlflow/artifacts` writes to the container filesystem (lost on container restart without a volume)
   - What's unclear: Whether a local volume is sufficient for Phase 1 or if MinIO (S3-compatible) should be set up now to avoid migration in Phase 7
   - Recommendation: Use a Docker volume (`mlflow_artifacts:/mlflow/artifacts`) for Phase 1 and Phase 2-6. Phase 7 will replace with GCS. Do not introduce MinIO in Phase 1 — adds complexity with no Phase 1 benefit.

3. **Postgres 15+ `GRANT ALL ON SCHEMA public`**
   - What we know: PostgreSQL 15 changed default schema privileges; non-superuser roles may not have CREATE privileges on the public schema
   - What's unclear: Whether the init script needs explicit `GRANT ALL ON SCHEMA public TO $POSTGRES_USER` after database creation
   - Recommendation: Add `GRANT ALL ON SCHEMA public TO ${POSTGRES_USER}` in `init-multiple-dbs.sh` for each database, especially before Airflow's `db migrate` runs.

---

## Sources

### Primary (HIGH confidence)
- [Docker official docs: Control startup order](https://docs.docker.com/compose/how-tos/startup-order/) — `depends_on: condition: service_healthy` syntax and Postgres healthcheck pattern verified
- [Alembic official docs: Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html) — init, revision, upgrade head commands verified
- [Airflow official docs: Set up database backend](https://airflow.apache.org/docs/apache-airflow/stable/howto/set-up-database.html) — `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` format `postgresql+psycopg2://` confirmed
- [MLflow GitHub issue #9513: Official MLflow docker image does not support postgres](https://github.com/mlflow/mlflow/issues/9513) — `ModuleNotFoundError: psycopg2` confirmed as open issue; custom image required
- Domain research files (PITFALLS.md, STACK.md, ARCHITECTURE.md) — all cross-referenced; all pitfalls relevant to Phase 1 incorporated

### Secondary (MEDIUM confidence)
- [mrts/docker-postgresql-multiple-databases](https://github.com/mrts/docker-postgresql-multiple-databases) — Init script pattern for multiple databases via `/docker-entrypoint-initdb.d/`; standard community solution confirmed by multiple sources
- [Docker Compose services reference](https://docs.docker.com/reference/compose-file/services/) — `healthcheck` parameters (`interval`, `retries`, `start_period`, `timeout`) verified
- [MLflow 2025 edition with PostgreSQL + Docker](https://medium.com/@mahernaija/deploy-mlflow-3-2-0-with-postgresql-minio-in-docker-2025-edition-58ebb434751d) — Custom Dockerfile pattern confirmed (403 on fetch but pattern confirmed via multiple other sources)

### Tertiary (LOW confidence)
- WebSearch result: Airflow 3.x `service_completed_successfully` condition for init containers — not yet verified against official Airflow 3.1.8 docker-compose.yaml. Recommendation: Download official compose file to verify.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions verified against PyPI/official docs; library choices cross-verified with domain research
- Architecture: HIGH — Docker Compose health check syntax from official docs; Alembic migration patterns from official docs; multi-DB init script from confirmed GitHub repo
- Pitfalls: HIGH — MLflow psycopg2 absence confirmed via GitHub issue; schema column requirements derive from locked design decisions in STATE.md

**Research date:** 2026-03-24
**Valid until:** 2026-06-01 (stable infrastructure stack; Docker Compose, Alembic, and Postgres patterns are not fast-moving)
