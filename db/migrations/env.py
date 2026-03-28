"""Alembic environment configuration for credit-risk-ml-pipeline.

Reads the database URL from the APP_DB_URL environment variable.
Supports both online (live DB) and offline (SQL script) migration modes.

For async connections (postgresql+asyncpg://), the URL is automatically
rewritten to the synchronous psycopg2 driver because Alembic's migration
runner uses synchronous connections.
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from db.models import Base

# Alembic Config object — provides access to values in alembic.ini
config = context.config

# Set up Python logging from the ini file configuration
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The SQLAlchemy metadata object that autogenerate will inspect
target_metadata = Base.metadata


def get_url() -> str:
    """Read the database URL from APP_DB_URL environment variable.

    Converts async-style postgresql+asyncpg:// URLs to synchronous
    postgresql+psycopg2:// because Alembic uses synchronous connections
    for running migrations.

    Returns:
        A synchronous PostgreSQL connection URL string.

    Raises:
        RuntimeError: If APP_DB_URL is not set in the environment.
    """
    url = os.environ.get("APP_DB_URL")
    if not url:
        raise RuntimeError(
            "APP_DB_URL environment variable is not set. "
            "Set it before running alembic commands."
        )
    # Alembic runs migrations synchronously — rewrite async driver prefix
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
    return url


def run_migrations_offline() -> None:
    """Run migrations in offline mode (generate SQL without a live DB).

    Configures the context with just a URL rather than an Engine,
    so no actual database connection is required. Useful for generating
    migration SQL to review or apply manually.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in online mode against a live database connection.

    Creates an Engine from the resolved URL and associates a connection
    with the Alembic migration context, then runs all pending migrations.
    """
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
