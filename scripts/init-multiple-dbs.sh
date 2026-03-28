#!/bin/bash
# init-multiple-dbs.sh
#
# PostgreSQL init script that creates multiple databases on first start.
# Mount this file to /docker-entrypoint-initdb.d/ in your postgres service.
#
# Usage:
#   Set POSTGRES_MULTIPLE_DATABASES=airflow_db,mlflow_db,app_db
#   in docker-compose.yml environment — no passwords stored in this file.
#
# PostgreSQL 15+ changed the default schema privileges so GRANT ALL ON
# SCHEMA public is required explicitly, in addition to database-level grants.

set -e
set -u

create_user_and_database() {
    local database=$1
    echo "Creating database '$database' owned by '$POSTGRES_USER'"

    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        CREATE DATABASE $database OWNER $POSTGRES_USER;
        GRANT ALL PRIVILEGES ON DATABASE $database TO $POSTGRES_USER;
EOSQL

    # Connect to the new database to grant schema-level privileges
    # Required for PostgreSQL 15+ where public schema is no longer
    # world-writable by default.
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$database" <<-EOSQL
        GRANT ALL ON SCHEMA public TO $POSTGRES_USER;
EOSQL

    echo "Database '$database' created and permissions granted."
}

if [ -n "${POSTGRES_MULTIPLE_DATABASES:-}" ]; then
    echo "Multiple database creation requested: $POSTGRES_MULTIPLE_DATABASES"

    for db in $(echo "$POSTGRES_MULTIPLE_DATABASES" | tr ',' ' '); do
        create_user_and_database "$db"
    done

    echo "All databases created successfully."
fi
