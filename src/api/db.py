"""SQLAlchemy session factory for the scoring API.

The Session factory is module-scope, initialized from APP_DB_URL at import
time. Route handlers and BackgroundTasks create sessions via:
    with Session() as session:
        ...

Using a module-scope factory (not request-scoped dependency injection) is
required for BackgroundTask compatibility: dependency-injected sessions close
when the HTTP response completes, before the BackgroundTask runs, causing
DetachedInstanceError. A fresh factory-created session avoids this entirely.
"""
from __future__ import annotations
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_engine = create_engine(
    os.environ["APP_DB_URL"],
    pool_pre_ping=True,   # detect stale connections before checkout
    pool_size=5,
    max_overflow=10,
)

Session = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
