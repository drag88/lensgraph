"""Postgres DSN resolution for LensGraph.

POSTGRES_DSN env var beats the local default. `db.migrate` consumes this via
psycopg.connect(dsn()) directly; long-running workers (step 3+) wrap their
own pool over the same DSN.
"""

from __future__ import annotations

import os

DEFAULT_DSN = "postgresql://lensgraph:lensgraph@localhost:5432/lensgraph"


def dsn() -> str:
    """Return the active Postgres DSN."""
    return os.environ.get("POSTGRES_DSN", DEFAULT_DSN)
