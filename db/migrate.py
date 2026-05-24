"""Raw-SQL migration runner.

`python -m db.migrate` applies pending migrations to $POSTGRES_DSN. Migration
files live in db/migrations/ and are named `NNNN_<slug>.sql`; the version is
the filename without `.sql`. Applied versions are recorded in
`schema_migrations(version text PK, applied_at timestamptz)`.

Each file runs in its own transaction. A mid-file failure rolls back that
file's partial state and leaves prior migrations committed; the runner
re-raises so subsequent files are not attempted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg

from db.conn import dsn as resolve_dsn

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _discover() -> list[tuple[str, Path]]:
    """Return [(version, path), ...] sorted lexicographically by filename."""
    return [(p.stem, p) for p in sorted(MIGRATIONS_DIR.glob("*.sql"))]


def _applied_versions(conn: psycopg.Connection) -> set[str]:
    """Versions already in schema_migrations; empty set if the table doesn't exist yet."""
    row = conn.execute("SELECT to_regclass('public.schema_migrations')").fetchone()
    if row is None or row[0] is None:
        return set()
    rows = conn.execute("SELECT version FROM schema_migrations").fetchall()
    return {r[0] for r in rows}


def apply(dsn: str | None = None) -> list[str]:
    """Apply all pending migrations. Return list of versions applied this call."""
    target = dsn or resolve_dsn()
    discovered = _discover()
    if not discovered:
        return []

    applied: list[str] = []
    # autocommit=True so `with conn.transaction()` opens an explicit BEGIN/COMMIT
    # block per file; a failing file rolls back without affecting prior commits.
    with psycopg.connect(target, autocommit=True) as conn:
        already = _applied_versions(conn)
        for version, path in discovered:
            if version in already:
                continue
            sql = path.read_text()
            with conn.transaction():
                conn.execute(sql)
                conn.execute(
                    "INSERT INTO schema_migrations(version) VALUES (%s)",
                    (version,),
                )
            applied.append(version)
    return applied


def main(argv: list[str] | None = None) -> int:
    applied = apply()
    if applied:
        for v in applied:
            print(f"applied {v}")
    else:
        print("no pending migrations")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
