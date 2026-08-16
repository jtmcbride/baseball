"""DuckDB-backed warehouse: local, single-writer, reads Parquet in place.

7.7M pitches x ~120 columns is comfortably within DuckDB's range, so there is no
performance reason to reach for Postgres. The adapter exists for *multi-user*
hosting, not for scale.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from bbcore.config import Settings, get_settings
from bbcore.storage.base import Warehouse

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _assert_identifier(name: str) -> None:
    """Logical table names are interpolated into DDL, so validate them."""
    if not _IDENTIFIER.match(name):
        raise ValueError(f"Unsafe logical table name: {name!r}")


class DuckDBWarehouse(Warehouse):
    def __init__(
        self,
        db_path: Path | None = None,
        *,
        settings: Settings | None = None,
        read_only: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.db_path = db_path or self.settings.duckdb_file
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lake_root = self.settings.lake_dir
        # DuckDB connections are not thread-safe; FastAPI will hit this from a
        # threadpool, so serialize access rather than hand out a shared cursor.
        self._lock = threading.RLock()
        self._con = duckdb.connect(str(self.db_path), read_only=read_only)
        self._con.execute("SET enable_progress_bar = false")

    # --- Warehouse -----------------------------------------------------------

    def execute(self, sql: str, params: Mapping[str, Any] | None = None) -> pa.Table:
        with self._lock:
            rel = self._con.execute(sql, dict(params or {}))
            # `.arrow()` returns a RecordBatchReader in duckdb >= 1.4; materialize
            # it here so callers always get a Table regardless of version.
            fetch = getattr(rel, "to_arrow_table", None) or rel.fetch_arrow_table
            result = fetch()
            if isinstance(result, pa.RecordBatchReader):  # pragma: no cover
                result = result.read_all()
            return result

    def execute_none(self, sql: str, params: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            self._con.execute(sql, dict(params or {}))

    def register_lake_table(self, logical_name: str, relative_path: str) -> None:
        pattern = str(self.lake_root / relative_path)
        # DDL cannot take bound parameters in DuckDB, so the path is inlined.
        # Quote-escape it rather than trusting the caller: these patterns come
        # from config, and a stray quote would otherwise be a SQL injection.
        literal = pattern.replace("'", "''")
        _assert_identifier(logical_name)
        with self._lock:
            self._con.execute(
                f"CREATE OR REPLACE VIEW {logical_name} AS SELECT * FROM "
                # hive_partitioning is deliberately OFF: partition values are
                # written into the files themselves (see transforms.statcast), and
                # DuckDB 1.5.5 fails when a query projects only a synthesized
                # partition column. union_by_name keeps schemas compatible across
                # seasons whose optional columns differ (bat tracking is 2024+).
                f"read_parquet('{literal}', hive_partitioning = false, union_by_name = true)"
            )

    def write_table(self, logical_name: str, table: pa.Table, *, replace: bool = True) -> None:
        with self._lock:
            self._con.register("_incoming", table)
            try:
                _assert_identifier(logical_name)
                if replace:
                    self._con.execute(
                        f"CREATE OR REPLACE TABLE {logical_name} AS SELECT * FROM _incoming"
                    )
                else:
                    self._con.execute(f"INSERT INTO {logical_name} SELECT * FROM _incoming")
            finally:
                self._con.unregister("_incoming")

    def table_exists(self, logical_name: str) -> bool:
        with self._lock:
            res = self._con.execute(
                "SELECT count(*) FROM duckdb_tables() WHERE table_name = ? "
                "UNION ALL SELECT count(*) FROM duckdb_views() WHERE view_name = ?",
                [logical_name, logical_name],
            ).fetchall()
        return any(r[0] > 0 for r in res)

    def close(self) -> None:
        with self._lock:
            self._con.close()

    # --- DuckDB extras -------------------------------------------------------

    def write_parquet_partitioned(
        self,
        table: pa.Table,
        relative_path: str,
        partition_cols: list[str],
    ) -> Path:
        """Write Arrow to the lake as hive-partitioned Parquet, overwriting the
        partitions present in `table` and leaving all others untouched."""
        target = self.lake_root / relative_path
        target.mkdir(parents=True, exist_ok=True)
        cols = ", ".join(partition_cols)
        with self._lock:
            self._con.register("_out", table)
            try:
                self._con.execute(
                    f"COPY (SELECT * FROM _out) TO '{target}' "
                    f"(FORMAT PARQUET, PARTITION_BY ({cols}), "
                    f"OVERWRITE_OR_IGNORE true, COMPRESSION zstd, FILENAME_PATTERN 'part_{{uuid}}')"
                )
            finally:
                self._con.unregister("_out")
        return target


def open_warehouse(*, read_only: bool = False, settings: Settings | None = None) -> Warehouse:
    """Factory honoring BB_WAREHOUSE_BACKEND. The only place a backend is chosen."""
    s = settings or get_settings()
    if s.warehouse_backend == "duckdb":
        return DuckDBWarehouse(settings=s, read_only=read_only)
    raise NotImplementedError(
        f"Warehouse backend {s.warehouse_backend!r} is not implemented yet "
        "(PostgresWarehouse lands in M3). Set BB_WAREHOUSE_BACKEND=duckdb."
    )
