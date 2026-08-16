"""The storage seam.

Every read the API and ML layers perform goes through `Warehouse`. DuckDB backs it
today; Postgres backs it when this needs to serve more than one person. Because the
logical table names and the SQL are identical across both, that swap is a config
change rather than a migration.

Consequence for callers: never open a DuckDB connection directly, and never embed a
Parquet path in a query. Ask the warehouse for a table by its logical name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import polars as pl
import pyarrow as pa


class Warehouse(ABC):
    """Read/write interface over the analytical store."""

    @abstractmethod
    def execute(self, sql: str, params: Mapping[str, Any] | None = None) -> pa.Table:
        """Run a query and return Arrow.

        Arrow rather than pandas is deliberate: it is what the API serves on the
        wire and what the ML layer consumes, so this avoids a conversion on every
        hot path.
        """

    @abstractmethod
    def execute_none(self, sql: str, params: Mapping[str, Any] | None = None) -> None:
        """Run a statement for its side effect (DDL, INSERT, CREATE TABLE AS)."""

    @abstractmethod
    def register_lake_table(self, logical_name: str, relative_path: str) -> None:
        """Expose a Parquet dataset in the lake under a logical name.

        `relative_path` is relative to the lake root and may contain hive partition
        globs, e.g. `fact_pitch/season=*/*.parquet`.
        """

    @abstractmethod
    def write_table(self, logical_name: str, table: pa.Table, *, replace: bool = True) -> None:
        """Materialize an Arrow table into the warehouse under a logical name."""

    @abstractmethod
    def table_exists(self, logical_name: str) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...

    # --- Convenience ---------------------------------------------------------

    def query_df(self, sql: str, params: Mapping[str, Any] | None = None) -> pl.DataFrame:
        return pl.from_arrow(self.execute(sql, params))  # type: ignore[return-value]

    def scalar(self, sql: str, params: Mapping[str, Any] | None = None) -> Any:
        tbl = self.execute(sql, params)
        if tbl.num_rows == 0:
            return None
        return tbl.column(0)[0].as_py()

    def __enter__(self) -> Warehouse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
