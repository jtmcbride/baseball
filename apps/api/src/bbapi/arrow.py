"""Arrow IPC responses for the large payloads.

One pitcher-season is roughly 3,000 rows x 20 columns. As JSON that is ~2 MB of
text the browser must parse into objects; as Arrow IPC it is ~200 KB that lands as
typed arrays the WebGL layer can hand straight to a buffer with no per-row work.

That difference is what separates a pitch explorer that feels instant from one
that feels like a report generator, and it is not something to retrofit: every
chart's data path is written against whichever format ships first.

JSON remains the right choice for small, human-shaped payloads (a player profile,
an arsenal table), so both are available and each route picks deliberately.
"""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.ipc as ipc
from fastapi import Response

ARROW_MEDIA_TYPE = "application/vnd.apache.arrow.stream"


def arrow_response(table: pa.Table, *, cache_seconds: int | None = None) -> Response:
    sink = io.BytesIO()
    # Compress the stream: pitch data is highly repetitive across columns, and
    # zstd typically halves it again over raw IPC.
    options = ipc.IpcWriteOptions(compression="zstd")
    with ipc.new_stream(sink, table.schema, options=options) as writer:
        writer.write_table(table)

    headers = {"X-Row-Count": str(table.num_rows)}
    if cache_seconds:
        headers["Cache-Control"] = f"public, max-age={cache_seconds}"
    return Response(content=sink.getvalue(), media_type=ARROW_MEDIA_TYPE, headers=headers)


# Completed seasons never change, so they can be cached hard. Only the current
# season needs revalidation, and even that tolerates an hour.
IMMUTABLE_SEASON_TTL = 60 * 60 * 24 * 30
CURRENT_SEASON_TTL = 60 * 60


def season_ttl(season: int | None, current_season: int) -> int:
    if season is None or season >= current_season:
        return CURRENT_SEASON_TTL
    return IMMUTABLE_SEASON_TTL
