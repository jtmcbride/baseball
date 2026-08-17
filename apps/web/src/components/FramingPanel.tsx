/**
 * Catcher framing runs — the called-strike model's residual, credited to the
 * catcher.
 *
 * `Sum (actual_strike - P(strike)) * strike_value(count)` over every take this
 * catcher received. `P(strike)` deliberately has no idea who is catching, so
 * this is what's left over after an average umpire and average pitcher are
 * subtracted out — not a raw strike rate, which would also reward a pitcher
 * who lives on the edge of the zone.
 */

import { type FramingRow, type ZoneGrid } from "../lib/api";
import { StrikeZoneHeatmap } from "./StrikeZoneHeatmap";
import { divergingColor, number } from "../lib/scales";

interface Props {
  row: FramingRow;
  /** The spatial framing-edge grid (viz #20) — optional so the scalar card
   * still renders for a catcher whose season falls under the grid's own,
   * stricter per-cell qualifier even though he clears the season total's. */
  grid?: ZoneGrid;
}

const HALF_RANGE = 15;

export function FramingPanel({ row, grid }: Props) {
  const known = Number.isFinite(row.framing_runs);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "flex", gap: 20, flexWrap: "wrap", alignItems: "flex-start" }}>
        <div
          style={{
            display: "inline-flex",
            flexDirection: "column",
            gap: 2,
            padding: "8px 10px",
            minWidth: 160,
            borderRadius: "var(--radius)",
            border: "1px solid var(--border)",
            borderLeft: `3px solid ${known ? divergingColor(row.framing_runs, 0, HALF_RANGE) : "var(--border)"}`,
          }}
        >
          <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>Framing runs</div>
          <div style={{ fontSize: 22, fontVariantNumeric: "tabular-nums" }}>
            {row.framing_runs > 0 ? "+" : ""}
            {number(row.framing_runs, 1)}
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
            {row.n.toLocaleString()} takes received
          </div>
        </div>

        {grid && (
          <div>
            <StrikeZoneHeatmap grid={grid} width={280} height={340} />
          </div>
        )}
      </div>

      <p style={{ margin: 0, fontSize: 11, color: "var(--text-muted)" }}>
        Positive means more strikes were called than an average umpire/pitcher combination would
        produce at these locations and counts. Runs, over a season — for scale, the best and worst
        full-time catchers land roughly ±10 to ±20.
        {grid && " The map to the right breaks that same residual out by location — where his receiving gains or costs strikes."}
      </p>
    </div>
  );
}
