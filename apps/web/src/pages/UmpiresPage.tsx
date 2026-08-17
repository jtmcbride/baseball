/**
 * Umpire zone map (viz #13).
 *
 * Umpires have no player-page home — there's no `dim_player` row for one — so
 * this is a standalone browse-and-select view instead of a filter on
 * `PlayerPage`: pick an umpire off the season leaderboard, see where their
 * called-strike rate actually sits against the rulebook zone.
 */

import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { StrikeZoneHeatmap } from "../components/StrikeZoneHeatmap";
import { api } from "../lib/api";
import { number } from "../lib/scales";
import { useFilters } from "../store/filters";

// The client-drawn boundary line: where the smoothed actual-strike-rate
// surface crosses 50%, i.e. this umpire's own effective edge of the zone.
const CONTOUR_LEVEL = 50;

export function UmpiresPage() {
  const season = useFilters((s) => s.season);
  const [selected, setSelected] = useState<number | null>(null);

  const leaders = useQuery({
    queryKey: ["umpire-leaders", season],
    queryFn: () => api.umpireLeaders({ season: season ?? undefined, limit: 30 }),
  });

  // Default to the top-ranked umpire once the leaderboard loads, but don't
  // fight a user's own click — only reset selection if it fell off the list
  // (e.g. the season changed under them).
  useEffect(() => {
    if (!leaders.data || leaders.data.length === 0) return;
    setSelected((cur) => (cur && leaders.data.some((r) => r.mlbam_id === cur) ? cur : leaders.data[0].mlbam_id));
  }, [leaders.data]);

  const zone = useQuery({
    queryKey: ["umpire-zone", selected, season],
    queryFn: () => api.zones(selected!, "umpire", "strike_rate", season ?? undefined),
    enabled: !!selected,
    retry: false,
  });

  const selectedRow = leaders.data?.find((r) => r.mlbam_id === selected);

  return (
    <div style={{ display: "grid", gap: 16, gridTemplateColumns: "minmax(280px,1fr) minmax(340px,1fr)" }}>
      <section className="card">
        <h3>Umpires{season ? ` · ${season}` : ""}</h3>
        <p className="subtitle">
          Ranked by |edge| — actual vs. expected called-strike rate on borderline takes, against the
          called-strike model's average-umpire baseline. Click a row.
        </p>
        {leaders.isLoading ? (
          <Skeleton h={300} />
        ) : leaders.data && leaders.data.length > 0 ? (
          <table>
            <thead>
              <tr>
                <th>Umpire</th>
                <th>Edge</th>
                <th>Framing runs</th>
                <th>Takes</th>
              </tr>
            </thead>
            <tbody>
              {leaders.data.map((r) => (
                <tr
                  key={r.mlbam_id}
                  onClick={() => setSelected(r.mlbam_id)}
                  style={{
                    cursor: "pointer",
                    background: selected === r.mlbam_id ? "var(--gridline)" : undefined,
                  }}
                >
                  <td>{r.full_name ?? r.mlbam_id}</td>
                  <td>
                    {r.edge > 0 ? "+" : ""}
                    {(r.edge * 100).toFixed(1)}pp
                  </td>
                  <td>
                    {r.framing_runs != null
                      ? `${r.framing_runs > 0 ? "+" : ""}${number(r.framing_runs, 1)}`
                      : "—"}
                  </td>
                  <td>{r.n.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
            No umpire grades for this season — run <code>bb-ml called-strike-mart</code>.
          </p>
        )}
      </section>

      <section className="card">
        <h3>{selectedRow?.full_name ?? "Zone shape"}</h3>
        <p className="subtitle">
          Called-strike rate by location. The line traces the 50% crossing — this umpire's actual
          zone boundary, against the rulebook rectangle.
        </p>
        {zone.isError ? (
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
            No zone grid for this umpire-season — falls below the grid's own pitch qualifier.
          </p>
        ) : zone.data ? (
          <StrikeZoneHeatmap grid={zone.data} contourAt={CONTOUR_LEVEL} />
        ) : (
          <Skeleton h={380} />
        )}
      </section>
    </div>
  );
}

function Skeleton({ h }: { h: number }) {
  return <div style={{ height: h, background: "var(--gridline)", borderRadius: 4, opacity: 0.4 }} />;
}
