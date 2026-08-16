import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { ArsenalTable } from "../components/ArsenalTable";
import { MovementPlot, type MovementPoint } from "../components/MovementPlot";
import { ReleasePlot, type ReleasePoint } from "../components/ReleasePlot";
import { StrikeZoneHeatmap } from "../components/StrikeZoneHeatmap";
import { VeloTrend, type VeloPoint } from "../components/VeloTrend";
import { api, columns } from "../lib/api";
import { useFilters } from "../store/filters";

export function PlayerPage() {
  const { playerId, role, season, metric, pitchType, vsHand, setPitchType } = useFilters();

  const profile = useQuery({
    queryKey: ["profile", playerId],
    queryFn: () => api.profile(playerId!),
    enabled: !!playerId,
  });

  const arsenal = useQuery({
    queryKey: ["arsenal", playerId, season],
    queryFn: () => api.arsenal(playerId!, season ?? undefined),
    enabled: !!playerId && role === "pitcher",
  });

  const pitches = useQuery({
    queryKey: ["pitches", playerId, role, season, vsHand],
    queryFn: () =>
      api.pitches({
        [role === "pitcher" ? "pitcher_id" : "batter_id"]: playerId,
        season: season ?? undefined,
        vs_hand: vsHand ?? undefined,
      }),
    enabled: !!playerId,
  });

  const zone = useQuery({
    queryKey: ["zone", playerId, role, metric, season],
    queryFn: () => api.zones(playerId!, role, metric, season ?? undefined),
    enabled: !!playerId,
    retry: false,
  });

  // One Arrow payload feeds three charts — decoded once, filtered per chart.
  const rows = useMemo(() => {
    if (!pitches.data) return [];
    return columns<MovementPoint & ReleasePoint & VeloPoint>(pitches.data, [
      "pitch_type", "release_speed", "ivb_in", "hb_arm_in",
      "release_pos_x", "release_pos_z", "inning", "is_whiff",
    ]);
  }, [pitches.data]);

  const filtered = useMemo(
    () => (pitchType ? rows.filter((r) => r.pitch_type === pitchType) : rows),
    [rows, pitchType],
  );

  if (!playerId) {
    return (
      <p style={{ color: "var(--text-muted)", padding: "40px 0", textAlign: "center" }}>
        Search for a player to begin.
      </p>
    );
  }

  const player = profile.data?.player;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <header>
        <h2 style={{ margin: 0, fontSize: 22 }}>{player?.full_name ?? "…"}</h2>
        <p style={{ margin: 0, color: "var(--text-secondary)", fontSize: 13 }}>
          {player?.primary_position} · bats {player?.bats ?? "?"} / throws {player?.throws ?? "?"}
          {season ? ` · ${season}` : " · all seasons"}
          {pitchType && (
            <>
              {" · "}
              <button
                onClick={() => setPitchType(null)}
                style={{
                  background: "none", border: "none", padding: 0, cursor: "pointer",
                  color: "var(--family-fastball)", font: "inherit",
                }}
              >
                filtered to {pitchType} ✕
              </button>
            </>
          )}
        </p>
      </header>

      <div style={{ display: "grid", gap: 16, gridTemplateColumns: "repeat(auto-fit,minmax(340px,1fr))" }}>
        <section className="card">
          <h3>Location profile</h3>
          <p className="subtitle">
            Smoothed surface with reliability fading. {zone.data?.n_pitches?.toLocaleString() ?? "—"} pitches.
          </p>
          {zone.isError ? (
            <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
              No zone grid — this player may fall below the qualifier.
            </p>
          ) : zone.data ? (
            <StrikeZoneHeatmap grid={zone.data} />
          ) : (
            <Skeleton h={380} />
          )}
        </section>

        <section className="card">
          <h3>Movement</h3>
          <p className="subtitle">Arm-side normalized, so hands are directly comparable.</p>
          {rows.length ? <MovementPlot points={filtered} /> : <Skeleton h={400} />}
        </section>

        <section className="card">
          <h3>Release point</h3>
          <p className="subtitle">Catcher's view. Tight clustering hides the pitch.</p>
          {rows.length ? <ReleasePlot points={filtered} /> : <Skeleton h={300} />}
        </section>

        <section className="card">
          <h3>Velocity by inning</h3>
          <p className="subtitle">Fatigue curve across the outing.</p>
          {rows.length ? <VeloTrend points={filtered} /> : <Skeleton h={240} />}
        </section>
      </div>

      {role === "pitcher" && (
        <section className="card">
          <h3>Arsenal</h3>
          <p className="subtitle">
            Percentiles rank each pitch against the same pitch type league-wide. Click a row to filter.
          </p>
          {arsenal.data ? (
            <ArsenalTable rows={arsenal.data} onSelect={setPitchType} selected={pitchType} />
          ) : (
            <Skeleton h={160} />
          )}
        </section>
      )}
    </div>
  );
}

function Skeleton({ h }: { h: number }) {
  return <div style={{ height: h, background: "var(--gridline)", borderRadius: 4, opacity: 0.4 }} />;
}
