/**
 * A whole at-bat: every pitch in one 3D scene, over the hitter's own zone map.
 *
 * A plate appearance is a sequence, not a set of independent pitches, and the
 * two halves of this panel are the two things a reader needs to see it that
 * way. The 3D scene shows the pitches sharing (or not sharing) a tunnel — the
 * same instrument as the arsenal comparison, pointed at one at-bat. The heat map
 * beside it is the BATTER's season, with this at-bat's pitches drawn on top in
 * order, which is what turns "he threw a slider down and away" into "he threw it
 * into this hitter's coldest square".
 *
 * The zone map is the batter's whole season and the marks are six pitches from
 * one afternoon. That mismatch is the point — the surface is the prior, the
 * marks are the event — but it is also why the marks are drawn as marks and
 * never folded into the surface.
 *
 * Untracked pitches (pitch-clock violations, ABS calls, pre-tracking seasons)
 * appear in the sequence strip and not in the 3D scene. They have no flight to
 * draw, and dropping them from the strip too would renumber the at-bat and make
 * a count appear out of nowhere.
 */

import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { api, type AtBatPitch, type AtBatSummary } from "../lib/api";
import { familyColor, familyOf, labelOf, number } from "../lib/scales";
import type { PhysicsParams } from "../lib/trajectory";
import { PitchScene3D, type ScenePitch } from "./PitchScene3D";
import { StrikeZoneHeatmap, type ZoneMark } from "./StrikeZoneHeatmap";

interface Props {
  playerId: number;
  role: "pitcher" | "batter";
  season: number | null;
  /** Which zone surface to draw behind the at-bat — the shared filter-bar metric. */
  metric: string;
}

/** True when the pitch carries the full 9-parameter fit a flight needs. */
function hasFlight(p: AtBatPitch): boolean {
  return (
    p.release_pos_x != null && p.release_pos_y != null && p.release_pos_z != null &&
    p.vx0 != null && p.vy0 != null && p.vz0 != null &&
    p.ax != null && p.ay != null && p.az != null
  );
}

function physicsOf(p: AtBatPitch): PhysicsParams {
  return {
    release_pos_x: p.release_pos_x!, release_pos_y: p.release_pos_y!, release_pos_z: p.release_pos_z!,
    vx0: p.vx0!, vy0: p.vy0!, vz0: p.vz0!,
    ax: p.ax!, ay: p.ay!, az: p.az!,
  };
}

function outcomeOf(p: AtBatPitch): string {
  if (p.events) return p.events.replace(/_/g, " ");
  return (p.description ?? "").replace(/_/g, " ");
}

function labelAtBat(ab: AtBatSummary): string {
  const result = ab.result ? ab.result.replace(/_/g, " ") : "—";
  return `${ab.game_date} · ${ab.batter_name ?? ab.batter} vs ${ab.pitcher_name ?? ab.pitcher}` +
    ` · ${ab.pitches}p · ${result}`;
}

export function AtBat3D({ playerId, role, season, metric }: Props) {
  const [chosen, setChosen] = useState<string | null>(null);
  const [selectedPitch, setSelectedPitch] = useState<number | null>(null);
  const [showAll, setShowAll] = useState(true);

  // A stale at-bat key from the previous player would 404 against this one.
  useEffect(() => {
    setChosen(null);
    setSelectedPitch(null);
  }, [playerId, season]);

  const list = useQuery({
    queryKey: ["atbats", playerId, role, season],
    queryFn: () =>
      api.atBats({
        [role === "pitcher" ? "pitcher_id" : "batter_id"]: playerId,
        season: season ?? undefined,
        limit: 60,
      }),
    enabled: !!playerId,
    retry: false,
  });

  // Default to the most recent at-bat with something to fly.
  const fallback = list.data?.find((a) => a.tracked_pitches > 0) ?? list.data?.[0];
  const key = chosen ?? (fallback ? `${fallback.game_pk}:${fallback.at_bat_number}` : null);
  const [gamePk, atBatNumber] = key ? key.split(":").map(Number) : [null, null];

  const detail = useQuery({
    queryKey: ["atbat", gamePk, atBatNumber],
    queryFn: () => api.atBat(gamePk!, atBatNumber!),
    enabled: gamePk != null && atBatNumber != null,
    retry: false,
  });

  const zone = useQuery({
    queryKey: ["zone", detail.data?.batter, "batter", metric, season],
    queryFn: () => api.zones(detail.data!.batter, "batter", metric, season ?? undefined),
    enabled: !!detail.data?.batter,
    retry: false,
  });

  // Memoized rather than defaulted inline: `?? []` is a fresh array every
  // render, and every memo downstream of it would recompute on each one.
  const pitches = useMemo(() => detail.data?.pitches ?? [], [detail.data]);
  const flyable = useMemo(() => pitches.filter(hasFlight), [pitches]);

  // Selecting a pitch from a previous at-bat would emphasize a pitch number
  // that means something else here.
  useEffect(() => setSelectedPitch(null), [key]);

  // `selectedPitch` stays null until the reader picks one, and only a real pick
  // dims the rest: defaulting the emphasis to pitch #1 made "whole at-bat" look
  // like a one-pitch view with three ghosts behind it. `single` is the separate
  // question of which pitch the one-pitch view shows when nothing is picked.
  const single = selectedPitch ?? flyable[0]?.pitch_number ?? null;

  const scenePitches: ScenePitch[] = useMemo(() => {
    const source = showAll ? flyable : flyable.filter((p) => p.pitch_number === single);
    return source.map((p) => ({
      id: `p${p.pitch_number}`,
      trajectory: physicsOf(p),
      pitchType: p.pitch_type,
    }));
  }, [flyable, showAll, single]);

  const marks: ZoneMark[] = useMemo(
    () =>
      pitches
        .filter((p) => p.plate_x != null && p.plate_z_norm != null)
        .map((p) => ({
          key: `m${p.pitch_number}`,
          x: p.plate_x!,
          z: p.plate_z_norm!,
          label: String(p.pitch_number),
          pitchType: p.pitch_type,
          emphasis: p.pitch_number === selectedPitch,
          title: `#${p.pitch_number} ${labelOf(p.pitch_type)} · ${p.balls}-${p.strikes} · ${outcomeOf(p)}`,
          onClick: () => setSelectedPitch(p.pitch_number),
        })),
    [pitches, selectedPitch],
  );

  if (list.isError) {
    return <Muted>Could not load at-bats for this player.</Muted>;
  }
  if (!list.data) return <Skeleton h={420} />;
  if (!list.data.length) return <Muted>No at-bats found.</Muted>;

  const ab = detail.data;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <select
          value={key ?? ""}
          onChange={(e) => setChosen(e.target.value)}
          style={{
            padding: "4px 8px", borderRadius: 4, border: "1px solid var(--gridline)",
            background: "var(--surface-1)", color: "var(--text-primary)", fontSize: 12,
            maxWidth: 460,
          }}
        >
          {list.data.map((a) => (
            <option key={`${a.game_pk}:${a.at_bat_number}`} value={`${a.game_pk}:${a.at_bat_number}`}>
              {labelAtBat(a)}
            </option>
          ))}
        </select>
        <div style={{ display: "flex", border: "1px solid var(--gridline)", borderRadius: 4 }}>
          {[true, false].map((all) => (
            <button
              key={String(all)}
              onClick={() => setShowAll(all)}
              style={{
                background: showAll === all ? "var(--gridline)" : "none",
                border: "none", padding: "3px 9px", fontSize: 11, cursor: "pointer",
                color: showAll === all ? "var(--text-primary)" : "var(--text-muted)",
              }}
            >
              {all ? "whole at-bat" : "one pitch"}
            </button>
          ))}
        </div>
      </div>

      {detail.isError && <Muted>Could not load that at-bat.</Muted>}
      {!ab ? (
        <Skeleton h={420} />
      ) : (
        <>
          <p style={{ margin: 0, fontSize: 12, color: "var(--text-secondary)" }}>
            {ab.pitcher_name ?? ab.pitcher} to {ab.batter_name ?? ab.batter}
            {ab.inning != null && ` · inning ${ab.inning}`} · {pitches.length} pitches ·{" "}
            <strong style={{ color: "var(--text-primary)" }}>
              {(ab.result ?? ab.result_description ?? "—").replace(/_/g, " ")}
            </strong>
            {ab.run_value != null && ` · ${ab.run_value > 0 ? "+" : ""}${ab.run_value.toFixed(2)} run value`}
          </p>

          <PitchSequence
            pitches={pitches}
            active={showAll ? selectedPitch : single}
            onSelect={(n) => setSelectedPitch(n === selectedPitch ? null : n)}
          />

          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-start" }}>
            {flyable.length ? (
              <PitchScene3D
                pitches={scenePitches}
                szTop={ab.sz_top}
                szBot={ab.sz_bot}
                stand={ab.stand}
                width={520}
                height={380}
                highlightId={selectedPitch != null ? `p${selectedPitch}` : null}
                loop
              />
            ) : (
              <Muted>
                No tracked physics for any pitch in this at-bat — the sequence is still above.
              </Muted>
            )}

            <div>
              {zone.data ? (
                <StrikeZoneHeatmap grid={zone.data} marks={marks} connectMarks />
              ) : zone.isError ? (
                <div style={{ width: 320 }}>
                  <Muted>
                    No zone grid for this batter{season ? ` in ${season}` : ""} — he may fall below
                    the qualifier. The pitch locations still plot against the rulebook zone in the
                    3D view.
                  </Muted>
                </div>
              ) : (
                <Skeleton h={380} />
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function PitchSequence({
  pitches, active, onSelect,
}: {
  pitches: AtBatPitch[];
  active: number | null;
  onSelect: (n: number) => void;
}) {
  return (
    <div style={{ display: "flex", gap: 6, overflowX: "auto", paddingBottom: 4 }}>
      {pitches.map((p) => {
        const tracked = hasFlight(p);
        const on = p.pitch_number === active;
        return (
          <button
            key={p.pitch_number}
            onClick={() => tracked && onSelect(p.pitch_number)}
            disabled={!tracked}
            title={tracked ? undefined : "No tracking data for this pitch"}
            style={{
              flex: "0 0 auto", minWidth: 92, textAlign: "left",
              padding: "6px 8px", borderRadius: 6,
              border: `1px solid ${on ? "var(--text-secondary)" : "var(--gridline)"}`,
              background: "none", cursor: tracked ? "pointer" : "default",
              opacity: tracked ? 1 : 0.5, font: "inherit", fontSize: 11,
              color: "var(--text-secondary)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <span
                style={{
                  width: 8, height: 8, borderRadius: "50%",
                  background: familyColor(familyOf(p.pitch_type)),
                }}
              />
              <span style={{ color: "var(--text-muted)" }}>
                #{p.pitch_number} · {p.balls}-{p.strikes}
              </span>
            </div>
            <div style={{ color: "var(--text-primary)", fontWeight: 600, marginTop: 2 }}>
              {labelOf(p.pitch_type)}
            </div>
            <div>{number(p.release_speed, 1)} mph</div>
            <div style={{ color: "var(--text-muted)" }}>{outcomeOf(p) || "—"}</div>
          </button>
        );
      })}
    </div>
  );
}

function Muted({ children }: { children: React.ReactNode }) {
  return <p style={{ color: "var(--text-muted)", fontSize: 13, margin: 0 }}>{children}</p>;
}

function Skeleton({ h }: { h: number }) {
  return <div style={{ height: h, background: "var(--gridline)", borderRadius: 4, opacity: 0.4 }} />;
}
