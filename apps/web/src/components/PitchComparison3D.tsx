/**
 * Pitch comparison: several flights in one 3D scene.
 *
 * Two questions, one instrument. Within a pitcher, the arsenal overlaid answers
 * "do these come out of the same tunnel?" — the reason a 92mph changeup gets
 * swung through is that for the first 35 feet it is indistinguishable from the
 * fastball. Across two pitchers, the same overlay answers "whose slider is
 * whose?" for pitches that a movement plot shows as two dots with no shared
 * frame of reference.
 *
 * What is drawn is a REAL pitch per type, not an average one — see the
 * `/pitches/arsenal-trajectories` docstring for why an averaged trajectory is a
 * path nobody threw. That has a consequence worth stating in the caption and
 * not burying here: the exemplars come from different plate appearances, so the
 * strike zone drawn behind them belongs to no single hitter. It is a reference
 * rectangle (the mean of those pitches' own measured zones), which is why the
 * commit-point readout, not the zone, carries the comparison's actual numbers.
 *
 * The measurements come from `lib/tunnel.ts`, which samples every flight at the
 * same DISTANCE from the plate rather than the same elapsed time — the
 * difference between measuring divergence and measuring velocity.
 */

import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { api, type ArsenalTrajectory, type PlayerSummary } from "../lib/api";
import { familyColor, familyOf, labelOf, number, pitchDash } from "../lib/scales";
import { reconstructFlight } from "../lib/trajectory";
import {
  COMMIT_DISTANCE_FT, plateSpreadIn, reactionMs, tunnelRatio, tunnelSpreadIn,
  type SyncMode,
} from "../lib/tunnel";
import { PitchScene3D, type ScenePitch } from "./PitchScene3D";

interface Props {
  pitcherId: number;
  pitcherName: string;
  season: number | null;
  vsHand: "L" | "R" | null;
}

interface Entry {
  key: string;
  row: ArsenalTrajectory;
  group: 0 | 1;
  ownerName: string;
}

function entriesOf(rows: ArsenalTrajectory[] | undefined, group: 0 | 1, ownerName: string): Entry[] {
  return (rows ?? []).map((row) => ({
    key: `${group}:${row.pitch_type ?? "UN"}`,
    row,
    group,
    ownerName,
  }));
}

export function PitchComparison3D({ pitcherId, pitcherName, season, vsHand }: Props) {
  const [opponent, setOpponent] = useState<PlayerSummary | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [highlight, setHighlight] = useState<string | null>(null);
  const [syncMode, setSyncMode] = useState<SyncMode>("release");
  const [commitDistance, setCommitDistance] = useState(COMMIT_DISTANCE_FT);

  // A held opponent from a previous pitcher would silently compare the new
  // pitcher against someone the reader chose for a different page.
  useEffect(() => setOpponent(null), [pitcherId]);

  const mine = useQuery({
    queryKey: ["arsenal-traj", pitcherId, season, vsHand],
    queryFn: () => api.arsenalTrajectories(pitcherId, season ?? undefined, vsHand),
    enabled: !!pitcherId,
    retry: false,
  });

  const theirs = useQuery({
    queryKey: ["arsenal-traj", opponent?.mlbam_id, season, vsHand],
    queryFn: () => api.arsenalTrajectories(opponent!.mlbam_id, season ?? undefined, vsHand),
    enabled: !!opponent,
    retry: false,
  });

  const entries = useMemo(
    () => [
      ...entriesOf(mine.data, 0, pitcherName),
      ...entriesOf(theirs.data, 1, opponent?.full_name ?? "Comparison"),
    ],
    [mine.data, theirs.data, pitcherName, opponent],
  );

  // Default selection. One pitcher: the whole arsenal, which is the question
  // ("how does his stuff tunnel?"). Two pitchers: the primary pitch of each,
  // because eight lines from two arm slots is a tangle nobody can read — the
  // reader adds back what they want to compare.
  const defaultKey = useMemo(
    () =>
      opponent
        ? [mine.data?.[0], theirs.data?.[0]]
            .map((r, i) => (r ? `${i}:${r.pitch_type ?? "UN"}` : null))
            .filter((k): k is string => k != null)
        : entries.map((e) => e.key),
    [entries, mine.data, theirs.data, opponent],
  );
  const defaultSignature = defaultKey.join("|");
  useEffect(() => {
    setSelected(new Set(defaultKey));
    // `defaultKey` is rebuilt every render; its contents are the real trigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultSignature]);

  const shown = useMemo(() => entries.filter((e) => selected.has(e.key)), [entries, selected]);

  const flights = useMemo(
    () => shown.map((e) => reconstructFlight(e.row)),
    [shown],
  );

  const scenePitches: ScenePitch[] = useMemo(
    () =>
      shown.map((e) => ({
        id: e.key,
        trajectory: e.row,
        pitchType: e.row.pitch_type,
        group: e.group,
      })),
    [shown],
  );

  // The zone here is a reference rectangle, not a hitter's — see the file
  // comment. Averaging the exemplars' own measured zones keeps it honest at the
  // scale of a real strike zone without implying it belongs to anybody.
  const szTop = shown.length ? shown.reduce((a, e) => a + e.row.sz_top, 0) / shown.length : 3.4;
  const szBot = shown.length ? shown.reduce((a, e) => a + e.row.sz_bot, 0) / shown.length : 1.6;

  const tunnel = flights.length > 1 ? tunnelSpreadIn(flights, commitDistance) : null;
  const plate = flights.length > 1 ? plateSpreadIn(flights) : null;
  const ratio = flights.length > 1 ? tunnelRatio(flights, commitDistance) : null;

  if (mine.isError) {
    return (
      <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
        No tracked pitch types for this pitcher{season ? ` in ${season}` : ""} — the physics fit
        is missing, or every type falls below the 15-pitch floor.
      </p>
    );
  }
  if (!mine.data) {
    return <div style={{ height: 380, background: "var(--gridline)", borderRadius: 4, opacity: 0.4 }} />;
  }

  return (
    <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-start" }}>
      <PitchScene3D
        pitches={scenePitches}
        szTop={szTop}
        szBot={szBot}
        stand={shown[0]?.row.stand ?? "R"}
        width={520}
        height={380}
        syncMode={syncMode}
        commitDistance={commitDistance}
        highlightId={highlight}
        loop
      >
        <SyncToggle mode={syncMode} onChange={setSyncMode} />
      </PitchScene3D>

      <div style={{ flex: "1 1 300px", minWidth: 280, display: "flex", flexDirection: "column", gap: 12 }}>
        <PitchLegend
          title={pitcherName}
          entries={entries.filter((e) => e.group === 0)}
          selected={selected}
          onToggle={(k) => setSelected(toggle(selected, k))}
          onHighlight={setHighlight}
        />

        {opponent && theirs.data && (
          <PitchLegend
            title={opponent.full_name}
            entries={entries.filter((e) => e.group === 1)}
            selected={selected}
            onToggle={(k) => setSelected(toggle(selected, k))}
            onHighlight={setHighlight}
          />
        )}
        {opponent && theirs.isError && (
          <p style={{ fontSize: 12, color: "var(--text-muted)", margin: 0 }}>
            No tracked pitch types for {opponent.full_name}
            {season ? ` in ${season}` : ""}.
          </p>
        )}

        <ComparePicker
          opponent={opponent}
          excludeId={pitcherId}
          onPick={setOpponent}
          onClear={() => setOpponent(null)}
        />

        <TunnelReadout
          commitDistance={commitDistance}
          onCommitDistance={setCommitDistance}
          reaction={flights.length ? Math.min(...flights.map((f) => reactionMs(f, commitDistance))) : null}
          tunnel={tunnel}
          plate={plate}
          ratio={ratio}
          n={flights.length}
        />
      </div>
    </div>
  );
}

function toggle(set: Set<string>, key: string): Set<string> {
  const next = new Set(set);
  if (!next.delete(key)) next.add(key);
  return next;
}

function PitchLegend({
  title, entries, selected, onToggle, onHighlight,
}: {
  title: string;
  entries: Entry[];
  selected: Set<string>;
  onToggle: (key: string) => void;
  onHighlight: (key: string | null) => void;
}) {
  if (!entries.length) return null;
  const group = entries[0].group;
  return (
    <div>
      <div
        style={{
          display: "flex", alignItems: "center", gap: 6, fontSize: 12,
          color: "var(--text-secondary)", marginBottom: 4,
        }}
      >
        <GroupGlyph group={group} />
        <strong style={{ color: "var(--text-primary)" }}>{title}</strong>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {entries.map((e) => {
          const on = selected.has(e.key);
          return (
            <button
              key={e.key}
              onClick={() => onToggle(e.key)}
              onMouseEnter={() => onHighlight(e.key)}
              onMouseLeave={() => onHighlight(null)}
              style={{
                display: "grid",
                gridTemplateColumns: "28px 1fr auto auto",
                alignItems: "center",
                gap: 8,
                padding: "3px 6px",
                background: "none",
                border: "1px solid transparent",
                borderRadius: 4,
                cursor: "pointer",
                textAlign: "left",
                font: "inherit",
                fontSize: 12,
                color: on ? "var(--text-primary)" : "var(--text-muted)",
                opacity: on ? 1 : 0.55,
              }}
            >
              <LinePreview pitchType={e.row.pitch_type} />
              <span>{labelOf(e.row.pitch_type)}</span>
              <span style={{ fontVariantNumeric: "tabular-nums" }}>
                {number(e.row.velo_avg, 1)} mph
              </span>
              <span style={{ fontVariantNumeric: "tabular-nums", color: "var(--text-muted)" }}>
                {number(e.row.usage_pct, 0)}%
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** The non-colour channel for "whose pitch is this" — matches the 3D markers. */
function GroupGlyph({ group }: { group: 0 | 1 }) {
  return (
    <svg width={12} height={12} aria-hidden>
      {group === 1 ? (
        <path d="M6,1 L11,6 L6,11 L1,6 Z" fill="var(--text-secondary)" />
      ) : (
        <circle cx={6} cy={6} r={4.5} fill="var(--text-secondary)" />
      )}
    </svg>
  );
}

/** Hue = family, dash = pitch type — the same encoding the 3D line uses. */
function LinePreview({ pitchType }: { pitchType: string | null }) {
  const dash = pitchDash(pitchType);
  return (
    <svg width={26} height={10} aria-hidden>
      <line
        x1={1} y1={5} x2={25} y2={5}
        stroke={familyColor(familyOf(pitchType))}
        strokeWidth={2.5}
        strokeLinecap="round"
        // Dash sizes are in feet in the 3D scene; scaled to pixels here purely
        // so the legend's pattern reads as the same pattern.
        strokeDasharray={dash ? `${dash[0] * 6} ${dash[1] * 6}` : undefined}
      />
    </svg>
  );
}

function SyncToggle({ mode, onChange }: { mode: SyncMode; onChange: (m: SyncMode) => void }) {
  return (
    <div style={{ display: "flex", gap: 0, border: "1px solid var(--gridline)", borderRadius: 4 }}>
      {(["release", "plate"] as const).map((m) => (
        <button
          key={m}
          onClick={() => onChange(m)}
          title={
            m === "release"
              ? "Every ball leaves the hand at once — what the hitter sees."
              : "Every ball crosses the plate at once — velocity removed, shape only."
          }
          style={{
            background: mode === m ? "var(--gridline)" : "none",
            border: "none", padding: "2px 8px", fontSize: 11, cursor: "pointer",
            color: mode === m ? "var(--text-primary)" : "var(--text-muted)",
          }}
        >
          {m === "release" ? "sync release" : "sync plate"}
        </button>
      ))}
    </div>
  );
}

function ComparePicker({
  opponent, excludeId, onPick, onClear,
}: {
  opponent: PlayerSummary | null;
  excludeId: number;
  onPick: (p: PlayerSummary) => void;
  onClear: () => void;
}) {
  const [q, setQ] = useState("");
  const { data } = useQuery({
    queryKey: ["search", q],
    queryFn: () => api.searchPlayers(q),
    enabled: q.length >= 2,
  });

  if (opponent) {
    return (
      <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
        comparing against <strong style={{ color: "var(--text-primary)" }}>{opponent.full_name}</strong>{" "}
        <button
          onClick={onClear}
          style={{
            background: "none", border: "none", padding: 0, cursor: "pointer",
            color: "var(--family-fastball)", font: "inherit",
          }}
        >
          ✕
        </button>
      </div>
    );
  }

  return (
    <div style={{ position: "relative" }}>
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Compare with another pitcher…"
        style={{
          width: "100%", padding: "6px 8px", fontSize: 12,
          background: "var(--surface-1)", color: "var(--text-primary)",
          border: "1px solid var(--border)", borderRadius: "var(--radius)",
        }}
      />
      {q.length >= 2 && data && data.length > 0 && (
        <ul
          style={{
            position: "absolute", zIndex: 10, top: "100%", left: 0, right: 0,
            margin: "4px 0 0", padding: 0, listStyle: "none", maxHeight: 220,
            overflowY: "auto", background: "var(--surface-1)",
            border: "1px solid var(--border)", borderRadius: "var(--radius)",
          }}
        >
          {data
            .filter((p) => p.mlbam_id !== excludeId)
            .map((p) => (
              <li key={p.mlbam_id}>
                <button
                  onClick={() => {
                    onPick(p);
                    setQ("");
                  }}
                  style={{
                    display: "block", width: "100%", textAlign: "left", padding: "6px 8px",
                    background: "none", border: "none", cursor: "pointer",
                    color: "var(--text-primary)", fontSize: 12,
                  }}
                >
                  {p.full_name}
                  <span style={{ color: "var(--text-muted)", marginLeft: 8 }}>
                    {p.primary_position} · {p.throws ?? "?"}HP
                  </span>
                </button>
              </li>
            ))}
        </ul>
      )}
    </div>
  );
}

function TunnelReadout({
  commitDistance, onCommitDistance, reaction, tunnel, plate, ratio, n,
}: {
  commitDistance: number;
  onCommitDistance: (d: number) => void;
  reaction: number | null;
  tunnel: number | null;
  plate: number | null;
  ratio: number | null;
  n: number;
}) {
  return (
    <div style={{ fontSize: 12, color: "var(--text-secondary)", borderTop: "1px solid var(--gridline)", paddingTop: 8 }}>
      <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <span style={{ fontSize: 10, color: "var(--text-muted)", whiteSpace: "nowrap" }}>
          commit point
        </span>
        <input
          type="range"
          min={10}
          max={40}
          step={0.2}
          value={commitDistance}
          onChange={(e) => onCommitDistance(Number(e.target.value))}
          style={{ flex: 1 }}
        />
        <span style={{ fontSize: 10, color: "var(--text-muted)", width: 74, textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
          {commitDistance.toFixed(1)} ft
          {reaction != null && ` · ${reaction.toFixed(0)}ms`}
        </span>
      </label>

      {n < 2 ? (
        <p style={{ margin: 0, color: "var(--text-muted)" }}>
          Select two or more pitches to measure how far apart they are when the hitter commits.
        </p>
      ) : (
        <>
          <dl style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "2px 10px", margin: 0 }}>
            <dt>Widest gap at the commit point</dt>
            <dd style={{ margin: 0, fontVariantNumeric: "tabular-nums", color: "var(--text-primary)" }}>
              {tunnel!.toFixed(1)}″
            </dd>
            <dt>Widest gap at the plate</dt>
            <dd style={{ margin: 0, fontVariantNumeric: "tabular-nums", color: "var(--text-primary)" }}>
              {plate!.toFixed(1)}″
            </dd>
            <dt>Late separation</dt>
            <dd style={{ margin: 0, fontVariantNumeric: "tabular-nums", color: "var(--text-primary)" }}>
              {ratio == null ? "—" : `${ratio.toFixed(1)}×`}
            </dd>
          </dl>
          <p style={{ margin: "6px 0 0", color: "var(--text-muted)" }}>
            Measured at the same distance out, not the same instant — a shared clock would
            measure the velocity difference instead. The commit point itself is an assumption
            ({COMMIT_DISTANCE_FT}ft ≈ 167ms on a mid-90s fastball), which is why it is a slider.
          </p>
        </>
      )}
    </div>
  );
}
