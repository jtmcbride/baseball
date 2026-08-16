/**
 * At-bat replay: each pitch as a small multiple, actual next to predicted.
 *
 * This is where the ML and the viz meet. Each card shows the same zone frame
 * as the heatmap (so a reader's eye carries over) with the actual pitch as a
 * dot, and the model's top pitch-type calls as small ranked bars below it —
 * the prediction was made from state strictly before this pitch, never from
 * its own outcome.
 */

import { useState } from "react";
import type { ReplayPitch } from "../lib/api";
import { familyColor, familyOf, labelOf } from "../lib/scales";

const PLATE_HALF_FT = 0.83;
// Matches bbml.features.schema: LOC_X_MIN/MAX, LOC_Z_MIN/MAX.
const LOC_X_MIN = -1.5,
  LOC_X_MAX = 1.5;
const LOC_Z_MIN = -0.5,
  LOC_Z_MAX = 1.5;

const THUMB_W = 64;
const THUMB_H = 76;

function xScale(x: number) {
  return ((x - LOC_X_MIN) / (LOC_X_MAX - LOC_X_MIN)) * THUMB_W;
}
function zScale(z: number) {
  return THUMB_H - ((z - LOC_Z_MIN) / (LOC_Z_MAX - LOC_Z_MIN)) * THUMB_H;
}

function ZoneThumb({ pitch }: { pitch: ReplayPitch }) {
  const hasLoc = pitch.actual_plate_x != null && pitch.actual_plate_z_norm != null;
  const color = familyColor(familyOf(pitch.actual_pitch_type));
  return (
    <svg width={THUMB_W} height={THUMB_H} role="img" aria-label="Actual pitch location">
      <rect
        x={xScale(-PLATE_HALF_FT)}
        y={zScale(1)}
        width={xScale(PLATE_HALF_FT) - xScale(-PLATE_HALF_FT)}
        height={zScale(0) - zScale(1)}
        fill="none"
        stroke="var(--gridline)"
        strokeWidth={1}
      />
      {hasLoc && (
        <circle
          cx={xScale(pitch.actual_plate_x!)}
          cy={zScale(pitch.actual_plate_z_norm!)}
          r={4}
          fill={color}
          stroke="var(--surface-1)"
          strokeWidth={1}
        />
      )}
    </svg>
  );
}

function PredictionBars({ pitch }: { pitch: ReplayPitch }) {
  const top = pitch.predicted_pitch_type.slice(0, 3);
  const max = top[0]?.probability ?? 1;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2, width: THUMB_W }}>
      {top.map((p) => {
        const hit = p.pitch_type === pitch.actual_pitch_type;
        return (
          <div
            key={p.pitch_type}
            title={`${labelOf(p.pitch_type)} · ${(p.probability * 100).toFixed(0)}%`}
            style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10 }}
          >
            <span
              style={{
                width: 18,
                color: hit ? "var(--text-primary)" : "var(--text-muted)",
                fontWeight: hit ? 700 : 400,
              }}
            >
              {p.pitch_type}
            </span>
            <span
              style={{
                display: "block",
                height: 5,
                borderRadius: 2,
                width: Math.max(2, (p.probability / max) * (THUMB_W - 22)),
                background: familyColor(familyOf(p.pitch_type)),
                opacity: hit ? 1 : 0.45,
              }}
            />
          </div>
        );
      })}
    </div>
  );
}

function PitchCard({ pitch }: { pitch: ReplayPitch }) {
  const top1 = pitch.predicted_pitch_type[0];
  const correct = top1?.pitch_type === pitch.actual_pitch_type;
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 6,
        padding: 8,
        borderRadius: 6,
        border: `1px solid ${correct ? "var(--text-secondary)" : "var(--gridline)"}`,
        minWidth: THUMB_W + 16,
        flex: "0 0 auto",
      }}
    >
      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
        {pitch.balls}-{pitch.strikes}
      </div>
      <ZoneThumb pitch={pitch} />
      <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-primary)" }}>
        {labelOf(pitch.actual_pitch_type)}
      </div>
      <PredictionBars pitch={pitch} />
      {correct && (
        <div style={{ fontSize: 9, color: "var(--text-secondary)" }}>top pick ✓</div>
      )}
    </div>
  );
}

interface AtBatGroup {
  at_bat_number: number;
  pitches: ReplayPitch[];
}

function groupByAtBat(pitches: ReplayPitch[]): AtBatGroup[] {
  const groups = new Map<number, ReplayPitch[]>();
  for (const p of pitches) {
    const g = groups.get(p.at_bat_number) ?? [];
    g.push(p);
    groups.set(p.at_bat_number, g);
  }
  return [...groups.entries()]
    .sort(([a], [b]) => a - b)
    .map(([at_bat_number, ps]) => ({ at_bat_number, pitches: ps }));
}

export function ReplayStrip({ pitches }: { pitches: ReplayPitch[] }) {
  const [expanded, setExpanded] = useState<number | null>(null);
  const atBats = groupByAtBat(pitches);
  const hits = pitches.filter((p) => p.predicted_pitch_type[0]?.pitch_type === p.actual_pitch_type).length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <p style={{ margin: 0, fontSize: 12, color: "var(--text-secondary)" }}>
        Model's top pick matched the actual pitch on {hits} of {pitches.length} pitches (
        {((hits / Math.max(pitches.length, 1)) * 100).toFixed(0)}%). Bold label + ring means the
        top-ranked call was right.
      </p>
      {atBats.map((ab) => {
        const isOpen = expanded === null || expanded === ab.at_bat_number;
        return (
          <div key={ab.at_bat_number}>
            <button
              onClick={() => setExpanded(expanded === ab.at_bat_number ? null : ab.at_bat_number)}
              style={{
                background: "none",
                border: "none",
                padding: 0,
                marginBottom: 4,
                cursor: "pointer",
                font: "inherit",
                fontSize: 12,
                color: "var(--text-muted)",
              }}
            >
              at-bat {ab.at_bat_number} · {ab.pitches.length} pitches
            </button>
            {isOpen && (
              <div style={{ display: "flex", gap: 8, overflowX: "auto", paddingBottom: 4 }}>
                {ab.pitches
                  .sort((a, b) => a.pitch_number - b.pitch_number)
                  .map((p) => (
                    <PitchCard key={p.pitch_number} pitch={p} />
                  ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
