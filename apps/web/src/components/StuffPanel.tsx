/**
 * Stuff+ / Location+ / Pitching+ — the model-graded arsenal.
 *
 * Three grades of the same pitches on the same 100-is-average scale, which is
 * the entire point: the interesting reading is never one number, it is the gap
 * between two. Elite shape with mediocre placement (Stuff+ 115 / Location+ 94)
 * and the reverse are completely different pitchers, and a single "pitch
 * quality" number hides which one you are looking at.
 *
 * Bars are diverging from 100 rather than filled from zero. A Stuff+ of 96 is
 * slightly below average, not "96% of a good pitch", and a zero-based bar would
 * render the entire league as a row of nearly identical near-full bars.
 */

import { ALL_PITCHES, type StuffRow } from "../lib/api";
import { divergingColor, familyColor, familyOf, labelOf, number } from "../lib/scales";

/** Saturates the ramp at +/- this many points. ~2.5 SD of a pitch-type grade. */
const HALF_RANGE = 25;
const CENTRE = 100;

const GRADES = [
  { key: "stuff_plus", label: "Stuff+", hint: "shape alone — velocity, movement, release" },
  { key: "location_plus", label: "Location+", hint: "placement and count alone" },
  { key: "pitching_plus", label: "Pitching+", hint: "both together" },
] as const;

interface Props {
  rows: StuffRow[];
  onSelect?: (pitchType: string | null) => void;
  selected?: string | null;
}

export function StuffPanel({ rows, onSelect, selected }: Props) {
  if (!rows.length) return <p style={{ color: "var(--text-muted)" }}>No graded pitches.</p>;

  const overall = rows.find((r) => r.pitch_type === ALL_PITCHES);
  const byType = rows.filter((r) => r.pitch_type !== ALL_PITCHES);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {overall && (
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
          {GRADES.map((g) => (
            <Headline key={g.key} label={g.label} hint={g.hint} value={overall[g.key]} />
          ))}
        </div>
      )}

      <table>
        <thead>
          <tr>
            <th>Pitch</th>
            <th>Usage</th>
            {GRADES.map((g) => (
              <th key={g.key} style={{ minWidth: 116 }}>
                {g.label}
              </th>
            ))}
            <th>RV/100</th>
          </tr>
        </thead>
        <tbody>
          {byType.map((r) => {
            const active = selected === r.pitch_type;
            return (
              <tr
                key={r.pitch_type}
                onClick={() => onSelect?.(active ? null : r.pitch_type)}
                style={{
                  cursor: onSelect ? "pointer" : "default",
                  background: active ? "var(--gridline)" : undefined,
                }}
              >
                <td>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    <svg width={8} height={8} aria-hidden>
                      <circle cx={4} cy={4} r={4} fill={familyColor(familyOf(r.pitch_type))} />
                    </svg>
                    {labelOf(r.pitch_type)}
                  </span>
                </td>
                <td>{number(r.usage_pct, 1)}%</td>
                {GRADES.map((g) => (
                  <td key={g.key}>
                    <PlusBar value={r[g.key]} />
                  </td>
                ))}
                <td>{number(r.rv_per_100, 2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <p style={{ margin: 0, fontSize: 11, color: "var(--text-muted)" }}>
        100 is league average, 10 points is one standard deviation, higher is better for the
        pitcher. Graded from a count-neutral run value with ball-in-play luck removed, so the
        grade and the RV/100 beside it deliberately disagree — that gap is the pitcher's luck.
      </p>
    </div>
  );
}

function Headline({ label, hint, value }: { label: string; hint: string; value: number }) {
  return (
    <div
      style={{
        flex: "1 1 120px",
        padding: "8px 10px",
        borderRadius: "var(--radius)",
        border: "1px solid var(--border)",
        // A tint rather than a fill: this is a summary, not a heatmap cell, and
        // a saturated block behind a number hurts its contrast.
        borderLeft: `3px solid ${divergingColor(value, CENTRE, HALF_RANGE)}`,
      }}
    >
      <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>{label}</div>
      <div style={{ fontSize: 22, fontVariantNumeric: "tabular-nums" }}>{number(value, 1)}</div>
      <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{hint}</div>
    </div>
  );
}

/** Diverging bar: fills left or right from a fixed centre tick at 100. */
function PlusBar({ value }: { value: number | null }) {
  if (value == null || !Number.isFinite(value)) {
    return <span style={{ color: "var(--text-muted)", fontSize: 11 }}>—</span>;
  }
  const t = Math.max(-1, Math.min(1, (value - CENTRE) / HALF_RANGE));
  const half = Math.abs(t) * 50;
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span
        style={{
          position: "relative", flex: 1, height: 8, minWidth: 56,
          background: "var(--gridline)", borderRadius: 2,
        }}
      >
        <span
          style={{
            position: "absolute", top: 0, bottom: 0,
            left: t >= 0 ? "50%" : `${50 - half}%`,
            width: `${half}%`,
            background: divergingColor(value, CENTRE, HALF_RANGE),
            borderRadius: 2,
          }}
        />
        {/* The average tick stays visible through the bar — without it a long
            fill and a short one are hard to compare across rows. */}
        <span
          style={{
            position: "absolute", left: "50%", top: -2, bottom: -2,
            width: 1, background: "var(--axis)",
          }}
        />
      </span>
      <span
        style={{
          fontSize: 11, minWidth: 28, textAlign: "right",
          fontVariantNumeric: "tabular-nums", color: "var(--text-secondary)",
        }}
      >
        {number(value, 0)}
      </span>
    </span>
  );
}
