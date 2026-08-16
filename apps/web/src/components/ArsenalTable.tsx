/**
 * Pitch arsenal with league percentile bars.
 *
 * This also serves as the table view the palette's contrast relief requires: the
 * numbers are readable without relying on any colour channel.
 */

import type { ArsenalRow } from "../lib/api";
import { familyColor, familyOf, labelOf, number } from "../lib/scales";

interface Props {
  rows: ArsenalRow[];
  onSelect?: (pitchType: string | null) => void;
  selected?: string | null;
}

export function ArsenalTable({ rows, onSelect, selected }: Props) {
  if (!rows.length) return <p style={{ color: "var(--text-muted)" }}>No arsenal data.</p>;

  return (
    <table>
      <thead>
        <tr>
          <th>Pitch</th>
          <th>Usage</th>
          <th>Velo</th>
          <th>IVB</th>
          <th>HB</th>
          <th>Whiff%</th>
          <th>CSW%</th>
          <th>RV/100</th>
          <th style={{ width: 90 }}>CSW pct</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
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
              <td>{number(r.velo_avg, 1)}</td>
              <td>{number(r.ivb_in, 1)}</td>
              <td>{number(r.hb_arm_in, 1)}</td>
              <td>{number(r.whiff_pct, 1)}</td>
              <td>{number(r.csw_pct, 1)}</td>
              <td>{number(r.rv_per_100, 2)}</td>
              <td>
                <PercentileBar value={r.pct_csw} />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/**
 * Percentile bar. Null when the sample is too small to rank — the mart withholds
 * percentiles below 50 pitches rather than reporting a 99th-percentile grade off
 * six pitches.
 */
function PercentileBar({ value }: { value: number | null }) {
  if (value == null) {
    return <span style={{ color: "var(--text-muted)", fontSize: 11 }}>—</span>;
  }
  const pct = Math.round(value * 100);
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span
        style={{
          flex: 1, height: 6, background: "var(--gridline)",
          borderRadius: 3, overflow: "hidden",
        }}
      >
        <span
          style={{
            display: "block", width: `${pct}%`, height: "100%",
            background: "var(--family-fastball)", borderRadius: 3,
          }}
        />
      </span>
      <span style={{ fontSize: 11, color: "var(--text-secondary)", minWidth: 18 }}>{pct}</span>
    </span>
  );
}
