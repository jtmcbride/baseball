/**
 * Model #2's re-derived arsenal, next to `ArsenalTable`'s Savant-labelled one.
 *
 * The two tables are meant to sit side by side on the player page: one reads
 * Savant's own `pitch_type`, the other re-derives shape from scratch and
 * grades where they agree. `arsenal_size_diff` is the one-number headline
 * (positive = a real split Savant missed, negative = a merge Savant should
 * have made); `season_purity` is how cleanly each re-derived cluster maps
 * back onto one Savant label.
 */

import type { ArsenalClusterRow, SimilarPitcherRow } from "../lib/api";
import { divergingColor, familyColor, familyOf, labelOf, number } from "../lib/scales";

const HALF_RANGE = 4; // matches ARSENAL_METRICS.arsenal_size_diff

interface Props {
  rows: ArsenalClusterRow[];
  similar?: SimilarPitcherRow[];
}

export function ArsenalClusterPanel({ rows, similar }: Props) {
  if (!rows.length) return <p style={{ color: "var(--text-muted)" }}>No re-derived arsenal.</p>;

  const first = rows[0];
  const diff = first.arsenal_size_diff;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <Headline
          label="Re-derived vs. Savant"
          value={`${first.cluster_k} vs ${first.savant_pitch_types}`}
          diverge={divergingColor(diff, 0, HALF_RANGE)}
          hint={diff === 0 ? "exact agreement" : diff > 0 ? `+${diff}: Savant under-split` : `${diff}: Savant over-split`}
        />
        <Headline
          label="Season purity"
          value={`${(first.season_purity * 100).toFixed(0)}%`}
          diverge={divergingColor(first.season_purity, 0.85, 0.15)}
          hint="how cleanly clusters map to one Savant label"
        />
      </div>

      <div style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th>Cluster</th>
              <th>Usage</th>
              <th>Velo</th>
              <th>IVB</th>
              <th>HB</th>
              <th>Spin axis</th>
              <th>Savant majority</th>
              <th>Purity</th>
              <th># Savant labels</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.cluster_id}>
                <td>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 6, whiteSpace: "nowrap" }}>
                    <svg width={8} height={8} aria-hidden>
                      <circle cx={4} cy={4} r={4} fill={familyColor(familyOf(r.savant_majority))} />
                    </svg>
                    {r.label}
                  </span>
                </td>
                <td>{number(r.usage_pct, 1)}%</td>
                <td>{number(r.velo_avg, 1)}</td>
                <td>{number(r.ivb_in, 1)}</td>
                <td>{number(r.hb_arm_in, 1)}</td>
                <td>{number(r.spin_axis_arm_deg, 0)}°</td>
                <td style={{ whiteSpace: "nowrap" }}>{labelOf(r.savant_majority)}</td>
                <td>{(r.purity * 100).toFixed(0)}%</td>
                <td>{r.n_savant_labels}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {similar && similar.length > 0 && (
        <div>
          <h4 style={{ margin: "0 0 6px", fontSize: 13 }}>Who does this pitcher resemble?</h4>
          <ul style={{ margin: 0, padding: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 4 }}>
            {similar.map((s) => (
              <li
                key={`${s.neighbor_id}-${s.neighbor_season}`}
                style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}
              >
                <span>
                  {s.full_name ?? s.neighbor_id} <span style={{ color: "var(--text-muted)" }}>· {s.neighbor_season}</span>
                </span>
                <span style={{ color: "var(--text-secondary)" }}>{s.archetype_label ?? "—"}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <p style={{ margin: 0, fontSize: 11, color: "var(--text-muted)" }}>
        Re-derives each pitcher-season's arsenal from physical shape (velocity, movement, spin) and
        compares it back to Savant's own automated pitch_type labels. Positive vs. Savant means this
        model found a split Savant missed; negative means a merge. This never replaces pitch_type
        elsewhere in the app — it's a check on it.
      </p>
    </div>
  );
}

function Headline({
  label, value, diverge, hint,
}: {
  label: string;
  value: string;
  diverge: string;
  hint: string;
}) {
  return (
    <div
      style={{
        flex: "1 1 160px",
        padding: "8px 10px",
        borderRadius: "var(--radius)",
        border: "1px solid var(--border)",
        borderLeft: `3px solid ${diverge}`,
      }}
    >
      <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>{label}</div>
      <div style={{ fontSize: 20, fontVariantNumeric: "tabular-nums" }}>{value}</div>
      <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{hint}</div>
    </div>
  );
}
