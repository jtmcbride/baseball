/**
 * Swing-path plane value — is this batter's swing plane good against what he
 * actually sees?
 *
 * Both numbers are the same counterfactual (`SwingPathModel.plane_value`):
 * every swing scored twice, once at the batter's actual attack angle and once
 * at a matched league-median swing, with the pitch itself held fixed. Zero
 * means "no different from a neutral swing against these pitches" — the
 * meaningful reference point is 0, not some league-average bar, which is why
 * these render as diverging bars around zero rather than filled-from-zero
 * ones like the arsenal table's percentiles.
 */

import { type SwingRow } from "../lib/api";
import { divergingColor, number } from "../lib/scales";

interface Props {
  row: SwingRow;
}

export function SwingPanel({ row }: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <Headline
          label="Attack angle"
          hint="swing plane, degrees"
          value={`${number(row.attack_angle, 1)}°`}
        />
        <PlaneCard
          label="Whiffs avoided / 100"
          hint={`vs. a league-median plane · ${row.whiff_swings.toLocaleString()} swings`}
          value={row.whiff_plane_value_per_100}
          halfRange={15}
        />
        <PlaneCard
          label="Contact value / 100"
          hint={
            row.contact_swings
              ? `xwOBA pts on contact · ${row.contact_swings.toLocaleString()} swings`
              : "not enough contact swings this season"
          }
          value={row.contact_plane_value_per_100}
          halfRange={10}
        />
      </div>

      <p style={{ margin: 0, fontSize: 11, color: "var(--text-muted)" }}>
        Positive is better for the batter in both heads. Each swing is compared against a version
        of itself with the same pitch but a neutral, league-median swing plane — a swing plane is
        not good or bad on its own, only relative to the pitch it met.
      </p>
    </div>
  );
}

function Headline({ label, hint, value }: { label: string; hint: string; value: string }) {
  return (
    <div
      style={{
        flex: "1 1 120px",
        padding: "8px 10px",
        borderRadius: "var(--radius)",
        border: "1px solid var(--border)",
      }}
    >
      <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>{label}</div>
      <div style={{ fontSize: 22, fontVariantNumeric: "tabular-nums" }}>{value}</div>
      <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{hint}</div>
    </div>
  );
}

function PlaneCard({
  label,
  hint,
  value,
  halfRange,
}: {
  label: string;
  hint: string;
  value: number | null;
  halfRange: number;
}) {
  const known = value != null && Number.isFinite(value);
  return (
    <div
      style={{
        flex: "1 1 160px",
        padding: "8px 10px",
        borderRadius: "var(--radius)",
        border: "1px solid var(--border)",
        borderLeft: `3px solid ${known ? divergingColor(value, 0, halfRange) : "var(--border)"}`,
      }}
    >
      <div style={{ fontSize: 11, color: "var(--text-secondary)" }}>{label}</div>
      <div style={{ fontSize: 22, fontVariantNumeric: "tabular-nums" }}>
        {known ? (
          <>
            {value > 0 ? "+" : ""}
            {number(value, 2)}
          </>
        ) : (
          <span style={{ color: "var(--text-muted)" }}>—</span>
        )}
      </div>
      <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{hint}</div>
    </div>
  );
}
