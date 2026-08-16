/** The one filter row. Sits above the charts; every chart subscribes to it. */

import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { ZONE_METRICS } from "../lib/scales";
import { useFilters } from "../store/filters";

export function FilterBar() {
  const { season, metric, vsHand, role, setSeason, setMetric, setVsHand, setRole } = useFilters();
  const { data: seasons } = useQuery({ queryKey: ["seasons"], queryFn: api.seasons });

  return (
    <div
      style={{
        display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap",
        padding: "10px 16px", background: "var(--surface-1)",
        border: "1px solid var(--border)", borderRadius: "var(--radius)",
      }}
    >
      <Field label="Season">
        <select value={season ?? ""} onChange={(e) => setSeason(e.target.value ? +e.target.value : null)}>
          <option value="">All</option>
          {seasons?.map((s) => (
            <option key={s.season} value={s.season}>
              {s.season} ({s.pitches.toLocaleString()})
            </option>
          ))}
        </select>
      </Field>

      <Field label="View as">
        <select value={role} onChange={(e) => setRole(e.target.value as "batter" | "pitcher")}>
          <option value="pitcher">Pitcher</option>
          <option value="batter">Batter</option>
        </select>
      </Field>

      <Field label="Zone metric">
        <select value={metric} onChange={(e) => setMetric(e.target.value)}>
          {Object.values(ZONE_METRICS).map((m) => (
            <option key={m.key} value={m.key}>{m.label}</option>
          ))}
        </select>
      </Field>

      <Field label="vs. batter hand">
        <select value={vsHand ?? ""} onChange={(e) => setVsHand((e.target.value || null) as "L" | "R" | null)}>
          <option value="">Both</option>
          <option value="R">RHB</option>
          <option value="L">LHB</option>
        </select>
      </Field>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 2, fontSize: 11 }}>
      <span style={{ color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.03em" }}>
        {label}
      </span>
      {children}
    </label>
  );
}
