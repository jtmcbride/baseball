import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../lib/api";
import { useFilters } from "../store/filters";

export function PlayerSearch() {
  const [q, setQ] = useState("");
  const setPlayer = useFilters((s) => s.setPlayer);
  const role = useFilters((s) => s.role);

  const { data } = useQuery({
    queryKey: ["search", q],
    queryFn: () => api.searchPlayers(q),
    enabled: q.length >= 2,
  });

  return (
    <div style={{ position: "relative", minWidth: 260 }}>
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Search a player…"
        style={{
          width: "100%", padding: "8px 10px", fontSize: 14,
          background: "var(--surface-1)", color: "var(--text-primary)",
          border: "1px solid var(--border)", borderRadius: "var(--radius)",
        }}
      />
      {q.length >= 2 && data && data.length > 0 && (
        <ul
          style={{
            position: "absolute", zIndex: 10, top: "100%", left: 0, right: 0,
            margin: "4px 0 0", padding: 0, listStyle: "none", maxHeight: 280,
            overflowY: "auto", background: "var(--surface-1)",
            border: "1px solid var(--border)", borderRadius: "var(--radius)",
          }}
        >
          {data.map((p) => (
            <li key={p.mlbam_id}>
              <button
                onClick={() => {
                  setPlayer(p.mlbam_id, p.primary_position === "P" ? "pitcher" : role);
                  setQ("");
                }}
                style={{
                  display: "block", width: "100%", textAlign: "left", padding: "7px 10px",
                  background: "none", border: "none", cursor: "pointer",
                  color: "var(--text-primary)", fontSize: 13,
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
