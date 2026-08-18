/**
 * UMAP arsenal map (viz #12) -- every pitcher-season embedded and clustered,
 * plus "who does this pitcher resemble?" (model #11).
 *
 * Standalone tab like `UmpiresPage` rather than a `PlayerPage` panel: the
 * whole point is browsing across pitchers, not one pitcher's own page. The
 * global season filter narrows the cloud; leaving it unset (the default)
 * shows every pitcher-season in the mart at once.
 */

import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { ArsenalClusterPanel } from "../components/ArsenalClusterPanel";
import { ArsenalMap } from "../components/ArsenalMap";
import { api, type ArsenalEmbeddingRow } from "../lib/api";
import { ARSENAL_METRICS } from "../lib/scales";
import { useFilters } from "../store/filters";

export function ArsenalMapPage() {
  const season = useFilters((s) => s.season);
  const [selected, setSelected] = useState<ArsenalEmbeddingRow | null>(null);
  const [colorBy, setColorBy] = useState<string>("arsenal_size_diff");

  const embedding = useQuery({
    queryKey: ["arsenal-embedding", season],
    queryFn: () => api.arsenalEmbedding({ season: season ?? undefined }),
  });

  // Selection can go stale when the season filter changes under it (the
  // point may no longer be in the filtered cloud) -- clear rather than show
  // a detail pane for a point that's no longer plotted.
  useEffect(() => {
    setSelected(null);
  }, [season]);

  const clusters = useQuery({
    queryKey: ["arsenal-clusters", selected?.mlbam_id, selected?.season],
    queryFn: () => api.arsenalClusters(selected!.mlbam_id, selected!.season),
    enabled: !!selected,
    retry: false,
  });

  const similar = useQuery({
    queryKey: ["arsenal-similar", selected?.mlbam_id, selected?.season],
    queryFn: () => api.similarPitchers(selected!.mlbam_id, selected!.season, 8),
    enabled: !!selected,
    retry: false,
  });

  return (
    <div style={{ display: "grid", gap: 16, gridTemplateColumns: "minmax(420px,2fr) minmax(340px,1fr)" }}>
      <section className="card">
        <h3>Arsenal map{season ? ` · ${season}` : " · all seasons"}</h3>
        <p className="subtitle">
          Every qualifying pitcher-season, re-derived from physical pitch shape and reduced to 2D.
          Scroll to zoom, drag to pan, click a point. Outlined regions are archetypes; a submariner or
          knuckleballer should sit far from everyone else.
        </p>
        <div style={{ marginBottom: 8 }}>
          <label style={{ fontSize: 11, color: "var(--text-muted)" }}>
            Colour by{" "}
            <select value={colorBy} onChange={(e) => setColorBy(e.target.value)}>
              {Object.values(ARSENAL_METRICS).map((m) => (
                <option key={m.key} value={m.key}>{m.label}</option>
              ))}
            </select>
          </label>
        </div>
        {embedding.isLoading ? (
          <Skeleton h={520} />
        ) : embedding.data && embedding.data.length > 0 ? (
          // Keyed on season so a filter change fully remounts the map and
          // refits the viewport to the new (possibly much smaller) cloud,
          // rather than leaving a stale pan/zoom over emptier space.
          <ArsenalMap
            key={season ?? "all"}
            points={embedding.data}
            metric={colorBy}
            selectedId={selected?.mlbam_id}
            selectedSeason={selected?.season}
            onSelect={setSelected}
          />
        ) : (
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
            No embedded pitcher-seasons — run <code>bb-ml arsenal-embed</code>.
          </p>
        )}
      </section>

      <section className="card">
        <h3>{selected ? `Pitcher ${selected.mlbam_id} · ${selected.season}` : "Pick a point"}</h3>
        <p className="subtitle">
          {selected
            ? `${selected.archetype_label} archetype · re-derived ${selected.cluster_k} pitches vs. Savant's ${selected.savant_pitch_types}`
            : "Click any point on the map to see its re-derived arsenal and nearest neighbors."}
        </p>
        {!selected ? null : clusters.isError ? (
          <p style={{ color: "var(--text-muted)", fontSize: 13 }}>No cluster detail for this pitcher-season.</p>
        ) : clusters.data ? (
          <ArsenalClusterPanel rows={clusters.data} similar={similar.data} />
        ) : (
          <Skeleton h={300} />
        )}
      </section>
    </div>
  );
}

function Skeleton({ h }: { h: number }) {
  return <div style={{ height: h, background: "var(--gridline)", borderRadius: 4, opacity: 0.4 }} />;
}
