/**
 * UMAP arsenal map (viz #12): every pitcher-season embedded and clustered.
 *
 * Templated on `MovementPlot.tsx` — same "colour by continuous value, label
 * groups directly" idea, adapted for this map's own constraints:
 *
 *  - Point colour is `divergingColor` on the selected `ARSENAL_METRICS` key,
 *    not pitch family — the all-pairs CVD gate governs a scatter's point
 *    colour regardless of what it encodes, and the validated diverging ramp
 *    already clears that gate for every other chart in the app.
 *  - Archetypes (KMeans on the embedding's feature space, not the 2D
 *    coordinates) get a faint convex-hull outline plus a direct centroid
 *    label, MovementPlot's own precedent for group identity that isn't
 *    colour.
 *  - Pan/zoom is a single `<g transform>` around the points group, not a
 *    per-point screen-position recompute -- at ~4,200 points recomputing on
 *    every drag/wheel frame would be the actual perf cliff. Marker radius is
 *    counter-scaled so points don't balloon as you zoom in.
 */

import { useCallback, useMemo, useRef, useState } from "react";
import type { ArsenalEmbeddingRow } from "../lib/api";
import { convexHull, hullPath } from "../lib/hull";
import { ARSENAL_METRICS, DIVERGING_LEGEND_STOPS, divergingColor } from "../lib/scales";
import { fitViewport, nearestPoint, panBy, zoomAt, type Viewport } from "../lib/viewport";

interface Props {
  points: ArsenalEmbeddingRow[];
  metric: string;
  width?: number;
  height?: number;
  selectedId?: number | null;
  selectedSeason?: number | null;
  onSelect?: (row: ArsenalEmbeddingRow) => void;
}

const POINT_RADIUS = 2.6;

export function ArsenalMap({
  points, metric, width = 620, height = 520, selectedId, selectedSeason, onSelect,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const dragState = useRef<{ x: number; y: number } | null>(null);

  const bounds = useMemo(() => {
    if (points.length === 0) return { xMin: -1, xMax: 1, yMin: -1, yMax: 1 };
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    return { xMin: Math.min(...xs), xMax: Math.max(...xs), yMin: Math.min(...ys), yMax: Math.max(...ys) };
  }, [points]);

  const [vp, setVp] = useState<Viewport>(() => fitViewport(bounds, width, height));
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const def = ARSENAL_METRICS[metric] ?? ARSENAL_METRICS.arsenal_size_diff;

  const archetypes = useMemo(() => {
    const groups = new Map<number, { label: string; pts: ArsenalEmbeddingRow[] }>();
    for (const p of points) {
      const g = groups.get(p.archetype_id) ?? { label: p.archetype_label, pts: [] };
      g.pts.push(p);
      groups.set(p.archetype_id, g);
    }
    return [...groups.entries()].map(([id, g]) => {
      const hull = convexHull(g.pts.map((p) => ({ x: p.x, y: p.y })));
      const cx = g.pts.reduce((s, p) => s + p.x, 0) / g.pts.length;
      const cy = g.pts.reduce((s, p) => s + p.y, 0) / g.pts.length;
      return { id, label: g.label, hull, cx, cy, n: g.pts.length };
    });
  }, [points]);

  const resetView = useCallback(() => setVp(fitViewport(bounds, width, height)), [bounds, width, height]);

  const onWheel = useCallback((e: React.WheelEvent<SVGSVGElement>) => {
    e.preventDefault();
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const factor = Math.exp(-e.deltaY * 0.0015);
    setVp((cur) => zoomAt(cur, sx, sy, factor));
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    dragState.current = { x: e.clientX, y: e.clientY };
    (e.target as Element).setPointerCapture(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (!dragState.current) return;
    const dx = e.clientX - dragState.current.x;
    const dy = e.clientY - dragState.current.y;
    dragState.current = { x: e.clientX, y: e.clientY };
    setVp((cur) => panBy(cur, -dx, -dy));
  }, []);

  const onPointerUp = useCallback(() => {
    dragState.current = null;
  }, []);

  const onClick = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect || !onSelect) return;
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const [dataX, dataY] = [sx / vp.scale + vp.x, sy / vp.scale + vp.y];
      const found = nearestPoint(points, dataX, dataY, vp);
      if (found) onSelect(found);
    },
    [points, vp, onSelect],
  );

  return (
    <figure style={{ margin: 0 }}>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        role="img"
        aria-label="UMAP arsenal map"
        style={{ cursor: "grab", touchAction: "none" }}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerUp}
        onClick={onClick}
      >
        <g transform={`scale(${vp.scale}) translate(${-vp.x},${-vp.y})`}>
          {archetypes.map((a) => {
            const d = hullPath(a.hull);
            return (
              <g key={a.id}>
                {d && (
                  <path
                    d={d}
                    fill="var(--text-muted)"
                    fillOpacity={0.06}
                    stroke="var(--text-muted)"
                    strokeOpacity={0.4}
                    strokeWidth={1 / vp.scale}
                  />
                )}
                <text
                  x={a.cx}
                  y={a.cy}
                  textAnchor="middle"
                  style={{ fontSize: 11 / vp.scale, fontWeight: 600 }}
                  fill="var(--text-primary)"
                  opacity={0.85}
                >
                  {a.label}
                </text>
              </g>
            );
          })}

          {points.map((p, i) => {
            const isSelected = p.mlbam_id === selectedId && p.season === selectedSeason;
            const isHover = hoverIdx === i;
            const raw = (p as unknown as Record<string, number>)[metric];
            return (
              <circle
                key={`${p.mlbam_id}-${p.season}`}
                cx={p.x}
                cy={p.y}
                r={(isSelected || isHover ? POINT_RADIUS * 1.8 : POINT_RADIUS) / vp.scale}
                fill={divergingColor(raw, def.mid, def.halfRange)}
                stroke={isSelected ? "var(--text-primary)" : undefined}
                strokeWidth={isSelected ? 1.5 / vp.scale : 0}
                opacity={0.75}
                onMouseEnter={() => setHoverIdx(i)}
                onMouseLeave={() => setHoverIdx((cur) => (cur === i ? null : cur))}
              />
            );
          })}
        </g>
      </svg>

      {hoverIdx != null && points[hoverIdx] && (
        <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 4 }}>
          {points[hoverIdx].primary_label} · {points[hoverIdx].primary_velo.toFixed(1)}mph ·{" "}
          {points[hoverIdx].archetype_label} · {points[hoverIdx].season}
        </div>
      )}

      <figcaption
        style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6, fontSize: 11, color: "var(--text-secondary)" }}
      >
        <span>{def.legendLabels?.[0] ?? "low"}</span>
        <span style={{ display: "flex" }}>
          {DIVERGING_LEGEND_STOPS.map((c, i) => (
            <span key={i} style={{ width: 12, height: 10, background: c }} />
          ))}
        </span>
        <span>{def.legendLabels?.[1] ?? "high"}</span>
        <span style={{ marginLeft: "auto" }}>{def.label}</span>
        <button
          onClick={resetView}
          style={{
            marginLeft: 8, fontSize: 11, padding: "2px 8px", borderRadius: "var(--radius)",
            border: "1px solid var(--border)", background: "var(--surface-1)",
            color: "var(--text-secondary)", cursor: "pointer",
          }}
        >
          Reset view
        </button>
      </figcaption>
    </figure>
  );
}
