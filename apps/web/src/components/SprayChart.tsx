/**
 * Spray chart (viz #8): every tracked batted ball plotted over a real park
 * wall outline, with the batter's smoothed xwOBA-on-contact contour
 * (`mart_batter_spray`) traced on top. Pan/zoom via `lib/viewport.ts`, same
 * `<g transform>` pattern as `ArsenalMap.tsx` — a full-field view benefits
 * from zooming into home-plate detail more than the fixed-scale strike zone
 * heatmap does.
 *
 * Coordinates are plain feet-from-home-plate (`x_ft`/`y_ft`, see
 * `bbetl.transforms.statcast.enrich`'s hit-coordinate comment) with the SVG
 * y-axis flipped (`-y_ft`) so straight-away CF renders at the top of the
 * chart, the same "up is away from the plate" convention as every park
 * diagram. The contour reuses `lib/contour.ts`'s `marchingSquares` directly —
 * same function `StrikeZoneHeatmap.tsx` uses for the umpire zone map's 50%
 * line — rather than that component itself, which is hardcoded to strike-zone
 * axes.
 */

import { useCallback, useMemo, useRef, useState } from "react";
import type { BattedBallRow, SpraySurface } from "../lib/api";
import { marchingSquares } from "../lib/contour";
import { parkPolygon, PARKS, type Point2 } from "../data/parks";
import { DIVERGING_LEGEND_STOPS, ZONE_METRICS, divergingColor } from "../lib/scales";
import { fitViewport, panBy, zoomAt, type Viewport } from "../lib/viewport";

interface Props {
  battedBalls: BattedBallRow[];
  contour?: SpraySurface;
  defaultTeam?: string | null;
  width?: number;
  height?: number;
}

const metric = ZONE_METRICS.spray;

export function SprayChart({ battedBalls, contour, defaultTeam, width = 520, height = 520 }: Props) {
  const [team, setTeam] = useState<string>(defaultTeam && PARKS[defaultTeam] ? defaultTeam : "NYY");
  const svgRef = useRef<SVGSVGElement>(null);
  const dragState = useRef<{ x: number; y: number } | null>(null);

  const dims = PARKS[team] ?? PARKS.NYY;
  const wall = useMemo(() => parkPolygon(dims), [dims]);
  const wallFlipped = useMemo(() => wall.map((p) => ({ x: p.x, y: -p.y })), [wall]);

  const points = useMemo(
    () => battedBalls.map((b) => ({ ...b, sx: b.x_ft, sy: -b.y_ft })),
    [battedBalls],
  );

  const bounds = useMemo(() => {
    const allX = wall.map((p) => p.x).concat(points.map((p) => p.sx));
    const allY = wall.map((p) => -p.y).concat(points.map((p) => p.sy));
    if (allX.length === 0) return { xMin: -350, xMax: 350, yMin: -450, yMax: 20 };
    return {
      xMin: Math.min(...allX, -30),
      xMax: Math.max(...allX, 30),
      yMin: Math.min(...allY),
      yMax: Math.max(...allY, 20),
    };
  }, [wall, points]);

  const [vp, setVp] = useState<Viewport>(() => fitViewport(bounds, width, height));
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

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

  // Grid cells + contour line, in feet-from-plate. Index i/j -> feet is a
  // linear map off the extent shipped alongside the mart, same convention
  // `StrikeZoneHeatmap.tsx` uses for its own index-space contour.
  const gridCells = useMemo(() => {
    if (!contour) return [];
    const { grid_n: n, x_min, x_max, y_min, y_max } = contour.extent;
    const dx = (x_max - x_min) / n;
    const dy = (y_max - y_min) / n;
    const out: { x0: number; y0: number; v: number | null; rel: number }[] = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const idx = i * n + j;
        out.push({
          x0: x_min + i * dx,
          y0: y_min + j * dy,
          v: contour.surface[idx],
          rel: contour.reliability[idx] ?? 0,
        });
      }
    }
    return out;
  }, [contour]);

  const cellW = contour ? (contour.extent.x_max - contour.extent.x_min) / contour.extent.grid_n : 0;
  const cellH = contour ? (contour.extent.y_max - contour.extent.y_min) / contour.extent.grid_n : 0;

  const contourLine = useMemo(() => {
    if (!contour) return [];
    const { grid_n: n, x_min, x_max, y_min, y_max } = contour.extent;
    const segs = marchingSquares(contour.surface, n, metric.mid);
    const dx = (x_max - x_min) / n;
    const dy = (y_max - y_min) / n;
    return segs.map(
      ([x1, y1, x2, y2]) =>
        [x_min + x1 * dx, y_min + y1 * dy, x_min + x2 * dx, y_min + y2 * dy] as [number, number, number, number],
    );
  }, [contour]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <select
        value={team}
        onChange={(e) => setTeam(e.target.value)}
        style={{
          alignSelf: "flex-start", padding: "4px 8px", borderRadius: 4,
          border: "1px solid var(--gridline)", background: "var(--surface-1)",
          color: "var(--text-primary)", fontSize: 12,
        }}
      >
        {Object.entries(PARKS)
          .sort(([, a], [, b]) => a.name.localeCompare(b.name))
          .map(([code, d]) => (
            <option key={code} value={code}>
              {d.name}
            </option>
          ))}
      </select>

      <figure style={{ margin: 0 }}>
        <svg
          ref={svgRef}
          width={width}
          height={height}
          role="img"
          aria-label="Spray chart over the selected park's wall outline"
          style={{ cursor: "grab", touchAction: "none" }}
          onWheel={onWheel}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={onPointerUp}
        >
          <g transform={`scale(${vp.scale}) translate(${-vp.x},${-vp.y})`}>
            {contour &&
              gridCells.map((c, idx) => {
                if (c.v == null || !Number.isFinite(c.v)) return null;
                const reliable = c.rel >= contour.extent.min_reliable_n;
                const opacity = reliable ? 0.55 : Math.max(0.03, (c.rel / contour.extent.min_reliable_n) * 0.2);
                return (
                  <rect
                    key={idx}
                    x={c.x0}
                    y={-(c.y0 + cellH)}
                    width={cellW + 0.5}
                    height={cellH + 0.5}
                    fill={divergingColor(c.v, metric.mid, metric.halfRange)}
                    opacity={opacity}
                  />
                );
              })}

            {contourLine.map(([x1, y1, x2, y2], idx) => (
              <line
                key={idx}
                x1={x1} y1={-y1} x2={x2} y2={-y2}
                stroke="var(--text-primary)"
                strokeWidth={1.5 / vp.scale}
                opacity={0.6}
              />
            ))}

            <path
              d={wallPath(wallFlipped)}
              fill="none"
              stroke="var(--text-primary)"
              strokeWidth={2 / vp.scale}
              opacity={0.85}
            />

            {points.map((p, i) => (
              <circle
                key={i}
                cx={p.sx}
                cy={p.sy}
                r={(hoverIdx === i ? 4.5 : 2.6) / vp.scale}
                fill={
                  p.estimated_woba_using_speedangle != null
                    ? divergingColor(p.estimated_woba_using_speedangle, metric.mid, metric.halfRange)
                    : "var(--text-muted)"
                }
                opacity={0.75}
                onMouseEnter={() => setHoverIdx(i)}
                onMouseLeave={() => setHoverIdx((cur) => (cur === i ? null : cur))}
              />
            ))}
          </g>
        </svg>

        {hoverIdx != null && points[hoverIdx] && (
          <div style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 4 }}>
            {points[hoverIdx].bb_type ?? "?"} · {points[hoverIdx].launch_speed?.toFixed(1) ?? "—"}mph ·{" "}
            {points[hoverIdx].events ?? "?"}
          </div>
        )}

        <figcaption
          style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 6, fontSize: 11, color: "var(--text-secondary)" }}
        >
          <span>cold</span>
          <span style={{ display: "flex" }}>
            {DIVERGING_LEGEND_STOPS.map((c, i) => (
              <span key={i} style={{ width: 12, height: 10, background: c }} />
            ))}
          </span>
          <span>hot</span>
          <span style={{ marginLeft: "auto" }}>{metric.label}</span>
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
    </div>
  );
}

function wallPath(points: Point2[]): string {
  if (points.length === 0) return "";
  return `M${points.map((p) => `${p.x},${p.y}`).join("L")}`;
}
