/**
 * Swing path viz (#19): attack angle vs. pitch descent angle, one point per
 * tracked swing (2023H2+ — see `SWING_TRACKING_NOTE` below), plus a swing-length
 * distribution. Templated on `ArsenalMap.tsx`'s hand-rolled-SVG-plus-
 * `lib/viewport.ts` pattern rather than a new charting library.
 *
 * Colour encodes pitch family, shape encodes pitch type — `familyColor`/
 * `pitchShape` from `lib/scales.ts`, unchanged, per the all-pairs CVD gate that
 * governs every scatter in this app. Whiff vs. contact is a second channel
 * layered on top: hollow marker for a whiff, filled for a swing that made
 * contact, so it doesn't need a fourth hue.
 */

import { useCallback, useMemo, useRef, useState } from "react";
import type { SwingPitchRow } from "../lib/api";
import { histogram } from "../lib/histogram";
import { familyColor, familyOf, markerPath, pitchShape } from "../lib/scales";
import { fitViewport, panBy, zoomAt, type Viewport } from "../lib/viewport";

interface Props {
  pitches: SwingPitchRow[];
  width?: number;
  height?: number;
}

const POINT_RADIUS = 3;

export function SwingPathScatter({ pitches, width = 480, height = 400 }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const dragState = useRef<{ x: number; y: number } | null>(null);

  const points = useMemo(
    () => pitches.map((p) => ({ ...p, x: p.vaa_deg, y: p.attack_angle })),
    [pitches],
  );

  const bounds = useMemo(() => {
    if (points.length === 0) return { xMin: -20, xMax: 5, yMin: -20, yMax: 30 };
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    return { xMin: Math.min(...xs), xMax: Math.max(...xs), yMin: Math.min(...ys), yMax: Math.max(...ys) };
  }, [points]);

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

  const swingLengths = useMemo(
    () => pitches.map((p) => p.swing_length).filter((v) => Number.isFinite(v)),
    [pitches],
  );
  const bins = useMemo(() => histogram(swingLengths, 16), [swingLengths]);
  const maxCount = Math.max(1, ...bins.map((b) => b.count));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <figure style={{ margin: 0 }}>
        <svg
          ref={svgRef}
          width={width}
          height={height}
          role="img"
          aria-label="Swing path scatter: attack angle vs. pitch descent angle"
          style={{ cursor: "grab", touchAction: "none" }}
          onWheel={onWheel}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={onPointerUp}
        >
          <g transform={`scale(${vp.scale}) translate(${-vp.x},${-vp.y})`}>
            {/* Reference line at attack_angle = 0 (level swing). */}
            <line
              x1={bounds.xMin - 50}
              x2={bounds.xMax + 50}
              y1={0}
              y2={0}
              stroke="var(--gridline)"
              strokeWidth={1 / vp.scale}
            />
            {points.map((p, i) => {
              const color = familyColor(familyOf(p.pitch_type));
              const shape = pitchShape(p.pitch_type);
              const r = (hoverIdx === i ? POINT_RADIUS * 1.6 : POINT_RADIUS) / vp.scale;
              return (
                <path
                  key={i}
                  d={markerPath(shape, r)}
                  transform={`translate(${p.x},${p.y})`}
                  fill={p.is_whiff ? "none" : color}
                  stroke={color}
                  strokeWidth={1.2 / vp.scale}
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
            {points[hoverIdx].pitch_type ?? "?"} · descent {points[hoverIdx].vaa_deg.toFixed(1)}° · attack{" "}
            {points[hoverIdx].attack_angle.toFixed(1)}° · {points[hoverIdx].is_whiff ? "whiff" : "contact/take"}
          </div>
        )}

        <figcaption
          style={{ display: "flex", alignItems: "center", gap: 10, marginTop: 6, fontSize: 11, color: "var(--text-secondary)" }}
        >
          <span>hollow = whiff · filled = contact</span>
          <span style={{ marginLeft: "auto" }}>x: pitch descent angle · y: attack angle</span>
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

      {bins.length > 0 && (
        <div>
          <div style={{ fontSize: 11, color: "var(--text-secondary)", marginBottom: 4 }}>
            Swing length distribution (ft)
          </div>
          <svg width={width} height={60} role="img" aria-label="Swing length histogram">
            {bins.map((b, i) => {
              const barWidth = width / bins.length - 1;
              const barHeight = (b.count / maxCount) * 54;
              return (
                <rect
                  key={i}
                  x={(i * width) / bins.length}
                  y={60 - barHeight}
                  width={barWidth}
                  height={barHeight}
                  fill="var(--family-fastball)"
                  opacity={0.6}
                />
              );
            })}
          </svg>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "var(--text-muted)" }}>
            <span>{bins[0].x0.toFixed(1)}</span>
            <span>{bins[bins.length - 1].x1.toFixed(1)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
