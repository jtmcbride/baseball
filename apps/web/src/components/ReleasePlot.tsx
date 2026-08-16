/**
 * Release points, from the catcher's view. Tight clustering across pitch types
 * is what hides the pitch from the hitter; a visible split by family is a tell.
 */

import { useMemo } from "react";
import { FAMILY_LABEL, familyColor, familyOf, markerPath, pitchShape } from "../lib/scales";

export interface ReleasePoint {
  pitch_type: string | null;
  release_pos_x: number | null;
  release_pos_z: number | null;
}

export function ReleasePlot({
  points, width = 320, height = 300,
}: { points: ReleasePoint[]; width?: number; height?: number }) {
  const pad = { top: 10, right: 10, bottom: 30, left: 38 };
  const w = width - pad.left - pad.right;
  const h = height - pad.top - pad.bottom;

  const valid = useMemo(
    () => points.filter((p) => p.release_pos_x != null && p.release_pos_z != null),
    [points],
  );

  const [xMin, xMax] = [-4.5, 4.5];
  const [zMin, zMax] = [0, 8];
  const x = (v: number) => ((v - xMin) / (xMax - xMin)) * w;
  const y = (v: number) => h - ((v - zMin) / (zMax - zMin)) * h;

  const families = useMemo(
    () => [...new Set(valid.map((p) => familyOf(p.pitch_type)))],
    [valid],
  );

  return (
    <figure style={{ margin: 0 }}>
      <svg width={width} height={height} role="img" aria-label="Release point plot">
        <g transform={`translate(${pad.left},${pad.top})`}>
          {[-4, -2, 0, 2, 4].map((t) => (
            <line key={t} className="grid-line" x1={x(t)} x2={x(t)} y1={0} y2={h} />
          ))}
          {[0, 2, 4, 6, 8].map((t) => (
            <g key={t}>
              <line className="grid-line" x1={0} x2={w} y1={y(t)} y2={y(t)} />
              <text className="axis-label" x={-6} y={y(t) + 3} textAnchor="end">{t}</text>
            </g>
          ))}
          {/* Ground line — release height is measured from it. */}
          <line className="axis-line" x1={0} x2={w} y1={y(0)} y2={y(0)} />

          {valid.map((p, i) => (
            <path
              key={i}
              d={markerPath(pitchShape(p.pitch_type), 2.6)}
              transform={`translate(${x(p.release_pos_x!)},${y(p.release_pos_z!)})`}
              fill={familyColor(familyOf(p.pitch_type))}
              opacity={0.4}
            />
          ))}

          <text className="axis-label" x={w / 2} y={h + 22} textAnchor="middle">
            release side (ft from centre)
          </text>
          <text className="axis-label" transform={`translate(-28,${h / 2}) rotate(-90)`} textAnchor="middle">
            release height (ft)
          </text>
        </g>
      </svg>
      <figcaption style={{ display: "flex", gap: 14, fontSize: 12, color: "var(--text-secondary)" }}>
        {families.map((f) => (
          <span key={f} style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <svg width={10} height={10} aria-hidden><circle cx={5} cy={5} r={4} fill={familyColor(f)} /></svg>
            {FAMILY_LABEL[f]}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
