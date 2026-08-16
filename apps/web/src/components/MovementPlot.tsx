/**
 * Pitch movement: horizontal break vs induced vertical break.
 *
 * Both axes are arm-side normalized, so a lefty and a righty with the same pitch
 * shape land in the same place. Without that, the two hands mirror each other and
 * every cross-handedness comparison silently inverts.
 *
 * Colour encodes pitch FAMILY (three all-pairs-validated hues) and marker shape
 * encodes the specific pitch. Cluster centroids carry direct labels, which is
 * both the standard way these plots are read and the "relief" the palette
 * requires for its lighter light-mode hue.
 */

import { useMemo, useState } from "react";
import {
  FAMILY_LABEL, familyColor, familyOf, labelOf, markerPath, pitchShape,
} from "../lib/scales";

export interface MovementPoint {
  pitch_type: string | null;
  release_speed: number | null;
  ivb_in: number | null;
  hb_arm_in: number | null;
  is_whiff?: boolean;
}

interface Props {
  points: MovementPoint[];
  width?: number;
  height?: number;
}

const LIMIT = 26; // inches; covers essentially every pitch thrown

export function MovementPlot({ points, width = 400, height = 400 }: Props) {
  const [hoverType, setHoverType] = useState<string | null>(null);

  const pad = { top: 12, right: 12, bottom: 34, left: 40 };
  const w = width - pad.left - pad.right;
  const h = height - pad.top - pad.bottom;

  const x = (v: number) => ((v + LIMIT) / (2 * LIMIT)) * w;
  const y = (v: number) => h - ((v + LIMIT) / (2 * LIMIT)) * h;

  const valid = useMemo(
    () => points.filter((p) => p.ivb_in != null && p.hb_arm_in != null && p.pitch_type),
    [points],
  );

  /** Centroids drive the direct labels — one per pitch type with enough volume. */
  const centroids = useMemo(() => {
    const groups = new Map<string, { hb: number; ivb: number; n: number; velo: number }>();
    for (const p of valid) {
      const k = p.pitch_type!;
      const g = groups.get(k) ?? { hb: 0, ivb: 0, n: 0, velo: 0 };
      g.hb += p.hb_arm_in!; g.ivb += p.ivb_in!; g.velo += p.release_speed ?? 0; g.n += 1;
      groups.set(k, g);
    }
    return [...groups.entries()]
      .filter(([, g]) => g.n >= 5)
      .map(([k, g]) => ({
        pitch_type: k, hb: g.hb / g.n, ivb: g.ivb / g.n,
        velo: g.velo / g.n, n: g.n,
      }))
      .sort((a, b) => b.n - a.n);
  }, [valid]);

  const families = useMemo(
    () => [...new Set(centroids.map((c) => familyOf(c.pitch_type)))],
    [centroids],
  );

  return (
    <figure style={{ margin: 0 }}>
      <svg width={width} height={height} role="img" aria-label="Pitch movement plot">
        <g transform={`translate(${pad.left},${pad.top})`}>
          {[-20, -10, 0, 10, 20].map((t) => (
            <g key={t}>
              <line className="grid-line" x1={x(t)} x2={x(t)} y1={0} y2={h} />
              <line className="grid-line" x1={0} x2={w} y1={y(t)} y2={y(t)} />
            </g>
          ))}
          {/* Origin axes sit above the grid: "no movement" is the reference. */}
          <line className="axis-line" x1={x(0)} x2={x(0)} y1={0} y2={h} />
          <line className="axis-line" x1={0} x2={w} y1={y(0)} y2={y(0)} />

          {[-20, -10, 10, 20].map((t) => (
            <text key={`xt${t}`} className="axis-label" x={x(t)} y={h + 14} textAnchor="middle">
              {t}
            </text>
          ))}
          {[-20, -10, 10, 20].map((t) => (
            <text key={`yt${t}`} className="axis-label" x={-8} y={y(t) + 3} textAnchor="end">
              {t}
            </text>
          ))}

          {valid.map((p, i) => {
            const dim = hoverType !== null && p.pitch_type !== hoverType;
            return (
              <path
                key={i}
                d={markerPath(pitchShape(p.pitch_type), 3.2)}
                transform={`translate(${x(p.hb_arm_in!)},${y(p.ivb_in!)})`}
                fill={familyColor(familyOf(p.pitch_type))}
                opacity={dim ? 0.06 : 0.42}
              />
            );
          })}

          {centroids.map((c) => {
            const dim = hoverType !== null && c.pitch_type !== hoverType;
            return (
              <g
                key={c.pitch_type}
                opacity={dim ? 0.25 : 1}
                onMouseEnter={() => setHoverType(c.pitch_type)}
                onMouseLeave={() => setHoverType(null)}
                style={{ cursor: "default" }}
              >
                {/* 2px surface ring separates overlapping centroid marks. */}
                <path
                  d={markerPath(pitchShape(c.pitch_type), 7)}
                  transform={`translate(${x(c.hb)},${y(c.ivb)})`}
                  fill={familyColor(familyOf(c.pitch_type))}
                  stroke="var(--surface-1)"
                  strokeWidth={2}
                />
                <text
                  x={x(c.hb) + 11} y={y(c.ivb) + 4}
                  style={{ fontSize: 11, fontWeight: 600 }}
                  fill="var(--text-primary)"
                >
                  {labelOf(c.pitch_type)}
                </text>
                <text
                  x={x(c.hb) + 11} y={y(c.ivb) + 16}
                  className="axis-label"
                >
                  {c.velo.toFixed(1)} mph · {c.n}
                </text>
              </g>
            );
          })}

          <text className="axis-label" x={w / 2} y={h + 30} textAnchor="middle">
            ← glove side · horizontal break (in) · arm side →
          </text>
          <text className="axis-label" transform={`translate(-28,${h / 2}) rotate(-90)`} textAnchor="middle">
            induced vertical break (in)
          </text>
        </g>
      </svg>

      {/* Legend is always present for >= 2 series, so identity never rests on
          colour alone. */}
      <figcaption
        style={{ display: "flex", gap: 14, fontSize: 12, color: "var(--text-secondary)", flexWrap: "wrap" }}
      >
        {families.map((f) => (
          <span key={f} style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <svg width={10} height={10} aria-hidden>
              <circle cx={5} cy={5} r={4} fill={familyColor(f)} />
            </svg>
            {FAMILY_LABEL[f]}
          </span>
        ))}
        <span style={{ color: "var(--text-muted)" }}>shape = pitch type</span>
      </figcaption>
    </figure>
  );
}
