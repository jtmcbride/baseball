/**
 * Velocity by pitch number within an outing — the fatigue curve.
 *
 * Binned by pitch count, one line per pitch family. A line needs a crosshair, and
 * bins with thin samples are drawn hollow rather than being silently averaged in
 * with the rest.
 */

import { useMemo, useState } from "react";
import { FAMILY_LABEL, type PitchFamily, familyColor, familyOf } from "../lib/scales";

export interface VeloPoint {
  pitch_type: string | null;
  release_speed: number | null;
  pitch_number_in_game?: number | null;
  inning: number | null;
}

const MIN_BIN_N = 5;

export function VeloTrend({
  points, width = 420, height = 240,
}: { points: VeloPoint[]; width?: number; height?: number }) {
  const [hoverInning, setHoverInning] = useState<number | null>(null);
  const pad = { top: 12, right: 12, bottom: 32, left: 40 };
  const w = width - pad.left - pad.right;
  const h = height - pad.top - pad.bottom;

  const series = useMemo(() => {
    const byFamily = new Map<PitchFamily, Map<number, { sum: number; n: number }>>();
    for (const p of points) {
      if (p.release_speed == null || p.inning == null || !p.pitch_type) continue;
      const fam = familyOf(p.pitch_type);
      if (!byFamily.has(fam)) byFamily.set(fam, new Map());
      const bins = byFamily.get(fam)!;
      const b = bins.get(p.inning) ?? { sum: 0, n: 0 };
      b.sum += p.release_speed; b.n += 1;
      bins.set(p.inning, b);
    }
    return [...byFamily.entries()].map(([family, bins]) => ({
      family,
      points: [...bins.entries()]
        .map(([inning, b]) => ({ inning, velo: b.sum / b.n, n: b.n }))
        .sort((a, b) => a.inning - b.inning),
    }));
  }, [points]);

  const all = series.flatMap((s) => s.points);
  if (!all.length) return <p style={{ color: "var(--text-muted)" }}>No velocity data.</p>;

  const innings = [...new Set(all.map((p) => p.inning))].sort((a, b) => a - b);
  const vMin = Math.floor(Math.min(...all.map((p) => p.velo)) - 1);
  const vMax = Math.ceil(Math.max(...all.map((p) => p.velo)) + 1);

  const x = (i: number) =>
    innings.length === 1 ? w / 2 : ((i - innings[0]) / (innings[innings.length - 1] - innings[0])) * w;
  const y = (v: number) => h - ((v - vMin) / (vMax - vMin)) * h;

  return (
    <figure style={{ margin: 0 }}>
      <svg
        width={width} height={height} role="img" aria-label="Velocity by inning"
        onMouseLeave={() => setHoverInning(null)}
      >
        <g transform={`translate(${pad.left},${pad.top})`}>
          {[vMin, (vMin + vMax) / 2, vMax].map((t) => (
            <g key={t}>
              <line className="grid-line" x1={0} x2={w} y1={y(t)} y2={y(t)} />
              <text className="axis-label" x={-6} y={y(t) + 3} textAnchor="end">{t.toFixed(0)}</text>
            </g>
          ))}
          {innings.map((i) => (
            <text key={i} className="axis-label" x={x(i)} y={h + 16} textAnchor="middle">{i}</text>
          ))}

          {/* Crosshair: a line chart gets a hover layer by default. */}
          {hoverInning != null && (
            <line
              x1={x(hoverInning)} x2={x(hoverInning)} y1={0} y2={h}
              stroke="var(--text-muted)" strokeWidth={1} strokeDasharray="3 3"
            />
          )}

          {series.map((s) => (
            <g key={s.family}>
              <path
                d={s.points.map((p, i) => `${i ? "L" : "M"}${x(p.inning)},${y(p.velo)}`).join("")}
                fill="none" stroke={familyColor(s.family)} strokeWidth={2}
                strokeLinejoin="round" strokeLinecap="round"
              />
              {s.points.map((p) => (
                <circle
                  key={p.inning}
                  cx={x(p.inning)} cy={y(p.velo)} r={4}
                  // Hollow marker = thin bin. Drawing it solid would present a
                  // 2-pitch average with the same authority as a 40-pitch one.
                  fill={p.n >= MIN_BIN_N ? familyColor(s.family) : "var(--surface-1)"}
                  stroke={familyColor(s.family)} strokeWidth={2}
                />
              ))}
            </g>
          ))}

          {/* Wide invisible hit targets — bigger than the marks themselves. */}
          {innings.map((i) => (
            <rect
              key={`hit${i}`}
              x={x(i) - w / (innings.length * 2 || 1)} y={0}
              width={w / (innings.length || 1)} height={h}
              fill="transparent" onMouseEnter={() => setHoverInning(i)}
            />
          ))}

          <text className="axis-label" x={w / 2} y={h + 30} textAnchor="middle">inning</text>
        </g>
      </svg>
      <figcaption style={{ display: "flex", gap: 14, fontSize: 12, color: "var(--text-secondary)", flexWrap: "wrap" }}>
        {series.map((s) => (
          <span key={s.family} style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
            <svg width={12} height={4} aria-hidden>
              <rect width={12} height={3} rx={1.5} fill={familyColor(s.family)} />
            </svg>
            {FAMILY_LABEL[s.family]}
            {hoverInning != null && (() => {
              const p = s.points.find((q) => q.inning === hoverInning);
              return p ? (
                <strong style={{ color: "var(--text-primary)" }}>{p.velo.toFixed(1)}</strong>
              ) : null;
            })()}
          </span>
        ))}
        <span style={{ color: "var(--text-muted)" }}>hollow = under {MIN_BIN_N} pitches</span>
      </figcaption>
    </figure>
  );
}
