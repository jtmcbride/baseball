/**
 * Smoothed hot/cold zone surface with an explicit reliability mask.
 *
 * The mask is the point of this component. A kernel-smoothed surface will happily
 * paint a corner cell at full saturation when that cell is interpolated almost
 * entirely from its neighbours — the estimate exists, but there is no data behind
 * it. Fading those cells and saying so in the legend is the difference between a
 * chart that informs and one that merely looks confident.
 */

import { useMemo, useState } from "react";
import type { ZoneGrid } from "../lib/api";
import { DIVERGING_LEGEND_STOPS, ZONE_METRICS, divergingColor } from "../lib/scales";

interface Props {
  grid: ZoneGrid;
  width?: number;
  height?: number;
}

// Rulebook zone in the grid's own coordinates: the plate is 17" wide (plus a
// ball's radius each side), and z is normalized so 0..1 IS the batter's zone.
const PLATE_HALF_FT = 0.83;

export function StrikeZoneHeatmap({ grid, width = 320, height = 380 }: Props) {
  const [hover, setHover] = useState<{ i: number; j: number } | null>(null);
  const metric = ZONE_METRICS[grid.metric] ?? ZONE_METRICS.whiff;
  const { grid_n: n, x_min, x_max, z_min, z_max, min_reliable_n } = grid.extent;

  const pad = { top: 8, right: 8, bottom: 28, left: 34 };
  const w = width - pad.left - pad.right;
  const h = height - pad.top - pad.bottom;
  const cw = w / n;
  const ch = h / n;

  const xScale = (xf: number) => ((xf - x_min) / (x_max - x_min)) * w;
  // z is inverted: higher pitches draw nearer the top of the SVG.
  const yScale = (zf: number) => h - ((zf - z_min) / (z_max - z_min)) * h;

  const cells = useMemo(() => {
    const out: { i: number; j: number; v: number | null; rel: number }[] = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const idx = i * n + j;
        out.push({ i, j, v: grid.surface[idx], rel: grid.reliability[idx] ?? 0 });
      }
    }
    return out;
  }, [grid, n]);

  const hovered = hover ? cells[hover.i * n + hover.j] : null;

  return (
    <figure style={{ margin: 0 }}>
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`${metric.label} by pitch location, catcher's view. ${grid.n_pitches} pitches.`}
      >
        <g transform={`translate(${pad.left},${pad.top})`}>
          {cells.map((c) => {
            if (c.v == null || !Number.isFinite(c.v)) return null;
            // Below the reliability floor the estimate is mostly borrowed from
            // neighbours, so it fades out rather than asserting a value.
            const reliable = c.rel >= min_reliable_n;
            const opacity = reliable ? 1 : Math.max(0.06, (c.rel / min_reliable_n) * 0.35);
            return (
              <rect
                key={`${c.i}-${c.j}`}
                x={c.i * cw}
                y={h - (c.j + 1) * ch}
                width={cw + 0.5}
                height={ch + 0.5}
                fill={divergingColor(c.v, metric.mid, metric.halfRange)}
                opacity={opacity}
                onMouseEnter={() => setHover({ i: c.i, j: c.j })}
                onMouseLeave={() => setHover(null)}
              />
            );
          })}

          {/* Rulebook strike zone, drawn on top as the reference frame. */}
          <rect
            x={xScale(-PLATE_HALF_FT)}
            y={yScale(1)}
            width={xScale(PLATE_HALF_FT) - xScale(-PLATE_HALF_FT)}
            height={yScale(0) - yScale(1)}
            fill="none"
            stroke="var(--text-primary)"
            strokeWidth={1.5}
            opacity={0.7}
          />
          {/* Thirds, to read location without counting pixels. */}
          {[1 / 3, 2 / 3].map((f) => (
            <g key={f} opacity={0.28}>
              <line
                x1={xScale(-PLATE_HALF_FT + f * 2 * PLATE_HALF_FT)}
                x2={xScale(-PLATE_HALF_FT + f * 2 * PLATE_HALF_FT)}
                y1={yScale(1)} y2={yScale(0)}
                stroke="var(--text-primary)" strokeWidth={1}
              />
              <line
                x1={xScale(-PLATE_HALF_FT)} x2={xScale(PLATE_HALF_FT)}
                y1={yScale(f)} y2={yScale(f)}
                stroke="var(--text-primary)" strokeWidth={1}
              />
            </g>
          ))}

          {hovered && (
            <rect
              x={hovered.i * cw} y={h - (hovered.j + 1) * ch}
              width={cw} height={ch}
              fill="none" stroke="var(--text-primary)" strokeWidth={1.5}
            />
          )}

          <text className="axis-label" x={w / 2} y={h + 18} textAnchor="middle">
            catcher's view · horizontal location
          </text>
          <text className="axis-label" transform={`translate(-22,${h / 2}) rotate(-90)`} textAnchor="middle">
            zone height
          </text>
        </g>
      </svg>

      <figcaption style={{ fontSize: 12, color: "var(--text-secondary)" }}>
        <ZoneLegend metric={grid.metric} />
        <div style={{ marginTop: 6, minHeight: 32 }}>
          {hovered && hovered.v != null ? (
            <>
              <strong style={{ color: "var(--text-primary)" }}>
                {metric.format(hovered.v)}
              </strong>{" "}
              · {Math.round(hovered.rel)} effective pitches
              {hovered.rel < min_reliable_n && (
                <span style={{ color: "var(--text-muted)" }}> · too few to trust</span>
              )}
            </>
          ) : (
            <span style={{ color: "var(--text-muted)" }}>
              Faded cells fall below {min_reliable_n} effective pitches — smoothed
              from neighbours, not measured.
            </span>
          )}
        </div>
      </figcaption>
    </figure>
  );
}

function ZoneLegend({ metric }: { metric: string }) {
  const def = ZONE_METRICS[metric] ?? ZONE_METRICS.whiff;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
      <span style={{ color: "var(--text-muted)" }}>
        {def.higherIsBatterGood ? "pitcher" : "batter"}
      </span>
      <div style={{ display: "flex", flex: 1, height: 8, borderRadius: 4, overflow: "hidden" }}>
        {DIVERGING_LEGEND_STOPS.map((c, i) => (
          <div key={i} style={{ background: c, flex: 1 }} />
        ))}
      </div>
      <span style={{ color: "var(--text-muted)" }}>
        {def.higherIsBatterGood ? "batter" : "pitcher"}
      </span>
    </div>
  );
}
