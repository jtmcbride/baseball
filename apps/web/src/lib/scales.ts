/**
 * The single source of truth for how data maps to visual channels.
 *
 * Every chart imports from here. Per-chart ad-hoc colour is what makes a
 * dashboard look assembled rather than designed, and it also quietly breaks the
 * accessibility guarantees: the palette was validated as a *set*, so a component
 * that invents its own fourth hue invalidates the whole check.
 */

// --- pitch taxonomy ---------------------------------------------------------

export type PitchFamily = "fastball" | "breaking" | "offspeed";

/**
 * Colour encodes FAMILY (3 hues), shape encodes the specific pitch.
 *
 * This is not a stylistic choice. The movement plot is a scatter, so any two
 * marks can end up adjacent and it is governed by the all-pairs CVD gate — which
 * the validated palette clears with three slots, not nine. Giving each of the
 * ~9 pitch types its own hue would fail that gate outright. Family + shape +
 * direct centroid labels carries the same information and stays legible under
 * protanopia and deuteranopia.
 */
export const PITCH_FAMILY: Record<string, PitchFamily> = {
  FF: "fastball", FA: "fastball", SI: "fastball", FC: "fastball",
  SL: "breaking", ST: "breaking", CU: "breaking", KC: "breaking",
  SV: "breaking", CS: "breaking", SC: "breaking",
  CH: "offspeed", FS: "offspeed", FO: "offspeed", EP: "offspeed", KN: "offspeed",
};

export const PITCH_LABEL: Record<string, string> = {
  FF: "4-Seam", FA: "Fastball", SI: "Sinker", FC: "Cutter",
  SL: "Slider", ST: "Sweeper", CU: "Curve", KC: "Knuckle-Curve",
  SV: "Slurve", CS: "Slow Curve", SC: "Screwball",
  CH: "Changeup", FS: "Splitter", FO: "Forkball", EP: "Eephus", KN: "Knuckleball",
};

export const FAMILY_LABEL: Record<PitchFamily, string> = {
  fastball: "Fastball",
  breaking: "Breaking",
  offspeed: "Offspeed",
};

export function familyOf(pitchType: string | null | undefined): PitchFamily {
  return (pitchType && PITCH_FAMILY[pitchType]) || "offspeed";
}

export function labelOf(pitchType: string | null | undefined): string {
  if (!pitchType) return "Unknown";
  return PITCH_LABEL[pitchType] ?? pitchType;
}

/** Resolves to a CSS custom property so light/dark swap in one place. */
export function familyColor(family: PitchFamily): string {
  return `var(--family-${family})`;
}

export function pitchColor(pitchType: string | null | undefined): string {
  return familyColor(familyOf(pitchType));
}

/**
 * Marker shape carries the specific pitch type within its family — the secondary
 * encoding that lets three hues cover nine categories. Stable per pitch type so
 * a filter that removes a series never repaints the survivors.
 */
export type MarkerShape = "circle" | "square" | "triangle" | "diamond";

const SHAPE_BY_PITCH: Record<string, MarkerShape> = {
  FF: "circle", SI: "square", FC: "triangle", FA: "diamond",
  SL: "circle", ST: "square", CU: "triangle", KC: "diamond",
  SV: "diamond", CS: "triangle", SC: "square",
  CH: "circle", FS: "square", FO: "triangle", EP: "diamond", KN: "diamond",
};

export function pitchShape(pitchType: string | null | undefined): MarkerShape {
  return (pitchType && SHAPE_BY_PITCH[pitchType]) || "circle";
}

/** SVG path for a marker of the given shape, centred on the origin. */
export function markerPath(shape: MarkerShape, r: number): string {
  switch (shape) {
    case "square":
      return `M${-r},${-r}h${2 * r}v${2 * r}h${-2 * r}Z`;
    case "triangle":
      return `M0,${-r * 1.15}L${r},${r * 0.85}L${-r},${r * 0.85}Z`;
    case "diamond":
      return `M0,${-r * 1.25}L${r * 1.25},0L0,${r * 1.25}L${-r * 1.25},0Z`;
    default:
      return `M${-r},0a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0`;
  }
}

// --- diverging ramp (hot/cold) ----------------------------------------------

const COOL = [
  "var(--div-cool-1)", "var(--div-cool-2)", "var(--div-cool-3)",
  "var(--div-cool-4)", "var(--div-cool-5)",
];
const WARM = [
  "var(--div-warm-1)", "var(--div-warm-2)", "var(--div-warm-3)",
  "var(--div-warm-4)", "var(--div-warm-5)",
];

/**
 * Map a value to a diverging colour around `mid`.
 *
 * Hot/cold is polarity — above or below league average — not magnitude, so it
 * takes a diverging ramp with a neutral midpoint rather than a single-hue
 * sequential ramp. Centring on the league average is what makes "hot" mean
 * "better than the average hitter" instead of "a big number".
 */
export function divergingColor(value: number, mid: number, halfRange: number): string {
  if (!Number.isFinite(value)) return "transparent";
  const t = Math.max(-1, Math.min(1, (value - mid) / halfRange));
  if (Math.abs(t) < 0.08) return "var(--div-mid)";
  const ramp = t > 0 ? WARM : COOL;
  const idx = Math.min(ramp.length - 1, Math.floor(Math.abs(t) * ramp.length));
  return ramp[idx];
}

export const DIVERGING_LEGEND_STOPS = [...COOL].reverse().concat("var(--div-mid)", ...WARM);

// --- metric definitions -----------------------------------------------------

export interface MetricDef {
  key: string;
  label: string;
  /** League-average centre for the diverging ramp. */
  mid: number;
  /** Distance from centre that saturates the ramp. */
  halfRange: number;
  format: (v: number) => string;
  /** True when a HIGHER value favours the batter. */
  higherIsBatterGood: boolean;
  /**
   * Legend endpoint labels, low then high. Most metrics are a batter-vs-pitcher
   * tug of war, so `higherIsBatterGood` alone is enough to derive them — this
   * override exists for metrics where neither end of the ramp is "the batter":
   * a catcher framing edge runs batter-favored to catcher-favored, and a raw
   * strike rate just runs ball to strike.
   */
  legendLabels?: [string, string];
}

export const ZONE_METRICS: Record<string, MetricDef> = {
  xwoba: {
    key: "xwoba", label: "xwOBA on contact", mid: 0.37, halfRange: 0.22,
    format: (v) => v.toFixed(3).replace(/^0/, ""), higherIsBatterGood: true,
  },
  whiff: {
    key: "whiff", label: "Whiff %", mid: 24, halfRange: 22,
    format: (v) => `${v.toFixed(0)}%`, higherIsBatterGood: false,
  },
  swing: {
    key: "swing", label: "Swing %", mid: 47, halfRange: 40,
    format: (v) => `${v.toFixed(0)}%`, higherIsBatterGood: true,
  },
  exit_velo: {
    key: "exit_velo", label: "Exit velocity", mid: 89, halfRange: 10,
    format: (v) => `${v.toFixed(1)}`, higherIsBatterGood: true,
  },
  run_value: {
    key: "run_value", label: "Run value", mid: 0, halfRange: 0.09,
    format: (v) => v.toFixed(3), higherIsBatterGood: true,
  },
  // Catcher framing edge (viz #20): actual_strike - P(strike) at that spot,
  // the same residual `framing_runs` sums, left un-aggregated. mid/halfRange
  // set from the real grid data (p1/p99 on reliable cells: -0.13 / +0.11).
  framing: {
    key: "framing", label: "Framing edge", mid: 0, halfRange: 0.10,
    format: (v) => (v >= 0 ? "+" : "") + v.toFixed(2),
    higherIsBatterGood: false,
    legendLabels: ["fewer strikes called", "more strikes called"],
  },
  // Umpire zone map (viz #13): the umpire's own actual called-strike rate by
  // location — no model score involved. The client draws its 50% contour as
  // the umpire's effective zone boundary against the rulebook rectangle.
  strike_rate: {
    key: "strike_rate", label: "Called-strike rate", mid: 50, halfRange: 50,
    format: (v) => `${v.toFixed(0)}%`,
    higherIsBatterGood: false,
    legendLabels: ["ball", "strike"],
  },
};

export const number = (v: number | null | undefined, digits = 1): string =>
  v == null || !Number.isFinite(v) ? "—" : v.toFixed(digits);
