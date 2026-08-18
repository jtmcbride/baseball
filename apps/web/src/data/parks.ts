/**
 * MLB park wall outlines, one polygon per team, in feet from home plate
 * (standard orientation: home plate at the origin, straight-away CF along
 * +y — same frame `x_ft`/`y_ft` use, see `bbetl.transforms.statcast.enrich`).
 * Frontend-only static reference data (viz #8's spray chart) — there is no
 * `dim_park` lake table, and nothing else in this app needs park geometry as
 * a joinable feature, so a table + route for data that never changes per
 * query would be pure overhead.
 *
 * Each park is built from five publicly documented distance markers — the
 * two foul lines, the two power alleys, and dead centre — rather than the
 * 3-point (LF/CF/RF) approximation this project explicitly rejected. Real MLB
 * fences are not five straight segments either, so `parkPolygon` below spline-
 * interpolates a smooth wall between those five measured points rather than
 * drawing a pentagon. Distances are current-era figures as commonly published
 * on team scoreboards/media guides; a handful of teams round differently
 * season to season, so treat these as "the shape", not survey-grade fence
 * data. Keyed by the Statcast `home_team` code (AZ, not ARI; ATH, not OAK).
 */

export interface ParkDimensions {
  name: string;
  /** Distances in feet at [LF line, LF-CF alley, CF, CF-RF alley, RF line]. */
  lf: number;
  lfAlley: number;
  cf: number;
  rfAlley: number;
  rf: number;
}

export const PARKS: Record<string, ParkDimensions> = {
  AZ: { name: "Chase Field", lf: 330, lfAlley: 374, cf: 407, rfAlley: 374, rf: 334 },
  ATL: { name: "Truist Park", lf: 335, lfAlley: 385, cf: 400, rfAlley: 375, rf: 325 },
  ATH: { name: "Sutter Health Park", lf: 330, lfAlley: 367, cf: 403, rfAlley: 367, rf: 325 },
  BAL: { name: "Oriole Park at Camden Yards", lf: 333, lfAlley: 364, cf: 410, rfAlley: 373, rf: 318 },
  BOS: { name: "Fenway Park", lf: 310, lfAlley: 379, cf: 390, rfAlley: 380, rf: 302 },
  CHC: { name: "Wrigley Field", lf: 355, lfAlley: 368, cf: 400, rfAlley: 368, rf: 353 },
  CIN: { name: "Great American Ball Park", lf: 328, lfAlley: 379, cf: 404, rfAlley: 370, rf: 325 },
  CLE: { name: "Progressive Field", lf: 325, lfAlley: 370, cf: 405, rfAlley: 375, rf: 325 },
  COL: { name: "Coors Field", lf: 347, lfAlley: 390, cf: 415, rfAlley: 375, rf: 350 },
  CWS: { name: "Rate Field", lf: 330, lfAlley: 375, cf: 400, rfAlley: 375, rf: 335 },
  DET: { name: "Comerica Park", lf: 345, lfAlley: 370, cf: 412, rfAlley: 365, rf: 330 },
  HOU: { name: "Daikin Park", lf: 315, lfAlley: 362, cf: 409, rfAlley: 373, rf: 326 },
  KC: { name: "Kauffman Stadium", lf: 330, lfAlley: 387, cf: 410, rfAlley: 387, rf: 330 },
  LAA: { name: "Angel Stadium", lf: 330, lfAlley: 387, cf: 396, rfAlley: 370, rf: 330 },
  LAD: { name: "Dodger Stadium", lf: 330, lfAlley: 375, cf: 395, rfAlley: 375, rf: 330 },
  MIA: { name: "loanDepot park", lf: 344, lfAlley: 386, cf: 400, rfAlley: 392, rf: 335 },
  MIL: { name: "American Family Field", lf: 344, lfAlley: 371, cf: 400, rfAlley: 374, rf: 345 },
  MIN: { name: "Target Field", lf: 339, lfAlley: 377, cf: 404, rfAlley: 367, rf: 328 },
  NYM: { name: "Citi Field", lf: 335, lfAlley: 358, cf: 408, rfAlley: 375, rf: 330 },
  NYY: { name: "Yankee Stadium", lf: 318, lfAlley: 399, cf: 408, rfAlley: 385, rf: 314 },
  PHI: { name: "Citizens Bank Park", lf: 329, lfAlley: 374, cf: 401, rfAlley: 369, rf: 330 },
  PIT: { name: "PNC Park", lf: 325, lfAlley: 389, cf: 399, rfAlley: 375, rf: 320 },
  SD: { name: "Petco Park", lf: 336, lfAlley: 390, cf: 396, rfAlley: 391, rf: 322 },
  SEA: { name: "T-Mobile Park", lf: 331, lfAlley: 378, cf: 401, rfAlley: 381, rf: 326 },
  SF: { name: "Oracle Park", lf: 339, lfAlley: 364, cf: 399, rfAlley: 415, rf: 309 },
  STL: { name: "Busch Stadium", lf: 336, lfAlley: 375, cf: 400, rfAlley: 375, rf: 335 },
  TB: { name: "Tropicana Field", lf: 315, lfAlley: 370, cf: 404, rfAlley: 370, rf: 322 },
  TEX: { name: "Globe Life Field", lf: 329, lfAlley: 372, cf: 407, rfAlley: 374, rf: 326 },
  TOR: { name: "Rogers Centre", lf: 328, lfAlley: 375, cf: 400, rfAlley: 375, rf: 328 },
  WSH: { name: "Nationals Park", lf: 336, lfAlley: 377, cf: 402, rfAlley: 370, rf: 335 },
};

export interface Point2 {
  x: number;
  y: number;
}

/**
 * Smooth (x, y) wall polygon for one park, `steps` points per quarter-arc
 * between the five measured markers. Catmull-Rom through the five
 * (angle, distance) pairs — foul-line-to-foul-line, angle in degrees from
 * dead centre, negative toward LF/3B — then converted to the field's (x, y)
 * frame (x = sin(angle) * distance, y = cos(angle) * distance).
 */
export function parkPolygon(dims: ParkDimensions, stepsPerSegment = 10): Point2[] {
  const markers: [number, number][] = [
    [-45, dims.lf],
    [-22.5, dims.lfAlley],
    [0, dims.cf],
    [22.5, dims.rfAlley],
    [45, dims.rf],
  ];

  const at = (i: number) => markers[Math.max(0, Math.min(markers.length - 1, i))];
  const points: Point2[] = [];
  for (let seg = 0; seg < markers.length - 1; seg++) {
    const [a0, d0] = at(seg - 1);
    const [a1, d1] = at(seg);
    const [a2, d2] = at(seg + 1);
    const [a3, d3] = at(seg + 2);
    const steps = seg === markers.length - 2 ? stepsPerSegment + 1 : stepsPerSegment;
    for (let s = 0; s < steps; s++) {
      const t = s / stepsPerSegment;
      const angle = catmullRom(a0, a1, a2, a3, t);
      const dist = catmullRom(d0, d1, d2, d3, t);
      const rad = (angle * Math.PI) / 180;
      points.push({ x: Math.sin(rad) * dist, y: Math.cos(rad) * dist });
    }
  }
  return points;
}

function catmullRom(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const t2 = t * t;
  const t3 = t2 * t;
  return (
    0.5 *
    (2 * p1 +
      (-p0 + p2) * t +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
      (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
  );
}
