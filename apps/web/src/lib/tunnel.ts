/**
 * Pitch-tunnelling geometry: what two flights look like where the batter has to
 * commit, and how to line several flights up in one animation.
 *
 * Overlaying trajectories is only interesting because of what it measures. Two
 * pitches that end up 18 inches apart at the plate may have been an inch apart
 * when the hitter had to start his swing — that early separation, not the final
 * one, is what makes a pitch play up.
 *
 * The commit point is a DISTANCE from the plate, not a time. That distinction is
 * the whole correctness of this module: sampling two flights at the same elapsed
 * time from release puts a 95mph fastball several feet closer to the plate than
 * an 86mph slider, and the gap you measure is then mostly the velocity
 * difference rather than the pitches diverging. Sampling both where they are the
 * same distance out removes velocity from the measurement and leaves the shape,
 * which is what "do these two tunnel?" is asking. (It is also the convention the
 * published tunnelling work uses.)
 *
 * `COMMIT_DISTANCE_FT` is a staging constant, not physics: 23.8ft from the plate
 * is the commonly cited commit point, about 167ms out for a mid-90s fastball.
 * It is surfaced and adjustable in the UI so a reader sees the assumption rather
 * than inheriting it. Everything else here is exact given the flights.
 */

import type { Flight, Vec3 } from "./trajectory";

/** Feet from the front of the plate at which a hitter is taken to be committed. */
export const COMMIT_DISTANCE_FT = 23.8;

export const FT_TO_IN = 12;

/** Time since release at which the flight is `distance` feet from the plate. */
export function commitTau(flight: Flight, distance = COMMIT_DISTANCE_FT): number {
  return flight.tauAtY(distance);
}

/** Milliseconds the hitter has left once the ball reaches `distance`. */
export function reactionMs(flight: Flight, distance = COMMIT_DISTANCE_FT): number {
  return (flight.tauTotal - commitTau(flight, distance)) * 1000;
}

export function distanceFt(a: Vec3, b: Vec3): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  const dz = a[2] - b[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function maxPairwiseIn(points: Vec3[]): number {
  let max = 0;
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      max = Math.max(max, distanceFt(points[i], points[j]));
    }
  }
  return max * FT_TO_IN;
}

/**
 * Largest gap, in inches, between any two flights at the commit distance.
 *
 * Fewer than two flights has no pair to separate and returns 0, not NaN.
 */
export function tunnelSpreadIn(flights: Flight[], distance = COMMIT_DISTANCE_FT): number {
  return maxPairwiseIn(flights.map((f) => f.positionAt(commitTau(f, distance))));
}

/** Largest gap, in inches, between any two flights where they cross the plate. */
export function plateSpreadIn(flights: Flight[]): number {
  return maxPairwiseIn(flights.map((f) => f.positionAt(f.tauTotal)));
}

/**
 * Plate separation per inch of commit-point separation.
 *
 * The number a pitch-design conversation actually wants: a ratio above ~1 means
 * the pair keeps diverging after the hitter is committed, which is the point of
 * throwing them from the same tunnel. Undefined (null, not Infinity) when the
 * pitches are indistinguishable at the commit point — a ratio with a zero
 * denominator would render as a spectacular and meaningless number.
 */
export function tunnelRatio(flights: Flight[], distance = COMMIT_DISTANCE_FT): number | null {
  const tunnel = tunnelSpreadIn(flights, distance);
  if (tunnel < 0.05) return null;
  return plateSpreadIn(flights) / tunnel;
}

export type SyncMode = "release" | "plate";

/**
 * Per-flight animation start offsets, in seconds of physics time.
 *
 * `release` starts every ball at once — the honest view, where a slower pitch
 * visibly falls behind and arrives late, exactly as it does to a hitter who has
 * already committed.
 *
 * `plate` staggers the starts so every ball crosses the plate on the same frame.
 * That is NOT what the hitter sees, and it is right for exactly one question:
 * "where do these pitches sit at the same distance out?" — it removes the
 * velocity difference from the animation and leaves the shape. Both are offered
 * because each one lies about what the other tells the truth about.
 */
export function startOffsets(flights: Flight[], mode: SyncMode): number[] {
  if (mode === "release") return flights.map(() => 0);
  const longest = Math.max(0, ...flights.map((f) => f.tauTotal));
  return flights.map((f) => longest - f.tauTotal);
}
