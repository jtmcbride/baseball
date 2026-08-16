/**
 * Exact pitch-flight reconstruction from Statcast's 9-parameter physics fit.
 *
 * `vx0/vy0/vz0/ax/ay/az` are NOT valid at the release point — they are the
 * fitted constant-acceleration trajectory evaluated at the fixed reference
 * y=50ft (a PITCHf/x-era convention Statcast still reports under). The actual
 * release point (`release_pos_x/y/z`) sits a few hundredths of a second
 * earlier in the flight, at y≈54ft. To animate release-to-plate without lying
 * about the shape of the path, we solve backward from y=50 to the release
 * y to get the velocity AT release, then integrate forward from there to
 * y=17/12 (the front edge of the plate, where Savant's own `plate_x`/`plate_z`
 * are measured).
 *
 * Validated against real plate_x/plate_z: ~0.003ft mean error across a
 * 200-pitch sample — see the backend's `TestTrajectory` for the same check.
 * Do not change the y=50 or y=17/12 reference points without re-validating
 * both here and in `apps/api/.../routers/pitches.py`.
 */

export interface PhysicsParams {
  release_pos_x: number;
  release_pos_y: number;
  release_pos_z: number;
  vx0: number;
  vy0: number;
  vz0: number;
  ax: number;
  ay: number;
  az: number;
}

export type Vec3 = [number, number, number];

const Y0_REF = 50; // ft — Statcast's fixed reference distance for vx0/vy0/vz0/ax/ay/az
export const PLATE_Y = 17 / 12; // ft — front edge of home plate

/** Time at which y(t) = yTarget, given y(t) = y0 + vy0*t + 0.5*ay*t^2. */
function solveT(y0: number, vy0: number, ay: number, yTarget: number): number {
  const a = 0.5 * ay;
  const b = vy0;
  const c = y0 - yTarget;
  const disc = Math.max(b * b - 4 * a * c, 0);
  const sq = Math.sqrt(disc);
  const r1 = (-b + sq) / (2 * a);
  const r2 = (-b - sq) / (2 * a);
  // Two mathematical roots; the physically meaningful one for a pitch (release
  // a few hundredths of a second from y=50, flight well under a second) is
  // always the smaller in magnitude — the other is a far-future artifact of
  // the quadratic fit having no physical meaning past the tracked window.
  return Math.abs(r1) < Math.abs(r2) ? r1 : r2;
}

export interface Flight {
  /** Seconds from release to the front of the plate. */
  tauTotal: number;
  /** Position in Statcast feet (x, y, z) at time `tau` since release. */
  positionAt(tau: number): Vec3;
  /** Speed in mph at time `tau` since release. */
  speedAt(tau: number): number;
}

export function reconstructFlight(p: PhysicsParams): Flight {
  const tRelease = solveT(Y0_REF, p.vy0, p.ay, p.release_pos_y);
  const vxR = p.vx0 + p.ax * tRelease;
  const vyR = p.vy0 + p.ay * tRelease;
  const vzR = p.vz0 + p.az * tRelease;

  const tauTotal = solveT(p.release_pos_y, vyR, p.ay, PLATE_Y);

  const positionAt = (tau: number): Vec3 => [
    p.release_pos_x + vxR * tau + 0.5 * p.ax * tau * tau,
    p.release_pos_y + vyR * tau + 0.5 * p.ay * tau * tau,
    p.release_pos_z + vzR * tau + 0.5 * p.az * tau * tau,
  ];

  const FT_S_TO_MPH = 3600 / 5280;
  const speedAt = (tau: number): number => {
    const vx = vxR + p.ax * tau;
    const vy = vyR + p.ay * tau;
    const vz = vzR + p.az * tau;
    return Math.sqrt(vx * vx + vy * vy + vz * vz) * FT_S_TO_MPH;
  };

  return { tauTotal, positionAt, speedAt };
}
