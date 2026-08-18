/**
 * Monotone-chain convex hull, for the archetype outlines on the UMAP arsenal
 * map. Archetypes get a hull rather than their own hue — an N-hue palette
 * would fail the all-pairs CVD gate `scales.test.ts` enforces (at most three
 * hues, family-only) — so the outline plus a direct centroid label is the
 * whole visual encoding for "this archetype's rough extent".
 */

export interface Point2 {
  x: number;
  y: number;
}

function cross(o: Point2, a: Point2, b: Point2): number {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

/**
 * Convex hull vertices in counter-clockwise order, starting from the
 * lowest-x (then lowest-y) point. Fewer than 3 distinct points returns the
 * input as-is (a hull isn't meaningful, but callers shouldn't have to
 * special-case it).
 */
export function convexHull(points: Point2[]): Point2[] {
  const pts = [...points].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  const n = pts.length;
  if (n < 3) return pts;

  const lower: Point2[] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper: Point2[] = [];
  for (let i = n - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/** SVG path `d` for a closed hull polygon, or null if too few points to draw one. */
export function hullPath(hull: Point2[]): string | null {
  if (hull.length < 3) return null;
  return `M${hull.map((p) => `${p.x},${p.y}`).join("L")}Z`;
}
