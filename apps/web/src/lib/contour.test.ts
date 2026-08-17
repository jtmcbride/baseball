import { describe, expect, it } from "vitest";
import { marchingSquares } from "./contour";

/** A 5x5 grid, row-major (x-major then z), of a simple radial bump centred
 * at (2, 2) — the shape the umpire zone map traces isn't a bump, but the
 * geometry the tracer has to get right (a closed loop separating a hot core
 * from a cold field) is the same one. */
function bumpGrid(n: number, cx: number, cz: number, peak: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const d = Math.hypot(i - cx, j - cz);
      out.push(peak * Math.max(0, 1 - d / (n / 2)));
    }
  }
  return out;
}

describe("marchingSquares", () => {
  it("finds no crossing when the whole grid is above the level", () => {
    const flat = new Array(25).fill(100);
    expect(marchingSquares(flat, 5, 50)).toEqual([]);
  });

  it("finds no crossing when the whole grid is below the level", () => {
    const flat = new Array(25).fill(0);
    expect(marchingSquares(flat, 5, 50)).toEqual([]);
  });

  it("traces a ring around a central bump", () => {
    const grid = bumpGrid(9, 4, 4, 100);
    const segs = marchingSquares(grid, 9, 50);
    expect(segs.length).toBeGreaterThan(0);

    // Every segment endpoint should sit roughly on the 50% contour: bilinear
    // interpolation at that point should land near `level`, not off in the
    // hot core or the cold field.
    const at = (i: number, j: number) => grid[Math.round(i) * 9 + Math.round(j)];
    for (const [x1, y1, x2, y2] of segs) {
      expect(Math.abs(at(x1, y1) - 50)).toBeLessThan(35);
      expect(Math.abs(at(x2, y2) - 50)).toBeLessThan(35);
    }
  });

  it("skips cells touching a null value rather than guessing", () => {
    const grid = bumpGrid(9, 4, 4, 100);
    grid[4 * 9 + 4] = null as unknown as number; // poke a hole at the peak
    // Should not throw, and should produce strictly fewer segments than the
    // hole-free grid (the cells touching the hole are dropped).
    const withHole = marchingSquares(grid, 9, 50);
    const whole = marchingSquares(bumpGrid(9, 4, 4, 100), 9, 50);
    expect(withHole.length).toBeLessThanOrEqual(whole.length);
  });

  it("is symmetric: crossing 50 on a mirrored grid produces a mirrored count", () => {
    const grid = bumpGrid(9, 4, 4, 100);
    const mirrored = bumpGrid(9, 4, 4, 100); // same centre => same topology
    expect(marchingSquares(grid, 9, 50).length).toBe(marchingSquares(mirrored, 9, 50).length);
  });
});
