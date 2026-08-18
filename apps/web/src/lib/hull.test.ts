import { describe, expect, it } from "vitest";
import { convexHull, hullPath, type Point2 } from "./hull";

describe("convexHull", () => {
  it("returns the input for fewer than 3 points", () => {
    const pts: Point2[] = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
    expect(convexHull(pts)).toEqual(pts);
  });

  it("finds the hull of a square with an interior point", () => {
    const pts: Point2[] = [
      { x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }, { x: 0, y: 10 },
      { x: 5, y: 5 }, // interior -- must not appear in the hull
    ];
    const hull = convexHull(pts);
    expect(hull).toHaveLength(4);
    expect(hull).not.toContainEqual({ x: 5, y: 5 });
  });

  it("keeps every point of an already-convex set", () => {
    const triangle: Point2[] = [{ x: 0, y: 0 }, { x: 4, y: 0 }, { x: 2, y: 4 }];
    expect(convexHull(triangle)).toHaveLength(3);
  });

  it("winds counter-clockwise", () => {
    const pts: Point2[] = [{ x: 0, y: 0 }, { x: 4, y: 0 }, { x: 4, y: 4 }, { x: 0, y: 4 }];
    const hull = convexHull(pts);
    let signedArea = 0;
    for (let i = 0; i < hull.length; i++) {
      const a = hull[i];
      const b = hull[(i + 1) % hull.length];
      signedArea += a.x * b.y - b.x * a.y;
    }
    expect(signedArea).toBeGreaterThan(0);
  });

  it("handles duplicate points without crashing", () => {
    const pts: Point2[] = [{ x: 1, y: 1 }, { x: 1, y: 1 }, { x: 1, y: 1 }];
    expect(() => convexHull(pts)).not.toThrow();
  });
});

describe("hullPath", () => {
  it("returns null for fewer than 3 points", () => {
    expect(hullPath([{ x: 0, y: 0 }])).toBeNull();
  });

  it("builds a closed SVG path for a triangle", () => {
    const hull: Point2[] = [{ x: 0, y: 0 }, { x: 4, y: 0 }, { x: 2, y: 4 }];
    expect(hullPath(hull)).toBe("M0,0L4,0L2,4Z");
  });
});
