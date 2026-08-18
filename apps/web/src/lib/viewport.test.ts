import { describe, expect, it } from "vitest";
import {
  fitViewport, MAX_SCALE, MIN_SCALE, nearestPoint, panBy, resetViewport, toData, toScreen, zoomAt,
} from "./viewport";

describe("toScreen / toData", () => {
  it("round-trips a point through screen and back to data space", () => {
    const vp = { x: 10, y: -5, scale: 2 };
    const [sx, sy] = toScreen(vp, 12, 3);
    const [dx, dy] = toData(vp, sx, sy);
    expect(dx).toBeCloseTo(12);
    expect(dy).toBeCloseTo(3);
  });

  it("at the identity viewport, screen equals data", () => {
    const vp = resetViewport();
    expect(toScreen(vp, 5, 7)).toEqual([5, 7]);
  });
});

describe("zoomAt", () => {
  it("keeps the data point under the cursor fixed on screen", () => {
    const vp = { x: 0, y: 0, scale: 1 };
    const cursor: [number, number] = [100, 50];
    const [beforeX, beforeY] = toData(vp, ...cursor);
    const zoomed = zoomAt(vp, cursor[0], cursor[1], 2);
    const [afterSx, afterSy] = toScreen(zoomed, beforeX, beforeY);
    expect(afterSx).toBeCloseTo(cursor[0]);
    expect(afterSy).toBeCloseTo(cursor[1]);
  });

  it("clamps scale to MAX_SCALE", () => {
    const vp = { x: 0, y: 0, scale: MAX_SCALE };
    const zoomed = zoomAt(vp, 0, 0, 10);
    expect(zoomed.scale).toBe(MAX_SCALE);
  });

  it("clamps scale to MIN_SCALE", () => {
    const vp = { x: 0, y: 0, scale: MIN_SCALE };
    const zoomed = zoomAt(vp, 0, 0, 0.01);
    expect(zoomed.scale).toBe(MIN_SCALE);
  });
});

describe("panBy", () => {
  it("moves the data point under a fixed screen position by the inverse delta", () => {
    const vp = { x: 0, y: 0, scale: 2 };
    const panned = panBy(vp, 20, -10);
    // Panning right by 20 screen px at scale 2 shifts the visible data
    // window left by 10 data units.
    expect(panned.x).toBeCloseTo(-10);
    expect(panned.y).toBeCloseTo(5);
  });

  it("does not change scale", () => {
    const vp = { x: 0, y: 0, scale: 3 };
    expect(panBy(vp, 5, 5).scale).toBe(3);
  });
});

describe("fitViewport", () => {
  it("centres the data bounds within the screen area", () => {
    const vp = fitViewport({ xMin: -10, xMax: 10, yMin: -5, yMax: 5 }, 400, 200, 0);
    const [sx, sy] = toScreen(vp, 0, 0);
    expect(sx).toBeCloseTo(200);
    expect(sy).toBeCloseTo(100);
  });

  it("handles degenerate (zero-extent) bounds without dividing by zero", () => {
    const vp = fitViewport({ xMin: 5, xMax: 5, yMin: 5, yMax: 5 }, 400, 200);
    expect(Number.isFinite(vp.scale)).toBe(true);
    expect(vp.scale).toBeLessThanOrEqual(MAX_SCALE);
  });
});

describe("nearestPoint", () => {
  const points = [
    { x: 0, y: 0, id: "a" },
    { x: 10, y: 0, id: "b" },
    { x: 10, y: 10, id: "c" },
  ];

  it("finds the closest point within range", () => {
    const vp = { x: 0, y: 0, scale: 1 };
    const found = nearestPoint(points, 1, 0.5, vp);
    expect(found?.id).toBe("a");
  });

  it("returns null when nothing is within maxDistScreen", () => {
    const vp = { x: 0, y: 0, scale: 1 };
    const found = nearestPoint(points, 1000, 1000, vp);
    expect(found).toBeNull();
  });

  it("returns null for an empty point list", () => {
    const vp = { x: 0, y: 0, scale: 1 };
    expect(nearestPoint([], 0, 0, vp)).toBeNull();
  });

  it("scales the hit-test radius by the current zoom", () => {
    // At scale=1, maxDistScreen=16 covers a point 10 data-units away.
    // At scale=0.5, the same 16 screen px covers a smaller data radius (32),
    // so a point 10 units away is still in range -- but a point 40 away
    // should only be reachable when zoomed further out.
    const vp = { x: 0, y: 0, scale: 0.5 };
    const far = [{ x: 30, y: 0, id: "far" }];
    expect(nearestPoint(far, 0, 0, vp, 16)).not.toBeNull();
    const vpZoomedIn = { x: 0, y: 0, scale: 4 };
    expect(nearestPoint(far, 0, 0, vpZoomedIn, 16)).toBeNull();
  });
});
