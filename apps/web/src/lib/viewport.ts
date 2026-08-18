/**
 * Pan/zoom transform math for the UMAP arsenal map.
 *
 * Nothing like this exists elsewhere in the app — the only zoom anywhere is
 * Three.js `OrbitControls` in the 3D pitch trajectory viz, which owns its own
 * camera. A 2D scatter has no such library underneath (no d3 imports, no
 * router, hand-rolled SVG everywhere), so this is the plain math: a single
 * `{x, y, scale}` viewport applied to data-space points to get screen-space
 * ones, and back.
 *
 * The viewport transform is meant to be applied to ONE `<g transform>`
 * wrapping the points group, not by recomputing every point's screen
 * position on every frame — with ~4,200 points that recomputation is the
 * difference between an SVG `transform` (free) and thousands of React
 * re-renders per drag frame.
 */

export interface Viewport {
  /** Data-space x that maps to the left edge of the viewport, at scale 1. */
  x: number;
  /** Data-space y that maps to the top edge of the viewport, at scale 1. */
  y: number;
  scale: number;
}

export const MIN_SCALE = 0.5;
export const MAX_SCALE = 40;

function clampScale(scale: number): number {
  return Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale));
}

export function toScreen(vp: Viewport, dataX: number, dataY: number): [number, number] {
  return [(dataX - vp.x) * vp.scale, (dataY - vp.y) * vp.scale];
}

export function toData(vp: Viewport, screenX: number, screenY: number): [number, number] {
  return [screenX / vp.scale + vp.x, screenY / vp.scale + vp.y];
}

/**
 * Zoom by `factor` (>1 zooms in) while keeping the data point currently under
 * `(screenX, screenY)` fixed on screen — the standard "zoom toward the
 * cursor" behaviour, not zoom-toward-origin.
 */
export function zoomAt(vp: Viewport, screenX: number, screenY: number, factor: number): Viewport {
  const scale = clampScale(vp.scale * factor);
  const [dataX, dataY] = toData(vp, screenX, screenY);
  // Solve for the new (x, y) that keeps (dataX, dataY) at the same screen
  // position under the new scale.
  return { scale, x: dataX - screenX / scale, y: dataY - screenY / scale };
}

/** Pan by a screen-space delta (e.g. mouse movement in pixels). */
export function panBy(vp: Viewport, dxScreen: number, dyScreen: number): Viewport {
  return { ...vp, x: vp.x - dxScreen / vp.scale, y: vp.y - dyScreen / vp.scale };
}

export function resetViewport(): Viewport {
  return { x: 0, y: 0, scale: 1 };
}

/**
 * Fit a viewport so that data-space bounds [xMin,xMax] x [yMin,yMax] centre
 * within a `width` x `height` screen area, at the given padding fraction.
 */
export function fitViewport(
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number },
  width: number,
  height: number,
  pad = 0.1,
): Viewport {
  const dw = Math.max(bounds.xMax - bounds.xMin, 1e-6);
  const dh = Math.max(bounds.yMax - bounds.yMin, 1e-6);
  const scale = clampScale((1 - pad) * Math.min(width / dw, height / dh));
  const cx = (bounds.xMin + bounds.xMax) / 2;
  const cy = (bounds.yMin + bounds.yMax) / 2;
  return { scale, x: cx - width / 2 / scale, y: cy - height / 2 / scale };
}

export interface Locatable {
  x: number;
  y: number;
}

/**
 * Nearest point to a data-space location, for hover/click hit-testing.
 *
 * A linear scan over ~4,200 points is fine here — it runs once per pointer
 * event, not per animation frame, and a spatial index would be real
 * complexity for a scatter this size. Returns null if `points` is empty or
 * nothing falls within `maxDistScreen` screen pixels (converted to data
 * space via the viewport's current scale).
 */
export function nearestPoint<T extends Locatable>(
  points: T[],
  dataX: number,
  dataY: number,
  vp: Viewport,
  maxDistScreen = 16,
): T | null {
  let best: T | null = null;
  let bestDist = Infinity;
  for (const p of points) {
    const d = Math.hypot(p.x - dataX, p.y - dataY);
    if (d < bestDist) {
      bestDist = d;
      best = p;
    }
  }
  const maxDistData = maxDistScreen / vp.scale;
  return best !== null && bestDist <= maxDistData ? best : null;
}
