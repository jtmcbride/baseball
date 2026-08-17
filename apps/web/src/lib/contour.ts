/**
 * Marching squares over a dense row-major (x-major then z) grid, contoured at
 * `level`. Used for the umpire zone map (viz #13): a raw called-strike-rate
 * heatmap is readable as color, but "where is this umpire's actual zone
 * boundary" is a line, not a gradient — this traces the 50% crossing so it can
 * be drawn on top of the rulebook rectangle `StrikeZoneHeatmap` already has.
 *
 * Segments are returned unstitched (a flat list of line pieces, not closed
 * polygons) — for a plain contour overlay a scatter of segments renders
 * identically to a stitched polyline, and stitching is real complexity this
 * chart doesn't need.
 */

type Point = [number, number];
type Segment = [number, number, number, number];

function interp(a: number, b: number, va: number, vb: number, level: number): number {
  return va === vb ? a : a + ((level - va) / (vb - va)) * (b - a);
}

export function marchingSquares(
  values: (number | null)[],
  n: number,
  level: number,
): Segment[] {
  const at = (i: number, j: number) => values[i * n + j];
  const segs: Segment[] = [];

  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < n - 1; j++) {
      const v00 = at(i, j);
      const v10 = at(i + 1, j);
      const v11 = at(i + 1, j + 1);
      const v01 = at(i, j + 1);
      if (v00 == null || v10 == null || v11 == null || v01 == null) continue;
      if (!Number.isFinite(v00) || !Number.isFinite(v10) || !Number.isFinite(v11) || !Number.isFinite(v01)) {
        continue;
      }

      const code =
        (v00 >= level ? 1 : 0) |
        (v10 >= level ? 2 : 0) |
        (v11 >= level ? 4 : 0) |
        (v01 >= level ? 8 : 0);
      if (code === 0 || code === 15) continue;

      const bottom: Point = [interp(i, i + 1, v00, v10, level), j];
      const right: Point = [i + 1, interp(j, j + 1, v10, v11, level)];
      const top: Point = [interp(i, i + 1, v01, v11, level), j + 1];
      const left: Point = [i, interp(j, j + 1, v00, v01, level)];

      const pairs: [Point, Point][] = (() => {
        switch (code) {
          case 1:
          case 14:
            return [[left, bottom]];
          case 2:
          case 13:
            return [[bottom, right]];
          case 3:
          case 12:
            return [[left, right]];
          case 4:
          case 11:
            return [[right, top]];
          case 6:
          case 9:
            return [[bottom, top]];
          case 7:
          case 8:
            return [[top, left]];
          case 5:
          case 10:
            // Saddle: ambiguous which diagonal pair connects, but both
            // resolutions draw the same two segments — the choice only
            // matters for filled regions, not a bare contour line.
            return [
              [left, bottom],
              [right, top],
            ];
          default:
            return [];
        }
      })();

      for (const [[x1, y1], [x2, y2]] of pairs) segs.push([x1, y1, x2, y2]);
    }
  }
  return segs;
}
