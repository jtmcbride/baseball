/**
 * Bin-edges/counts builder for the swing-length distribution (viz #19). Plain
 * SVG bars consume this directly — no charting library, same as every other
 * scatter/heatmap in this app.
 */

export interface Bin {
  x0: number;
  x1: number;
  count: number;
}

/**
 * Fixed-width bins spanning `[min(values), max(values)]`. A single distinct
 * value (or empty input) returns an empty bin list rather than dividing by a
 * zero-width range — callers should treat that as "not enough data to draw".
 */
export function histogram(values: number[], binCount = 20): Bin[] {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (min === max) return [];

  const width = (max - min) / binCount;
  const bins: Bin[] = Array.from({ length: binCount }, (_, i) => ({
    x0: min + i * width,
    x1: min + (i + 1) * width,
    count: 0,
  }));

  for (const v of values) {
    const idx = Math.min(binCount - 1, Math.floor((v - min) / width));
    bins[idx].count++;
  }
  return bins;
}
