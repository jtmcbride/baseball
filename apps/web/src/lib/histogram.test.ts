import { describe, expect, it } from "vitest";
import { histogram } from "./histogram";

describe("histogram", () => {
  it("returns an empty bin list for no values", () => {
    expect(histogram([])).toEqual([]);
  });

  it("returns an empty bin list when every value is equal", () => {
    expect(histogram([5, 5, 5])).toEqual([]);
  });

  it("bins a spread of values into the requested count", () => {
    const bins = histogram([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 5);
    expect(bins).toHaveLength(5);
    expect(bins.reduce((s, b) => s + b.count, 0)).toBe(11);
  });

  it("places the maximum value in the last bin, not off the end", () => {
    const bins = histogram([0, 10], 5);
    expect(bins[4].count).toBe(1);
    expect(bins[0].count).toBe(1);
  });

  it("defaults to 20 bins", () => {
    const bins = histogram([0, 1, 2, 3]);
    expect(bins).toHaveLength(20);
  });
});
