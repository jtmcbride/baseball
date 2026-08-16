import { describe, expect, it } from "vitest";
import {
  PITCH_FAMILY,
  divergingColor,
  familyOf,
  labelOf,
  markerPath,
  pitchColor,
  pitchShape,
} from "./scales";

describe("pitch taxonomy", () => {
  it("groups the fastball family", () => {
    for (const pt of ["FF", "SI", "FC", "FA"]) {
      expect(familyOf(pt)).toBe("fastball");
    }
  });

  it("groups breaking balls, including the sweeper", () => {
    for (const pt of ["SL", "ST", "CU", "KC", "SV"]) {
      expect(familyOf(pt)).toBe("breaking");
    }
  });

  it("groups offspeed", () => {
    expect(familyOf("CH")).toBe("offspeed");
    expect(familyOf("FS")).toBe("offspeed");
  });

  it("uses at most three hues", () => {
    // The movement plot is a scatter and so falls under the all-pairs CVD gate,
    // which the validated palette clears with three slots. A fourth hue here
    // would invalidate the palette check for the whole app.
    const hues = new Set(Object.keys(PITCH_FAMILY).map((pt) => pitchColor(pt)));
    expect(hues.size).toBeLessThanOrEqual(3);
  });

  it("handles unknown and null pitch types without throwing", () => {
    expect(familyOf(null)).toBe("offspeed");
    expect(familyOf("ZZ")).toBe("offspeed");
    expect(labelOf(null)).toBe("Unknown");
    expect(labelOf("ZZ")).toBe("ZZ");
  });

  it("gives distinct shapes within a family", () => {
    // Shape is the secondary encoding that lets 3 hues carry ~9 categories, so
    // same-colour pitches must not share a marker.
    const fastballs = ["FF", "SI", "FC", "FA"];
    const shapes = new Set(fastballs.map(pitchShape));
    expect(shapes.size).toBe(fastballs.length);

    const breaking = ["SL", "ST", "CU", "KC"];
    expect(new Set(breaking.map(pitchShape)).size).toBe(breaking.length);
  });

  it("keeps shape stable per pitch type", () => {
    // Colour and shape follow the entity, never its rank — a filter that drops
    // a series must not repaint the survivors.
    expect(pitchShape("SL")).toBe(pitchShape("SL"));
    expect(pitchColor("SL")).toBe(pitchColor("SL"));
  });
});

describe("markerPath", () => {
  it("produces a closed path for each shape", () => {
    for (const s of ["circle", "square", "triangle", "diamond"] as const) {
      const d = markerPath(s, 4);
      expect(d.length).toBeGreaterThan(0);
      expect(d.startsWith("M")).toBe(true);
    }
  });
});

describe("divergingColor", () => {
  it("returns the neutral midpoint near the centre", () => {
    expect(divergingColor(0.37, 0.37, 0.2)).toBe("var(--div-mid)");
  });

  it("sends values above the midpoint to the warm arm", () => {
    expect(divergingColor(0.55, 0.37, 0.2)).toContain("warm");
  });

  it("sends values below the midpoint to the cool arm", () => {
    expect(divergingColor(0.15, 0.37, 0.2)).toContain("cool");
  });

  it("clamps beyond the half-range instead of running off the ramp", () => {
    expect(divergingColor(99, 0.37, 0.2)).toBe("var(--div-warm-5)");
    expect(divergingColor(-99, 0.37, 0.2)).toBe("var(--div-cool-5)");
  });

  it("renders nothing for non-finite values", () => {
    expect(divergingColor(NaN, 0.37, 0.2)).toBe("transparent");
  });

  it("only ever emits CSS custom properties", () => {
    // Charts must not hardcode hex: light/dark has to swap in one place.
    for (let v = -1; v <= 1; v += 0.1) {
      const c = divergingColor(v, 0, 1);
      expect(c === "transparent" || c.startsWith("var(--")).toBe(true);
    }
  });
});
