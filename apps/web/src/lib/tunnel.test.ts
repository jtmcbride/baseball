import { describe, expect, it } from "vitest";
import { reconstructFlight } from "./trajectory";
import {
  COMMIT_DISTANCE_FT, commitTau, plateSpreadIn, reactionMs, startOffsets,
  tunnelRatio, tunnelSpreadIn,
} from "./tunnel";

// Real pitches, not synthetic ones, so the tunnel numbers below are numbers a
// pitching coach could sanity-check. The changeup is the same pitch
// trajectory.test.ts validates against Savant's own plate_x/plate_z.
const CHANGEUP = {
  release_pos_x: 2.06, release_pos_y: 53.92, release_pos_z: 5.89,
  vx0: -3.8224001736358852, vy0: -127.8111670831771, vz0: -0.9653097309231007,
  ax: 12.266138283627146, ay: 23.777471472373772, az: -24.966067478318195,
};
const FASTBALL = {
  release_pos_x: 2.01, release_pos_y: 53.85, release_pos_z: 5.95,
  vx0: -4.61, vy0: -140.2, vz0: -3.05,
  ax: 11.9, ay: 28.4, az: -12.1,
};

describe("commitTau", () => {
  it("puts the ball at the commit distance, not at a shared clock time", () => {
    for (const p of [CHANGEUP, FASTBALL]) {
      const f = reconstructFlight(p);
      const [, y] = f.positionAt(commitTau(f));
      expect(y).toBeCloseTo(COMMIT_DISTANCE_FT, 6);
    }
  });

  it("the faster pitch leaves the hitter less time from the same distance", () => {
    expect(reactionMs(reconstructFlight(FASTBALL))).toBeLessThan(
      reactionMs(reconstructFlight(CHANGEUP)),
    );
  });

  it("reaction time out of 23.8ft is in the known ballpark for a big-league pitch", () => {
    const ms = reactionMs(reconstructFlight(FASTBALL));
    expect(ms).toBeGreaterThan(140);
    expect(ms).toBeLessThan(200);
  });
});

describe("tunnelSpreadIn", () => {
  it("is zero for a single flight — there is no pair to separate", () => {
    expect(tunnelSpreadIn([reconstructFlight(CHANGEUP)])).toBe(0);
  });

  it("a flight compared with itself never separates", () => {
    const f = reconstructFlight(FASTBALL);
    expect(tunnelSpreadIn([f, f])).toBeCloseTo(0, 12);
  });

  it("two pitches from one arm slot are inches apart at the commit point", () => {
    const flights = [reconstructFlight(CHANGEUP), reconstructFlight(FASTBALL)];
    const tunnel = tunnelSpreadIn(flights);
    expect(tunnel).toBeGreaterThan(0);
    expect(tunnel).toBeLessThan(24);
  });

  it("separates further by the plate than at the commit point", () => {
    const flights = [reconstructFlight(CHANGEUP), reconstructFlight(FASTBALL)];
    expect(plateSpreadIn(flights)).toBeGreaterThan(tunnelSpreadIn(flights));
  });

  it("grows as the commit point moves later — the paths are still diverging", () => {
    const flights = [reconstructFlight(CHANGEUP), reconstructFlight(FASTBALL)];
    expect(tunnelSpreadIn(flights, 15)).toBeGreaterThan(tunnelSpreadIn(flights, 30));
  });
});

describe("tunnelRatio", () => {
  it("is plate spread over commit spread", () => {
    const flights = [reconstructFlight(CHANGEUP), reconstructFlight(FASTBALL)];
    expect(tunnelRatio(flights)!).toBeCloseTo(
      plateSpreadIn(flights) / tunnelSpreadIn(flights), 9,
    );
  });

  it("is null rather than Infinity when the pitches are on top of each other", () => {
    const f = reconstructFlight(CHANGEUP);
    expect(tunnelRatio([f, f])).toBeNull();
  });
});

describe("startOffsets", () => {
  it("release sync starts every ball on the same frame", () => {
    const flights = [reconstructFlight(CHANGEUP), reconstructFlight(FASTBALL)];
    expect(startOffsets(flights, "release")).toEqual([0, 0]);
  });

  it("plate sync makes every ball arrive on the same frame", () => {
    const flights = [reconstructFlight(CHANGEUP), reconstructFlight(FASTBALL)];
    const offsets = startOffsets(flights, "plate");
    const arrivals = flights.map((f, i) => offsets[i] + f.tauTotal);
    expect(arrivals[0]).toBeCloseTo(arrivals[1], 12);
    // The slower pitch is the one already in flight when the faster is released.
    expect(Math.min(...offsets)).toBe(0);
  });
});
