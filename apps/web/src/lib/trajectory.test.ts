import { describe, expect, it } from "vitest";
import { PLATE_Y, reconstructFlight } from "./trajectory";

// A real pitch (game_pk=777314, at_bat=1, pitch=1, Skubal CH) fetched from
// /pitches/trajectory. plate_x/plate_z are Savant's own measurement, computed
// independently of vx0/vy0/vz0/ax/ay/az — agreement here is the whole point.
const REAL_PITCH = {
  release_pos_x: 2.06,
  release_pos_y: 53.92,
  release_pos_z: 5.89,
  vx0: -3.8224001736358852,
  vy0: -127.8111670831771,
  vz0: -0.9653097309231007,
  ax: 12.266138283627146,
  ay: 23.777471472373772,
  az: -24.966067478318195,
};
const ACTUAL_PLATE_X = 1.3880767526368951;
const ACTUAL_PLATE_Z = 3.5438214047630403;

describe("reconstructFlight", () => {
  it("lands on Savant's own plate_x/plate_z at the front of the plate", () => {
    const flight = reconstructFlight(REAL_PITCH);
    const [x, y, z] = flight.positionAt(flight.tauTotal);
    expect(y).toBeCloseTo(PLATE_Y, 6);
    expect(x).toBeCloseTo(ACTUAL_PLATE_X, 2);
    expect(z).toBeCloseTo(ACTUAL_PLATE_Z, 2);
  });

  it("starts exactly at the release point at tau=0", () => {
    const flight = reconstructFlight(REAL_PITCH);
    const [x, y, z] = flight.positionAt(0);
    expect(x).toBeCloseTo(REAL_PITCH.release_pos_x, 9);
    expect(y).toBeCloseTo(REAL_PITCH.release_pos_y, 9);
    expect(z).toBeCloseTo(REAL_PITCH.release_pos_z, 9);
  });

  it("flight time is in the realistic range for a thrown pitch", () => {
    const flight = reconstructFlight(REAL_PITCH);
    expect(flight.tauTotal).toBeGreaterThan(0.3);
    expect(flight.tauTotal).toBeLessThan(0.6);
  });

  it("speed decreases from release to plate (drag, not lift)", () => {
    const flight = reconstructFlight(REAL_PITCH);
    expect(flight.speedAt(flight.tauTotal)).toBeLessThan(flight.speedAt(0));
  });
});
