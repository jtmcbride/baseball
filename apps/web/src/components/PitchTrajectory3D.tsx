/**
 * One pitch, flown in 3D from the batter's point of view.
 *
 * The scene itself — physics, coordinate mapping, camera, controls — lives in
 * `PitchScene3D`, which draws any number of pitches at once. This is the
 * single-pitch framing of it: no commit plane (there is nothing to tunnel
 * against with one pitch), and a caption that names the pitch and its velocity.
 */

import type { PitchTrajectory } from "../lib/api";
import { labelOf } from "../lib/scales";
import { PitchScene3D } from "./PitchScene3D";

export function PitchTrajectory3D({
  trajectory, width = 480, height = 340,
}: { trajectory: PitchTrajectory; width?: number; height?: number }) {
  return (
    <figure style={{ margin: 0 }}>
      <PitchScene3D
        pitches={[
          { id: "single", trajectory, pitchType: trajectory.pitch_type },
        ]}
        szTop={trajectory.sz_top}
        szBot={trajectory.sz_bot}
        stand={trajectory.stand}
        width={width}
        height={height}
        commitDistance={null}
      />
      <figcaption style={{ fontSize: 12, color: "var(--text-secondary)", marginTop: 2 }}>
        {labelOf(trajectory.pitch_type)}
        {trajectory.release_speed != null && ` · ${trajectory.release_speed.toFixed(1)} mph`}
      </figcaption>
    </figure>
  );
}
