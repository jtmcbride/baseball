/**
 * Animated 3D pitch flight, from the batter's point of view.
 *
 * The path is not an approximation: it's `reconstructFlight()` (lib/trajectory.ts)
 * evaluated at 60fps, the same exact-physics reconstruction validated against
 * Savant's own plate_x/plate_z on the backend. Everything else in this file —
 * camera placement, ball size, ground plane, slow-motion factor — is staging,
 * not physics, and is free to be approximate.
 *
 * Coordinate mapping: three.x = -statcast x (left/right), three.y = statcast z
 * (height, up), three.z = statcast y (distance from plate; plate at 0, rubber
 * at 60.5). The camera sits fixed near the plate and does not track the ball —
 * a real batter's head doesn't move either.
 *
 * The x negation matters: (x, y, z) -> (x, z, y) swaps two axes, which is a
 * parity-flipping transform — it silently turns Statcast's right-handed
 * system into a left-handed one, and Three.js assumes right-handed throughout
 * (camera orientation, cross products). Left uncorrected, the whole scene
 * mirrors left-right — a pitch that actually broke to the batter's right
 * would render as breaking left. Verified against Statcast's own plate_x
 * convention (positive = batter's right, toward first base) before shipping.
 */

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import type { PitchTrajectory } from "../lib/api";
import { familyColor, familyOf, labelOf } from "../lib/scales";
import { PLATE_Y, reconstructFlight } from "../lib/trajectory";

const PLATE_HALF_FT = 0.83;
const RUBBER_Y = 60.5;
const BALL_RADIUS_FT = 0.121;
// Real flight is ~0.4-0.5s — too fast to read. Stretch it for legibility; the
// SHAPE of the path is exact regardless of playback speed.
const SLOWMO = 4;

function cssColor(el: Element, varName: string): number {
  const raw = getComputedStyle(el).getPropertyValue(varName).trim();
  return new THREE.Color(raw || "#888888").getHex();
}

// Drawn at y=PLATE_Y (front edge, 17/12ft), not y=0 (the plate's back tip) —
// that's where plate_x/plate_z are measured and where the flight actually
// terminates. Drawing this at y=0 would leave the ball visibly landing short
// of its own strike zone.
function strikeZoneGeometry(szTop: number, szBot: number): THREE.BufferGeometry {
  const pts = [
    [-PLATE_HALF_FT, szBot], [PLATE_HALF_FT, szBot],
    [PLATE_HALF_FT, szTop], [-PLATE_HALF_FT, szTop],
    [-PLATE_HALF_FT, szBot],
  ].map(([x, z]) => new THREE.Vector3(x, z, PLATE_Y));
  return new THREE.BufferGeometry().setFromPoints(pts);
}

function homePlateGeometry(): THREE.BufferGeometry {
  // Regulation shape, flat on the ground (y=0), point toward the pitcher (+z).
  const w = PLATE_HALF_FT;
  const pts = [
    [-w, 0], [w, 0], [w, 0.7], [0, 1.1], [-w, 0.7], [-w, 0],
  ].map(([x, z]) => new THREE.Vector3(x, 0.01, z));
  return new THREE.BufferGeometry().setFromPoints(pts);
}

export function PitchTrajectory3D({
  trajectory, width = 480, height = 340,
}: { trajectory: PitchTrajectory; width?: number; height?: number }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [replayKey, setReplayKey] = useState(0);
  const [phase, setPhase] = useState<"flight" | "done">("flight");

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const flight = reconstructFlight(trajectory);
    const toThree = (x: number, y: number, z: number) => new THREE.Vector3(-x, z, y);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(cssColor(container, "--page"));

    const camera = new THREE.PerspectiveCamera(62, width / height, 0.1, 200);
    // Just behind and above the batter's head, offset toward whichever side
    // they stand — close enough to read as "their view" but pulled back far
    // enough to fit the pitcher AND the strike zone in frame at once (an
    // actual eye position right on the plate can't see its own zone — it's
    // beneath/around them, not a floating object). Aimed at the zone center so
    // the ball grows from a distant point near the pitcher into a full-size
    // ball crossing dead centre, which is also the least distorted composition
    // for a wide-angle lens this close to the subject.
    //
    // A right-handed batter's box sits on the third-base side (negative
    // Statcast x); with three.x = -statcast_x that's positive three.x — hence
    // R -> +1.6, not -1.6.
    const boxX = trajectory.stand === "L" ? -1.6 : 1.6;
    const zoneMidZ = (trajectory.sz_top + trajectory.sz_bot) / 2;
    camera.position.set(boxX, 6.4, -6);
    camera.lookAt(0, zoneMidZ, PLATE_Y);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const gridlineColor = cssColor(container, "--gridline");
    const axisColor = cssColor(container, "--axis");
    const textColor = cssColor(container, "--text-secondary");
    const ballColor = cssColor(container, "--text-primary");
    const familyColorHex = cssColor(container, familyColor(familyOf(trajectory.pitch_type)));

    // Ground: a plain, muted plane — just enough for depth cues.
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(30, 90),
      new THREE.MeshBasicMaterial({ color: gridlineColor, transparent: true, opacity: 0.35 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.set(0, 0, RUBBER_Y / 2);
    scene.add(ground);

    // Rubber-to-plate centerline, for depth orientation.
    const centerline = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0.01, 0),
        new THREE.Vector3(0, 0.01, RUBBER_Y),
      ]),
      new THREE.LineBasicMaterial({ color: axisColor, transparent: true, opacity: 0.5 }),
    );
    scene.add(centerline);

    const plate = new THREE.Line(
      homePlateGeometry(),
      new THREE.LineBasicMaterial({ color: textColor }),
    );
    scene.add(plate);

    const rubber = new THREE.Mesh(
      new THREE.BoxGeometry(2, 0.05, 0.5),
      new THREE.MeshBasicMaterial({ color: axisColor }),
    );
    rubber.position.set(0, 0.025, RUBBER_Y);
    scene.add(rubber);

    const zone = new THREE.Line(
      strikeZoneGeometry(trajectory.sz_top, trajectory.sz_bot),
      new THREE.LineBasicMaterial({ color: gridlineColor }),
    );
    scene.add(zone);

    // Full-path trail, sampled once — context for where the ball is headed.
    const N = 40;
    const trailPts = Array.from({ length: N + 1 }, (_, i) => {
      const [x, y, z] = flight.positionAt((i / N) * flight.tauTotal);
      return toThree(x, y, z);
    });
    const trail = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(trailPts),
      new THREE.LineBasicMaterial({ color: familyColorHex, transparent: true, opacity: 0.35 }),
    );
    scene.add(trail);

    const releaseMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.06, 12, 12),
      new THREE.MeshBasicMaterial({ color: familyColorHex }),
    );
    releaseMarker.position.copy(trailPts[0]);
    scene.add(releaseMarker);

    const ball = new THREE.Mesh(
      new THREE.SphereGeometry(BALL_RADIUS_FT, 16, 16),
      new THREE.MeshBasicMaterial({ color: ballColor }),
    );
    scene.add(ball);

    let rafId = 0;
    let startTime: number | null = null;
    let cancelled = false;

    const tick = (now: number) => {
      if (cancelled) return;
      if (startTime === null) startTime = now;
      const elapsedS = (now - startTime) / 1000;
      const tau = Math.min(elapsedS / SLOWMO, flight.tauTotal);
      const [x, y, z] = flight.positionAt(tau);
      ball.position.copy(toThree(x, y, z));

      if (tau >= flight.tauTotal) {
        setPhase("done");
      } else {
        rafId = requestAnimationFrame(tick);
      }
      renderer.render(scene, camera);
    };
    rafId = requestAnimationFrame(tick);

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
      renderer.dispose();
      scene.traverse((obj) => {
        if (obj instanceof THREE.Mesh || obj instanceof THREE.Line) {
          obj.geometry.dispose();
          const mat = obj.material;
          (Array.isArray(mat) ? mat : [mat]).forEach((m) => m.dispose());
        }
      });
      container.removeChild(renderer.domElement);
    };
  }, [trajectory, width, height, replayKey]);

  return (
    <figure style={{ margin: 0 }}>
      <div ref={containerRef} style={{ width, height, borderRadius: 6, overflow: "hidden" }} />
      <figcaption
        style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          fontSize: 12, color: "var(--text-secondary)", marginTop: 6,
        }}
      >
        <span>
          {labelOf(trajectory.pitch_type)}
          {trajectory.release_speed != null && ` · ${trajectory.release_speed.toFixed(1)} mph`}
          {" · batter's-eye view"}
        </span>
        <button
          onClick={() => { setPhase("flight"); setReplayKey((k) => k + 1); }}
          disabled={phase === "flight"}
          style={{
            background: "none", border: "1px solid var(--gridline)", borderRadius: 4,
            padding: "2px 8px", cursor: phase === "flight" ? "default" : "pointer",
            color: "var(--text-primary)", fontSize: 11,
          }}
        >
          replay
        </button>
      </figcaption>
    </figure>
  );
}
