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
 * at 60.5). The camera starts in a fixed batter's-eye position but is fully
 * user-controlled from there (OrbitControls: drag to orbit, scroll to zoom,
 * right-drag to pan) — it does not track the ball itself.
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
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { PitchTrajectory } from "../lib/api";
import { familyColor, familyOf, labelOf } from "../lib/scales";
import { PLATE_Y, reconstructFlight } from "../lib/trajectory";

const PLATE_HALF_FT = 0.83;
const RUBBER_Y = 60.5;
const BALL_RADIUS_FT = 0.121;
// 1 = real time (release to plate in its true ~0.4-0.5s). The slider goes
// down from there for a slow-motion look at movement/spin — never up, since
// real time is already the fastest a real pitch happens.
const DEFAULT_SPEED = 1;
const MIN_SPEED = 0.1;

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
  const [speed, setSpeed] = useState(DEFAULT_SPEED);
  // A ref, not just the `speed` state, so dragging the slider adjusts the
  // running animation's rate in place — it must NOT be a dependency of the
  // scene-setup effect below, or every tick of the drag would tear down and
  // rebuild the whole WebGL scene (and reset the user's camera orbit).
  const speedRef = useRef(DEFAULT_SPEED);
  speedRef.current = speed;

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

    // User-controlled camera: drag to orbit, scroll/pinch to zoom, right-drag
    // (or two-finger drag) to pan. Starting orientation is the staged
    // batter's-eye framing above; the user is free to move from there.
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, zoneMidZ, PLATE_Y);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 2;
    controls.maxDistance = 80;
    controls.maxPolarAngle = Math.PI * 0.49; // stop just short of going underground
    controls.update();

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
    let lastFrameTime: number | null = null;
    let tauElapsed = 0;
    let reachedEnd = false;
    let cancelled = false;

    // Runs continuously for the life of the component, not just while the
    // ball is in flight — OrbitControls needs a render every frame to feel
    // responsive to drag/zoom/pan, and damping needs `controls.update()`
    // every frame too, well after the ball itself has stopped moving.
    const tick = (now: number) => {
      if (cancelled) return;
      if (lastFrameTime === null) lastFrameTime = now;
      const dt = (now - lastFrameTime) / 1000;
      lastFrameTime = now;

      // Accumulate physics-time rather than deriving tau from wall-clock
      // elapsed directly, so dragging the speed slider mid-flight changes
      // the RATE from here on rather than jumping the ball to a new tau.
      tauElapsed = Math.min(tauElapsed + dt * speedRef.current, flight.tauTotal);
      const [x, y, z] = flight.positionAt(tauElapsed);
      ball.position.copy(toThree(x, y, z));

      if (tauElapsed >= flight.tauTotal && !reachedEnd) {
        reachedEnd = true;
        setPhase("done");
      }

      controls.update();
      renderer.render(scene, camera);
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
      controls.dispose();
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
          display: "flex", flexDirection: "column", gap: 6,
          fontSize: 12, color: "var(--text-secondary)", marginTop: 6,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span>
            {labelOf(trajectory.pitch_type)}
            {trajectory.release_speed != null && ` · ${trajectory.release_speed.toFixed(1)} mph`}
            {" · drag to orbit, scroll to zoom"}
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
        </div>
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 10, color: "var(--text-muted)", whiteSpace: "nowrap" }}>speed</span>
          <input
            type="range"
            min={MIN_SPEED}
            max={DEFAULT_SPEED}
            step={0.05}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <span
            style={{
              fontSize: 10, color: "var(--text-muted)", width: 36,
              textAlign: "right", fontVariantNumeric: "tabular-nums",
            }}
          >
            {(speed * 100).toFixed(0)}%
          </span>
        </label>
      </figcaption>
    </figure>
  );
}
