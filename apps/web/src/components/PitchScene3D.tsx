/**
 * Animated 3D pitch flight for ONE OR MANY pitches in a single scene.
 *
 * The paths are not approximations: each is `reconstructFlight()`
 * (lib/trajectory.ts) evaluated at 60fps, the same exact-physics reconstruction
 * validated against Savant's own plate_x/plate_z on the backend. Everything else
 * in this file — camera placement, ball size, ground plane, slow-motion factor,
 * the commit plane's translucency — is staging, not physics, and is free to be
 * approximate.
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
 *
 * Encoding, when more than one pitch is on screen: hue is the pitch FAMILY and
 * line pattern is the pitch type within it (`pitchDash`), the same split the 2D
 * charts use, so a sweeper reads as a sweeper across the whole app. `group`
 * separates two SOURCES — two pitchers, or the selected pitch of an at-bat from
 * the rest — by marker geometry (sphere vs. octahedron) rather than by a fourth
 * hue, which the validated three-slot palette does not have.
 *
 * Scene rebuilds are keyed on the pitch ids, not on the array's identity: this
 * component is rendered inside panels that rebuild their props every render, and
 * tearing down a WebGL context on every parent render would reset the reader's
 * camera orbit mid-drag. Speed, sync mode, commit distance and highlight all
 * mutate the live scene in place for the same reason.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { familyColor, familyOf, pitchDash } from "../lib/scales";
import { PLATE_Y, reconstructFlight, type Flight, type PhysicsParams } from "../lib/trajectory";
import { COMMIT_DISTANCE_FT, startOffsets, type SyncMode } from "../lib/tunnel";

const PLATE_HALF_FT = 0.83;
const RUBBER_Y = 60.5;
const BALL_RADIUS_FT = 0.121;
// 1 = real time (release to plate in its true ~0.4-0.5s). The slider goes
// down from there for a slow-motion look at movement/spin — never up, since
// real time is already the fastest a real pitch happens.
const DEFAULT_SPEED = 1;
const MIN_SPEED = 0.1;
// Seconds of physics time to hold the finished frame before looping. Long
// enough to read where the pitches ended up, short enough that a reader
// comparing two arsenals is not waiting on the animation.
const LOOP_HOLD_S = 1.2;

export interface ScenePitch {
  /** Stable identity; also the scene-rebuild key. */
  id: string;
  trajectory: PhysicsParams;
  pitchType: string | null;
  /** Which source this pitch belongs to — 0 and 1 get different ball geometry. */
  group?: 0 | 1;
  /** Drawn dimmer, for context pitches behind a highlighted one. */
  muted?: boolean;
}

interface Props {
  pitches: ScenePitch[];
  szTop: number;
  szBot: number;
  /** Batter handedness — places the camera in the right batter's box. */
  stand?: string;
  width?: number;
  height?: number;
  syncMode?: SyncMode;
  /** Feet from the plate for the commit plane; null hides it entirely. */
  commitDistance?: number | null;
  /** Id of the pitch to emphasize; every other line dims. */
  highlightId?: string | null;
  /** Loop the animation instead of stopping at the plate. */
  loop?: boolean;
  /** Extra controls rendered next to the replay button. */
  children?: React.ReactNode;
}

const toThree = (x: number, y: number, z: number) => new THREE.Vector3(-x, z, y);

/**
 * Resolve a design token to a Three.js hex.
 *
 * Accepts either a bare custom property (`--gridline`) or the `var(--x)` form
 * that `scales.ts` hands back, because `getPropertyValue` takes the property
 * NAME and returns empty string for the `var(...)` wrapper — which silently
 * fell through to the grey fallback and painted every trajectory the same
 * colour regardless of pitch family. WebGL has no CSS, so this is the one place
 * tokens have to be resolved by hand; everything downstream is a number.
 */
function cssColor(el: Element, token: string): number {
  const name = token.startsWith("var(") ? token.slice(4, -1).trim() : token;
  const raw = getComputedStyle(el).getPropertyValue(name).trim();
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

/** Ball/marker geometry per source group — the non-colour channel for "whose". */
function ballGeometry(group: 0 | 1, radius: number): THREE.BufferGeometry {
  return group === 1
    ? new THREE.OctahedronGeometry(radius * 1.25)
    : new THREE.SphereGeometry(radius, 16, 16);
}

interface Track {
  id: string;
  flight: Flight;
  ball: THREE.Mesh;
  line: THREE.Line;
  release: THREE.Mesh;
  commitMark: THREE.Mesh;
  material: THREE.LineBasicMaterial | THREE.LineDashedMaterial;
  baseOpacity: number;
}

export function PitchScene3D({
  pitches, szTop, szBot, stand = "R", width = 480, height = 340,
  syncMode = "release", commitDistance = COMMIT_DISTANCE_FT,
  highlightId = null, loop = false, children,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [phase, setPhase] = useState<"flight" | "done">("flight");
  const [speed, setSpeed] = useState(DEFAULT_SPEED);

  // Refs, not just state, so changing any of these adjusts the RUNNING
  // animation in place. Making them dependencies of the scene effect below
  // would tear down and rebuild the whole WebGL scene on every slider tick
  // (and snap the user's camera orbit back to its starting framing).
  const speedRef = useRef(DEFAULT_SPEED);
  speedRef.current = speed;
  const syncRef = useRef<SyncMode>(syncMode);
  syncRef.current = syncMode;
  const loopRef = useRef(loop);
  loopRef.current = loop;
  const pitchesRef = useRef(pitches);
  pitchesRef.current = pitches;
  const tracksRef = useRef<Track[]>([]);
  const commitRef = useRef<THREE.Line | null>(null);
  const replayRef = useRef<() => void>(() => {});

  // Ids, not the array, drive the rebuild — see the file comment.
  const sceneKey = useMemo(
    () => pitches.map((p) => `${p.id}:${p.group ?? 0}`).join("|"),
    [pitches],
  );

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const scenePitches = pitchesRef.current;

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
    const boxX = stand === "L" ? -1.6 : 1.6;
    const zoneMidZ = (szTop + szBot) / 2;
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

    scene.add(new THREE.Line(homePlateGeometry(), new THREE.LineBasicMaterial({ color: textColor })));

    const rubber = new THREE.Mesh(
      new THREE.BoxGeometry(2, 0.05, 0.5),
      new THREE.MeshBasicMaterial({ color: axisColor }),
    );
    rubber.position.set(0, 0.025, RUBBER_Y);
    scene.add(rubber);

    scene.add(new THREE.Line(
      strikeZoneGeometry(szTop, szBot),
      new THREE.LineBasicMaterial({ color: gridlineColor }),
    ));

    // The commit plane: where the hitter is out of time. Its distance is the
    // one staged number in the comparison (see lib/tunnel.ts), so it is drawn
    // as a surface the reader can see the pitches cross rather than left as a
    // figure under the chart.
    const commitOutline = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-2.5, 0.02, 0), new THREE.Vector3(2.5, 0.02, 0),
        new THREE.Vector3(2.5, 7, 0), new THREE.Vector3(-2.5, 7, 0),
        new THREE.Vector3(-2.5, 0.02, 0),
      ]),
      new THREE.LineBasicMaterial({ color: axisColor, transparent: true, opacity: 0.6 }),
    );
    scene.add(commitOutline);
    commitRef.current = commitOutline;

    const tracks: Track[] = scenePitches.map((p) => {
      const flight = reconstructFlight(p.trajectory);
      const hex = cssColor(container, familyColor(familyOf(p.pitchType)));
      const dash = pitchDash(p.pitchType);

      // Sampled once — the full path is context for where the ball is headed.
      const N = 60;
      const pts = Array.from({ length: N + 1 }, (_, i) => {
        const [x, y, z] = flight.positionAt((i / N) * flight.tauTotal);
        return toThree(x, y, z);
      });
      const geom = new THREE.BufferGeometry().setFromPoints(pts);
      const material = dash
        ? new THREE.LineDashedMaterial({
            color: hex, transparent: true, opacity: 0.75,
            dashSize: dash[0], gapSize: dash[1],
          })
        : new THREE.LineBasicMaterial({ color: hex, transparent: true, opacity: 0.75 });
      const line = new THREE.Line(geom, material);
      // Dashes are measured along the line; without this every dashed material
      // renders solid.
      if (dash) line.computeLineDistances();
      scene.add(line);

      const group = p.group ?? 0;
      const release = new THREE.Mesh(
        ballGeometry(group, 0.06),
        new THREE.MeshBasicMaterial({ color: hex }),
      );
      release.position.copy(pts[0]);
      scene.add(release);

      const commitMark = new THREE.Mesh(
        ballGeometry(group, 0.075),
        new THREE.MeshBasicMaterial({ color: hex }),
      );
      scene.add(commitMark);

      const ball = new THREE.Mesh(
        ballGeometry(group, BALL_RADIUS_FT),
        new THREE.MeshBasicMaterial({ color: scenePitches.length > 1 ? hex : ballColor }),
      );
      scene.add(ball);

      return { id: p.id, flight, ball, line, release, commitMark, material, baseOpacity: 0.75 };
    });
    tracksRef.current = tracks;

    let rafId = 0;
    let lastFrameTime: number | null = null;
    let tauElapsed = 0;
    let reachedEnd = false;
    let cancelled = false;

    const offsetsNow = () => startOffsets(tracks.map((t) => t.flight), syncRef.current);
    const totalNow = () => {
      const offs = offsetsNow();
      return Math.max(0, ...tracks.map((t, i) => offs[i] + t.flight.tauTotal));
    };

    replayRef.current = () => {
      tauElapsed = 0;
      reachedEnd = false;
      lastFrameTime = null; // otherwise the next frame's dt spans the paused time
      setPhase("flight");
    };

    // Runs continuously for the life of the component, not just while a ball is
    // in flight — OrbitControls needs a render every frame to feel responsive
    // to drag/zoom/pan, and damping needs `controls.update()` every frame too,
    // well after the balls themselves have stopped moving.
    const tick = (now: number) => {
      if (cancelled) return;
      if (lastFrameTime === null) lastFrameTime = now;
      const dt = (now - lastFrameTime) / 1000;
      lastFrameTime = now;

      const total = totalNow();
      // Accumulate physics-time rather than deriving tau from wall-clock
      // elapsed directly, so dragging the speed slider mid-flight changes the
      // RATE from here on rather than jumping the balls to a new tau.
      tauElapsed = Math.min(tauElapsed + dt * speedRef.current, total + LOOP_HOLD_S);

      const offs = offsetsNow();
      tracks.forEach((t, i) => {
        const tau = Math.max(0, Math.min(tauElapsed - offs[i], t.flight.tauTotal));
        const [x, y, z] = t.flight.positionAt(tau);
        t.ball.position.copy(toThree(x, y, z));
        // A ball that has not been released yet does not exist to look at, and
        // parking it at the release point would read as a pitch already thrown.
        t.ball.visible = tauElapsed >= offs[i];
      });

      if (tauElapsed >= total && !reachedEnd) {
        reachedEnd = true;
        setPhase("done");
      }
      if (loopRef.current && tauElapsed >= total + LOOP_HOLD_S) {
        replayRef.current();
      }

      controls.update();
      renderer.render(scene, camera);
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
      tracksRef.current = [];
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
      commitRef.current = null;
    };
    // `sceneKey` stands in for `pitches` deliberately — see the file comment.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sceneKey, width, height, szTop, szBot, stand]);

  // Commit plane + per-pitch commit markers, positioned in place so dragging
  // the distance slider does not rebuild the scene.
  useEffect(() => {
    const show = commitDistance != null;
    for (const t of tracksRef.current) {
      t.commitMark.visible = show;
      if (show) {
        const [x, y, z] = t.flight.positionAt(t.flight.tauAtY(commitDistance));
        t.commitMark.position.copy(toThree(x, y, z));
      }
    }
    if (commitRef.current) {
      commitRef.current.visible = show;
      if (show) commitRef.current.position.z = commitDistance;
    }
  }, [commitDistance, sceneKey]);

  // Highlight: dim everything that is not the hovered pitch, in place.
  useEffect(() => {
    for (const t of tracksRef.current) {
      const dim = highlightId != null && t.id !== highlightId;
      t.material.opacity = dim ? 0.12 : t.baseOpacity;
      const markOpacity = dim ? 0.15 : 1;
      for (const m of [t.release, t.commitMark, t.ball]) {
        const mat = m.material as THREE.MeshBasicMaterial;
        mat.transparent = true;
        mat.opacity = markOpacity;
      }
    }
  }, [highlightId, sceneKey]);

  // A sync-mode change re-times every ball, so the honest thing is to restart
  // the flight rather than teleport mid-air balls onto a new schedule.
  useEffect(() => {
    replayRef.current();
  }, [syncMode]);

  return (
    <figure style={{ margin: 0 }}>
      <div ref={containerRef} style={{ width, height, borderRadius: 6, overflow: "hidden" }} />
      <figcaption
        style={{
          display: "flex", flexDirection: "column", gap: 6,
          fontSize: 12, color: "var(--text-secondary)", marginTop: 6,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <span style={{ color: "var(--text-muted)" }}>drag to orbit, scroll to zoom</span>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {children}
            <button
              onClick={() => replayRef.current()}
              disabled={phase === "flight" && !loop}
              style={{
                background: "none", border: "1px solid var(--gridline)", borderRadius: 4,
                padding: "2px 8px",
                cursor: phase === "flight" && !loop ? "default" : "pointer",
                color: "var(--text-primary)", fontSize: 11,
              }}
            >
              replay
            </button>
          </div>
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
