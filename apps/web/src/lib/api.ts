/** API client. JSON for small payloads, Arrow IPC for pitch-level data. */

import { tableFromIPC, type Table } from "apache-arrow";

const BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function json<T>(path: string, params?: Record<string, unknown>): Promise<T> {
  const url = new URL(BASE + path);
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v));
  }
  const res = await fetch(url);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error((detail as { detail?: string }).detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

/**
 * Fetch an Arrow IPC payload.
 *
 * The bytes arrive as columnar typed arrays, so there is no per-row parse the
 * way `JSON.parse` imposes — measured at ~6.8x smaller than the equivalent JSON
 * on real pitch data. Charts read columns directly off the table.
 */
async function arrow(path: string, params?: Record<string, unknown>): Promise<Table> {
  const url = new URL(BASE + path);
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, String(v));
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return tableFromIPC(new Uint8Array(await res.arrayBuffer()));
}

// --- types ------------------------------------------------------------------

export interface PlayerSummary {
  mlbam_id: number;
  full_name: string;
  primary_position: string | null;
  bats: string | null;
  throws: string | null;
  mlb_debut_date: string | null;
}

export interface LeaderRow {
  mlbam_id: number;
  full_name: string;
  season: number;
  p_throws: string;
  pitches: number;
  velo_avg: number;
  csw_pct: number;
}

export interface ArsenalRow {
  mlbam_id: number;
  season: number;
  pitch_type: string;
  pitch_name: string | null;
  pitches: number;
  usage_pct: number;
  velo_avg: number;
  velo_max: number;
  spin_avg: number | null;
  extension_avg: number | null;
  ivb_in: number;
  hb_arm_in: number;
  whiff_pct: number | null;
  csw_pct: number;
  zone_pct: number;
  chase_pct: number;
  xwoba: number | null;
  rv_per_100: number | null;
  velo_diff_fb: number | null;
  ivb_diff_fb: number | null;
  hb_diff_fb: number | null;
  pct_velo: number | null;
  pct_whiff: number | null;
  pct_csw: number | null;
  pct_rv: number | null;
}

/**
 * A row of `mart_pitcher_stuff`. `pitch_type === "ALL"` is the usage-weighted
 * rollup for the whole season, not a pitch.
 */
export interface StuffRow {
  mlbam_id: number;
  season: number;
  pitch_type: string;
  stuff_plus: number;
  location_plus: number;
  pitching_plus: number;
  rv_per_100: number | null;
  pitches: number;
  usage_pct: number;
  full_name?: string;
}

export const ALL_PITCHES = "ALL";

export interface ZoneExtent {
  grid_n: number;
  x_min: number; x_max: number;
  z_min: number; z_max: number;
  min_reliable_n: number;
}

export interface ZoneGrid {
  mlbam_id: number;
  season: number;
  role: string;
  metric: string;
  n_pitches: number;
  grid_n: number;
  surface: (number | null)[];
  reliability: number[];
  extent: ZoneExtent;
  layout: string;
}

export interface LeagueShape {
  pitch_type: string;
  n: number;
  velo: number;
  ivb_in: number;
  hb_arm_in: number;
  ivb_sd: number;
  hb_sd: number;
}

export interface SeasonRow {
  season: number;
  pitches: number;
  games: number;
}

export interface GameSummary {
  game_pk: number;
  game_date: string;
  season: number;
  pitches: number;
}

export interface PredictedPitchType {
  pitch_type: string;
  probability: number;
}

export interface PredictedLocation {
  class: number;
  row: number;
  col: number;
  probability: number;
}

export interface PitchTrajectory {
  pitch_type: string | null;
  pitch_name: string | null;
  p_throws: string;
  stand: string;
  release_speed: number | null;
  release_extension: number | null;
  release_pos_x: number;
  release_pos_y: number;
  release_pos_z: number;
  vx0: number;
  vy0: number;
  vz0: number;
  ax: number;
  ay: number;
  az: number;
  plate_x: number;
  plate_z: number;
  sz_top: number;
  sz_bot: number;
}

export interface ReplayPitch {
  at_bat_number: number;
  pitch_number: number;
  balls: number;
  strikes: number;
  actual_pitch_type: string | null;
  actual_plate_x: number | null;
  actual_plate_z_norm: number | null;
  actual_location_class: number | null;
  predicted_pitch_type: PredictedPitchType[];
  predicted_location?: PredictedLocation[];
}

// --- endpoints --------------------------------------------------------------

export const api = {
  health: () => json<{ status: string; tables: Record<string, boolean>; pitches: number }>("/health"),
  seasons: () => json<SeasonRow[]>("/seasons"),
  searchPlayers: (q: string) => json<PlayerSummary[]>("/players/search", { q }),
  leaders: (season?: number, minPitches = 1) =>
    json<LeaderRow[]>("/players", { season, min_pitches: minPitches, limit: 100 }),
  profile: (id: number) =>
    json<{ player: PlayerSummary; seasons: { season: number; pitches_thrown: number; pitches_seen: number }[] }>(
      `/players/${id}/profile`,
    ),
  arsenal: (id: number, season?: number) => json<ArsenalRow[]>(`/players/${id}/arsenal`, { season }),
  stuff: (id: number, season?: number) => json<StuffRow[]>(`/stuff/${id}`, { season }),
  stuffLeaders: (params: Record<string, unknown>) => json<StuffRow[]>("/stuff", params),
  zoneExtent: () => json<ZoneExtent>("/zones/extent"),
  zones: (id: number, role: string, metric: string, season?: number) =>
    json<ZoneGrid>(`/zones/${id}`, { role, metric, season }),
  leagueShapes: (season?: number, hand = "R") =>
    json<LeagueShape[]>("/pitches/league-shapes", { season, hand }),
  pitches: (params: Record<string, unknown>) => arrow("/pitches", params),
  movement: (pitcherId: number, season?: number) =>
    arrow("/pitches/movement", { pitcher_id: pitcherId, season }),
  games: (id: number, season?: number) =>
    json<GameSummary[]>(`/players/${id}/games`, { season, limit: 25 }),
  replay: (gamePk: number, pitcherId: number) =>
    json<ReplayPitch[]>(`/games/${gamePk}/replay`, { pitcher_id: pitcherId }),
  trajectory: (gamePk: number, atBatNumber: number, pitchNumber: number) =>
    json<PitchTrajectory>("/pitches/trajectory", {
      game_pk: gamePk,
      at_bat_number: atBatNumber,
      pitch_number: pitchNumber,
    }),
};

/** Materialize selected Arrow columns as plain JS objects for chart code. */
export function columns<T>(table: Table, names: string[]): T[] {
  const cols = names.map((n) => table.getChild(n));
  const out: T[] = [];
  for (let i = 0; i < table.numRows; i++) {
    const row: Record<string, unknown> = {};
    names.forEach((n, j) => {
      const v = cols[j]?.get(i);
      row[n] = typeof v === "bigint" ? Number(v) : v;
    });
    out.push(row as T);
  }
  return out;
}
