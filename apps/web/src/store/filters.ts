/**
 * One filter state, shared by every chart on the page.
 *
 * Cross-filtering is the entire value of an interactive tool: changing season or
 * handedness must move the heatmap, the movement plot, and the arsenal table
 * together. Per-component local state produces a gallery of charts that happen
 * to sit on the same screen.
 */

import { create } from "zustand";

export type Role = "batter" | "pitcher";

interface FilterState {
  playerId: number | null;
  role: Role;
  season: number | null;
  metric: string;
  pitchType: string | null;
  vsHand: "L" | "R" | null;

  setPlayer: (id: number, role: Role) => void;
  setRole: (role: Role) => void;
  setSeason: (season: number | null) => void;
  setMetric: (metric: string) => void;
  setPitchType: (pt: string | null) => void;
  setVsHand: (h: "L" | "R" | null) => void;
  reset: () => void;
}

const initial = {
  playerId: null,
  role: "pitcher" as Role,
  season: null as number | null,
  metric: "whiff",
  pitchType: null as string | null,
  vsHand: null as "L" | "R" | null,
};

export const useFilters = create<FilterState>((set) => ({
  ...initial,
  // Selecting a player clears pitch-type, which is player-specific: keeping a
  // previous pitcher's sweeper selected would silently empty every chart.
  setPlayer: (playerId, role) => set({ playerId, role, pitchType: null }),
  setRole: (role) => set({ role, pitchType: null }),
  setSeason: (season) => set({ season }),
  setMetric: (metric) => set({ metric }),
  setPitchType: (pitchType) => set({ pitchType }),
  setVsHand: (vsHand) => set({ vsHand }),
  reset: () => set(initial),
}));
