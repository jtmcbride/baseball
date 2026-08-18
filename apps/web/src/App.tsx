import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { FilterBar } from "./components/FilterBar";
import { PlayerSearch } from "./components/PlayerSearch";
import { ArsenalMapPage } from "./pages/ArsenalMapPage";
import { PlayerPage } from "./pages/PlayerPage";
import { UmpiresPage } from "./pages/UmpiresPage";
import "./lib/theme.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Historical seasons are immutable, so cached data never goes stale in a
      // way the user would notice.
      staleTime: 5 * 60 * 1000,
      refetchOnWindowFocus: false,
    },
  },
});

type View = "players" | "umpires" | "arsenals";

export default function App() {
  const [view, setView] = useState<View>("players");

  return (
    <QueryClientProvider client={queryClient}>
      <div style={{ maxWidth: 1280, margin: "0 auto", padding: 20 }}>
        <header
          style={{
            display: "flex", gap: 16, alignItems: "center",
            justifyContent: "space-between", marginBottom: 16, flexWrap: "wrap",
          }}
        >
          <h1 style={{ margin: 0, fontSize: 18, letterSpacing: "-0.01em" }}>
            Baseball<span style={{ color: "var(--text-muted)" }}>/analytics</span>
          </h1>
          <nav style={{ display: "flex", gap: 4 }}>
            <ViewTab active={view === "players"} onClick={() => setView("players")}>
              Players
            </ViewTab>
            <ViewTab active={view === "umpires"} onClick={() => setView("umpires")}>
              Umpires
            </ViewTab>
            <ViewTab active={view === "arsenals"} onClick={() => setView("arsenals")}>
              Arsenal map
            </ViewTab>
          </nav>
          {view === "players" && <PlayerSearch />}
        </header>

        <div style={{ marginBottom: 16 }}>
          <FilterBar />
        </div>

        {view === "players" ? <PlayerPage /> : view === "umpires" ? <UmpiresPage /> : <ArsenalMapPage />}
      </div>
    </QueryClientProvider>
  );
}

function ViewTab({
  active, onClick, children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "6px 12px", fontSize: 13, borderRadius: "var(--radius)",
        border: "1px solid var(--border)", cursor: "pointer",
        background: active ? "var(--gridline)" : "var(--surface-1)",
        color: active ? "var(--text-primary)" : "var(--text-secondary)",
      }}
    >
      {children}
    </button>
  );
}
