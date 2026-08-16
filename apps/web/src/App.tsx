import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { FilterBar } from "./components/FilterBar";
import { PlayerSearch } from "./components/PlayerSearch";
import { PlayerPage } from "./pages/PlayerPage";
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

export default function App() {
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
          <PlayerSearch />
        </header>

        <div style={{ marginBottom: 16 }}>
          <FilterBar />
        </div>

        <PlayerPage />
      </div>
    </QueryClientProvider>
  );
}
