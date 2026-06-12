import { Routes, Route } from "react-router-dom";
import { AuthProvider, useAuth } from "./lib/auth";
import { RunProvider } from "./lib/runContext";
import { PaletteProvider, CommandPalette } from "./components/CommandPalette";
import { Layout } from "./components/Layout";
import { LoginGate } from "./components/LoginGate";
import { Spinner } from "./components/ui";
import { Dashboard } from "./pages/Dashboard";
import { Backtests } from "./pages/Backtests";
import { SecurityAnalysis } from "./pages/SecurityAnalysis";
import { HmmLab } from "./pages/HmmLab";
import { NeuralNetwork } from "./pages/NeuralNetwork";
import { Portfolio } from "./pages/Portfolio";
import { PaperTrading } from "./pages/PaperTrading";
import { StressTests } from "./pages/StressTests";
import { Experiments } from "./pages/Experiments";
import { Settings } from "./pages/Settings";
import { Terminal } from "./pages/Terminal";

function Shell() {
  const { ready, required, authed } = useAuth();

  if (!ready) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Spinner label="Connecting to engine…" />
      </div>
    );
  }
  if (required && !authed) return <LoginGate />;

  return (
    <RunProvider>
      <PaletteProvider>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/backtests" element={<Backtests />} />
            <Route path="/security" element={<SecurityAnalysis />} />
            <Route path="/hmm" element={<HmmLab />} />
            <Route path="/nn" element={<NeuralNetwork />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/paper" element={<PaperTrading />} />
            <Route path="/stress" element={<StressTests />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/terminal" element={<Terminal />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Dashboard />} />
          </Routes>
        </Layout>
        <CommandPalette />
      </PaletteProvider>
    </RunProvider>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Shell />
    </AuthProvider>
  );
}
