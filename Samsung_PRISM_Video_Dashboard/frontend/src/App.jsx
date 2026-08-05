import React, { useEffect, useState } from "react";
import { Routes, Route, NavLink } from "react-router-dom";
import { api } from "./services/api.js";
import Dashboard from "./pages/Dashboard.jsx";
import TestVideo from "./pages/TestVideo.jsx";
import BatchTesting from "./pages/BatchTesting.jsx";
import CompareVideos from "./pages/CompareVideos.jsx";
import Evaluation from "./pages/Evaluation.jsx";
import History from "./pages/History.jsx";
import ModelInfo from "./pages/ModelInfo.jsx";

const NAV = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/test", label: "Test Video" },
  { to: "/batch", label: "Batch Testing" },
  { to: "/compare", label: "Compare Videos" },
  { to: "/evaluation", label: "Evaluation" },
  { to: "/history", label: "Experiment History" },
  { to: "/model", label: "Model Information" },
];

export default function App() {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    api.modelStatus().then(setStatus).catch(() => setStatus({ mock: true }));
  }, []);

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="w-60 shrink-0 border-r border-slate-800 bg-slate-900/40 p-4">
        <div className="mb-6">
          <h1 className="text-sm font-bold leading-tight text-slate-100">
            Video Synthetic Media Detection
          </h1>
          <p className="text-xs text-slate-500">Research Testing Dashboard</p>
        </div>
        <nav className="space-y-1">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                `block rounded-md px-3 py-2 text-sm transition ${
                  isActive
                    ? "bg-indigo-600/20 text-indigo-300"
                    : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                }`
              }
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      {/* Main */}
      <div className="flex-1">
        {status?.mock && (
          <div className="border-b border-amber-700/40 bg-amber-500/10 px-6 py-2 text-center text-xs font-medium text-amber-400">
            MOCK MODEL — INFERENCE MODEL NOT CONNECTED
            {status.load_error ? ` (${status.load_error})` : ""}
          </div>
        )}
        <main className="mx-auto max-w-7xl p-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/test" element={<TestVideo />} />
            <Route path="/batch" element={<BatchTesting />} />
            <Route path="/compare" element={<CompareVideos />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/history" element={<History />} />
            <Route path="/model" element={<ModelInfo />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
