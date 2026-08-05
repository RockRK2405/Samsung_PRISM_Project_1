import React, { useEffect, useState } from "react";
import { api } from "../services/api.js";
import { Section, Spinner } from "../components/common.jsx";

export default function ModelInfo() {
  const [s, setS] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api.modelStatus().then(setS).catch((e) => setErr(e.message));
  }, []);

  if (err) return <p className="text-rose-400">{err}</p>;
  if (!s) return <Spinner label="Loading model info…" />;

  const rows = [
    ["Model Name", s.name],
    ["Architecture", s.architecture],
    ["Checkpoint", s.checkpoint],
    ["Model Version", s.version],
    ["Device", s.device],
    ["Input Size", `${s.input_size} × ${s.input_size}`],
    ["Frame Sampling Strategy", s.frame_sampling_strategy],
    ["Frames Sampled", s.frames_sampled],
    ["Decision Threshold", s.decision_threshold],
  ];

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-slate-100">Model Information</h2>

      <div className={`card border-l-4 ${s.mock ? "border-l-amber-500" : "border-l-emerald-500"}`}>
        <div className="card-title">Inference Status</div>
        <div className={`mt-1 text-lg font-semibold ${s.mock ? "text-amber-400" : "text-emerald-400"}`}>
          {s.mock ? "MOCK MODEL — inference model not connected" : "REAL model connected"}
        </div>
        {s.load_error && <div className="mt-1 text-xs text-slate-500">Reason: {s.load_error}</div>}
      </div>

      <Section title="Configuration (from config/config.yaml)">
        <table className="min-w-full">
          <tbody>
            {rows.map(([k, v]) => (
              <tr key={k} className="border-b border-slate-900">
                <td className="td w-56 text-slate-500">{k}</td>
                <td className="td font-mono text-xs">{String(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-3 text-xs text-slate-500">
          These values come from the config file — edit it there to change the
          checkpoint, threshold, or sampling, then restart the backend.
        </p>
      </Section>
    </div>
  );
}
