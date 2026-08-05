import React from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer, BarChart, Bar, Legend,
} from "recharts";

const AXIS = { stroke: "#64748b", fontSize: 11 };
const GRID = "#1e293b";

// Per-frame synthetic-probability timeline (X = approx timestamp / frame).
export function TimelineChart({ frames = [], threshold = 0.5 }) {
  const data = frames.map((f) => ({
    x: f.timestamp_approx != null ? f.timestamp_approx : f.frame_index,
    synthetic: f.synthetic_probability,
  }));
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: -12 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="x" tick={AXIS} label={{ value: "≈ time (s)", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 11 }} />
        <YAxis domain={[0, 1]} tick={AXIS} />
        <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
        <ReferenceLine y={threshold} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: `thr ${threshold}`, fill: "#f59e0b", fontSize: 10 }} />
        <Line type="monotone" dataKey="synthetic" stroke="#818cf8" strokeWidth={2} dot={{ r: 2 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// Confidence / score distribution histogram from a list of values.
export function DistributionChart({ real = [], synthetic = [], bins = 10 }) {
  const make = (arr) => {
    const counts = new Array(bins).fill(0);
    arr.forEach((v) => {
      const i = Math.min(bins - 1, Math.max(0, Math.floor(v * bins)));
      counts[i] += 1;
    });
    return counts;
  };
  const r = make(real);
  const s = make(synthetic);
  const data = Array.from({ length: bins }, (_, i) => ({
    bucket: `${(i / bins).toFixed(1)}`,
    Real: r[i],
    Synthetic: s[i],
  }));
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: -12 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="bucket" tick={AXIS} label={{ value: "P(synthetic)", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 11 }} />
        <YAxis tick={AXIS} allowDecimals={false} />
        <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="Real" fill="#34d399" />
        <Bar dataKey="Synthetic" fill="#fb7185" />
      </BarChart>
    </ResponsiveContainer>
  );
}

// Threshold-sweep line chart: accuracy / F1 / FPR vs threshold.
export function SweepChart({ sweep = [] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={sweep} margin={{ top: 8, right: 16, bottom: 8, left: -12 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="threshold" tick={AXIS} />
        <YAxis tick={AXIS} />
        <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155", fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line type="monotone" dataKey="accuracy" stroke="#818cf8" strokeWidth={2} />
        <Line type="monotone" dataKey="f1" stroke="#34d399" strokeWidth={2} />
        <Line type="monotone" dataKey="false_positive_rate" stroke="#fb7185" strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// Simple 2x2 confusion matrix.
export function ConfusionMatrix({ cm }) {
  if (!cm) return null;
  const Cell = ({ label, value, tone }) => (
    <div className={`rounded-md border border-slate-800 p-4 text-center ${tone}`}>
      <div className="text-2xl font-semibold text-slate-100">{value}</div>
      <div className="mt-1 text-xs text-slate-400">{label}</div>
    </div>
  );
  return (
    <div className="grid grid-cols-2 gap-2">
      <Cell label="True Synthetic (TP)" value={cm.tp} tone="bg-emerald-500/5" />
      <Cell label="Missed fake (FN)" value={cm.fn} tone="bg-rose-500/5" />
      <Cell label="False alarm (FP)" value={cm.fp} tone="bg-rose-500/5" />
      <Cell label="True Real (TN)" value={cm.tn} tone="bg-emerald-500/5" />
    </div>
  );
}
