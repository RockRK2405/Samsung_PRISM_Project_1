import React from "react";

export function StatCard({ title, value, sub }) {
  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div className="stat mt-2">{value ?? "—"}</div>
      {sub && <div className="mt-1 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

export function VerdictBadge({ value }) {
  const v = (value || "").toLowerCase();
  const cls =
    v === "synthetic" ? "badge-synthetic" : v === "real" ? "badge-real" : "badge-uncertain";
  return <span className={cls}>{value || "—"}</span>;
}

export function Section({ title, right, children }) {
  return (
    <div className="card">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200">{title}</h3>
        {right}
      </div>
      {children}
    </div>
  );
}

export function Field({ label, children }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-400">{label}</span>
      {children}
    </label>
  );
}

export function Spinner({ label }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-400">
      <div className="h-4 w-4 animate-spin rounded-full border-2 border-slate-600 border-t-indigo-400" />
      {label || "Working…"}
    </div>
  );
}

export function pct(x) {
  return x == null ? "—" : `${x}%`;
}

export function fmt(x, digits = 3) {
  return x == null ? "—" : Number(x).toFixed(digits);
}
