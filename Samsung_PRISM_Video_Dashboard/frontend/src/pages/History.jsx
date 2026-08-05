import React, { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../services/api.js";
import { Section, VerdictBadge, Spinner } from "../components/common.jsx";
import ResultView from "../components/ResultView.jsx";

export default function History() {
  const [rows, setRows] = useState(null);
  const [open, setOpen] = useState(null);
  const [params] = useSearchParams();

  function refresh() {
    api.experiments().then((d) => setRows(d.experiments)).catch(() => setRows([]));
  }
  useEffect(() => {
    refresh();
    const id = params.get("open");
    if (id) openDetail(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function openDetail(id) {
    api.experiment(id).then(setOpen).catch(() => {});
  }

  async function del(id) {
    await api.deleteExperiment(id);
    setOpen(null);
    refresh();
  }

  if (!rows) return <Spinner label="Loading history…" />;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-slate-100">Experiment History</h2>
        <a className="btn-ghost" href={api.exportAllCsvUrl()}>Export all (CSV)</a>
      </div>

      <Section title={`${rows.length} experiment(s)`}>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr className="border-b border-slate-800">
                {["ID", "Filename", "Date", "Ground Truth", "Prediction", "Confidence", "Generator", "Model", ""].map((h) => (
                  <th key={h} className="th">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id} className="border-b border-slate-900 hover:bg-slate-900/40">
                  <td className="td font-mono text-xs">{r.id}</td>
                  <td className="td">{r.filename}</td>
                  <td className="td text-slate-500">{new Date(r.created_at).toLocaleString()}</td>
                  <td className="td">{r.ground_truth || "—"}</td>
                  <td className="td"><VerdictBadge value={r.prediction} /></td>
                  <td className="td">{(r.confidence * 100).toFixed(1)}%</td>
                  <td className="td">{r.generator || "—"}</td>
                  <td className="td text-xs">{r.model_version}{r.mock ? " (mock)" : ""}</td>
                  <td className="td">
                    <button className="btn-ghost" onClick={() => openDetail(r.id)}>Open</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {open && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-slate-100">
              {open.filename} <span className="font-mono text-xs text-slate-500">({open.id})</span>
            </h3>
            <div className="flex gap-2">
              <a className="btn-ghost" href={api.exportJsonUrl(open.id)}>Export JSON</a>
              <button className="btn-ghost text-rose-400" onClick={() => del(open.id)}>Delete</button>
              <button className="btn-ghost" onClick={() => setOpen(null)}>Close</button>
            </div>
          </div>
          <ResultView
            payload={open.payload}
            metadata={open.metadata}
            evaluation={
              open.ground_truth && open.ground_truth !== "Unknown"
                ? {
                    ground_truth: open.ground_truth,
                    correct: open.ground_truth === open.prediction,
                    result: open.ground_truth === open.prediction ? "CORRECT" : "INCORRECT",
                  }
                : null
            }
          />
        </div>
      )}
    </div>
  );
}
