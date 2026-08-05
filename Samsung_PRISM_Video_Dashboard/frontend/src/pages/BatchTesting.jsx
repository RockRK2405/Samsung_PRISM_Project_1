import React, { useMemo, useState } from "react";
import { api } from "../services/api.js";
import { Section, Field, VerdictBadge, fmt } from "../components/common.jsx";

const GENERATORS = ["Gemini", "Veo", "Sora", "Runway", "Kling", "Other", "Unknown"];

export default function BatchTesting() {
  const [files, setFiles] = useState([]);
  const [datasetLabel, setDatasetLabel] = useState("Mixed Dataset");
  const [generator, setGenerator] = useState("");
  const [busy, setBusy] = useState(false);
  const [batch, setBatch] = useState(null);
  const [err, setErr] = useState(null);
  const [filter, setFilter] = useState("all");
  const [sortKey, setSortKey] = useState("filename");

  async function run() {
    if (!files.length) return;
    setBusy(true);
    setErr(null);
    setBatch(null);
    try {
      const res = await api.detectBatch(files, datasetLabel, datasetLabel === "Synthetic Dataset" ? generator : null);
      setBatch(res);
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  const rows = useMemo(() => {
    if (!batch) return [];
    let r = batch.results.map((x) => ({
      filename: x.metadata.filename,
      ground_truth: x.evaluation?.ground_truth,
      prediction: x.result.prediction,
      confidence: x.result.confidence,
      real: x.result.real_probability,
      synthetic: x.result.synthetic_probability,
      time: x.result.processing_time,
      correct: x.evaluation?.correct,
      generator: x.result.meta?.generator,
    }));
    if (filter === "correct") r = r.filter((x) => x.correct === true);
    if (filter === "incorrect") r = r.filter((x) => x.correct === false);
    if (filter === "real") r = r.filter((x) => x.prediction === "Real");
    if (filter === "synthetic") r = r.filter((x) => x.prediction === "Synthetic");
    r.sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (typeof av === "number") return bv - av;
      return String(av).localeCompare(String(bv));
    });
    return r;
  }, [batch, filter, sortKey]);

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-slate-100">Batch Testing</h2>

      <Section title="Upload multiple videos">
        <div className="grid gap-4 md:grid-cols-3">
          <Field label="Videos">
            <input
              type="file"
              multiple
              accept=".mp4,.mov,.avi,.mkv,.webm"
              className="input w-full"
              onChange={(e) => setFiles(Array.from(e.target.files))}
            />
          </Field>
          <Field label="Dataset label">
            <select className="input w-full" value={datasetLabel} onChange={(e) => setDatasetLabel(e.target.value)}>
              <option>Real Dataset</option>
              <option>Synthetic Dataset</option>
              <option>Mixed Dataset</option>
            </select>
          </Field>
          {datasetLabel === "Synthetic Dataset" && (
            <Field label="Generator (optional)">
              <select className="input w-full" value={generator} onChange={(e) => setGenerator(e.target.value)}>
                <option value="">Select…</option>
                {GENERATORS.map((g) => <option key={g} value={g}>{g}</option>)}
              </select>
            </Field>
          )}
        </div>
        <div className="mt-4 flex items-center gap-4">
          <button className="btn" disabled={busy || !files.length} onClick={run}>
            {busy ? "Processing…" : `Run batch (${files.length})`}
          </button>
          {busy && <span className="text-sm text-slate-400">Processing sequentially…</span>}
        </div>
      </Section>

      {err && <p className="text-rose-400">Batch failed: {err}</p>}

      {batch && (
        <Section
          title={`Results — ${batch.succeeded}/${batch.total} succeeded${batch.failed ? `, ${batch.failed} failed` : ""}`}
          right={
            <div className="flex gap-2">
              <select className="input" value={filter} onChange={(e) => setFilter(e.target.value)}>
                <option value="all">All</option>
                <option value="correct">Correct only</option>
                <option value="incorrect">Incorrect only</option>
                <option value="real">Predicted Real</option>
                <option value="synthetic">Predicted Synthetic</option>
              </select>
              <select className="input" value={sortKey} onChange={(e) => setSortKey(e.target.value)}>
                <option value="filename">Sort: Filename</option>
                <option value="confidence">Sort: Confidence</option>
                <option value="synthetic">Sort: Synthetic score</option>
                <option value="time">Sort: Processing time</option>
              </select>
              <a className="btn-ghost" href={api.exportAllCsvUrl()}>Export CSV</a>
            </div>
          }
        >
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-slate-800">
                  {["Video", "Ground Truth", "Prediction", "Confidence", "Real", "Synthetic", "Time", "Status"].map((h) => (
                    <th key={h} className="th">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i} className="border-b border-slate-900">
                    <td className="td">{r.filename}</td>
                    <td className="td">{r.ground_truth || "—"}</td>
                    <td className="td"><VerdictBadge value={r.prediction} /></td>
                    <td className="td">{(r.confidence * 100).toFixed(1)}%</td>
                    <td className="td">{fmt(r.real)}</td>
                    <td className="td">{fmt(r.synthetic)}</td>
                    <td className="td">{fmt(r.time, 2)}s</td>
                    <td className="td">
                      {r.correct == null ? "—" : r.correct
                        ? <span className="text-emerald-400">CORRECT</span>
                        : <span className="text-rose-400">INCORRECT</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {batch.errors?.length > 0 && (
            <div className="mt-4 text-xs text-rose-400">
              {batch.errors.map((e, i) => <div key={i}>{e.filename}: {e.error}</div>)}
            </div>
          )}
        </Section>
      )}
    </div>
  );
}
