import React, { useState } from "react";
import { api } from "../services/api.js";
import { Section, Field, VerdictBadge, fmt, Spinner } from "../components/common.jsx";
import { TimelineChart } from "../charts/Charts.jsx";

// Side-by-side comparison of two videos (typically 1 real + 1 synthetic)
// for research demonstrations.
export default function CompareVideos() {
  const [a, setA] = useState(null);
  const [b, setB] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  async function run(file, gt, setter) {
    return api.detect(file, gt, null).then((r) => setter(r));
  }

  async function go(fileA, fileB) {
    if (!fileA || !fileB) return;
    setBusy(true);
    setErr(null);
    try {
      await Promise.all([run(fileA, "Real", setA), run(fileB, "Synthetic", setB)]);
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  const [fileA, setFileA] = useState(null);
  const [fileB, setFileB] = useState(null);

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-slate-100">Compare Videos</h2>

      <Section title="Select two videos">
        <div className="grid gap-4 md:grid-cols-2">
          <Field label="Video A (labelled Real)">
            <input type="file" accept=".mp4,.mov,.avi,.mkv,.webm" className="input w-full"
              onChange={(e) => setFileA(e.target.files[0])} />
          </Field>
          <Field label="Video B (labelled Synthetic)">
            <input type="file" accept=".mp4,.mov,.avi,.mkv,.webm" className="input w-full"
              onChange={(e) => setFileB(e.target.files[0])} />
          </Field>
        </div>
        <button className="btn mt-4" disabled={busy || !fileA || !fileB} onClick={() => go(fileA, fileB)}>
          {busy ? "Running…" : "Compare"}
        </button>
      </Section>

      {busy && <Spinner label="Running both…" />}
      {err && <p className="text-rose-400">{err}</p>}

      {a && b && (
        <div className="grid gap-6 md:grid-cols-2">
          {[a, b].map((res, i) => (
            <Section key={i} title={res.metadata.filename}>
              <div className="space-y-3 text-sm">
                <Row k="Prediction" v={<VerdictBadge value={res.result.prediction} />} />
                <Row k="Decision margin" v={`${(res.result.confidence * 100).toFixed(1)}%`} />
                <Row k="Synthetic prob" v={fmt(res.result.synthetic_probability)} />
                <Row k="Real prob" v={fmt(res.result.real_probability)} />
                <Row k="Frames" v={res.result.num_frames_used} />
                <Row k="Processing time" v={`${fmt(res.result.processing_time, 2)}s`} />
                <Row k="Ground truth" v={res.evaluation?.ground_truth} />
                <Row
                  k="Result"
                  v={
                    res.evaluation?.correct == null ? "—" :
                    res.evaluation.correct
                      ? <span className="text-emerald-400">CORRECT</span>
                      : <span className="text-rose-400">INCORRECT</span>
                  }
                />
              </div>
              <div className="mt-4">
                <TimelineChart frames={res.result.frame_predictions} threshold={res.result.threshold} />
              </div>
            </Section>
          ))}
        </div>
      )}
    </div>
  );
}

function Row({ k, v }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-slate-500">{k}</span>
      <span className="text-slate-200">{v ?? "—"}</span>
    </div>
  );
}
