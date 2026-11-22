import React, { useState } from "react";
import { Section, VerdictBadge, fmt, pct } from "./common.jsx";
import { TimelineChart } from "../charts/Charts.jsx";

// Renders a full detection payload: final verdict, probabilities, temporal
// analysis, frame-level grid + timeline, explainability, and an auto summary.
// `payload` is the model_service dict; `metadata` is the video_service dict;
// `evaluation` is the correctness object (optional).
export default function ResultView({ payload, metadata, evaluation }) {
  const [zoom, setZoom] = useState(null);
  if (!payload) return null;

  const isSynth = payload.prediction === "Synthetic";
  const t = payload.temporal || {};

  return (
    <div className="space-y-6">
      {/* Final prediction banner */}
      <div className={`card border-l-4 ${isSynth ? "border-l-rose-500" : "border-l-emerald-500"}`}>
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="card-title">Final Prediction</div>
            <div className="mt-1 flex items-center gap-3">
              <span className={`text-3xl font-bold ${isSynth ? "text-rose-400" : "text-emerald-400"}`}>
                {payload.prediction === "Synthetic" ? "SYNTHETIC / DEEPFAKE" : "REAL"}
              </span>
              {payload.mock && <span className="badge-uncertain">MOCK</span>}
            </div>
            {payload.meta?.verdict_reason && (
              <div className="mt-1 text-sm text-slate-300">{payload.meta.verdict_reason}</div>
            )}
            <div className="mt-1 text-sm text-slate-400">
              Decision margin: {(payload.confidence * 100).toFixed(1)}%
              <span className="ml-2 text-slate-600" title="Confidence here = distance from the decision threshold, not P(correct).">ⓘ</span>
            </div>
          </div>
          {evaluation && evaluation.correct != null && (
            <div className="text-right">
              <div className="card-title">Ground Truth: {evaluation.ground_truth}</div>
              <div className={`mt-1 text-2xl font-bold ${evaluation.correct ? "text-emerald-400" : "text-rose-400"}`}>
                {evaluation.result}
              </div>
            </div>
          )}
        </div>
        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Metric label="Real Probability" value={fmt(payload.real_probability)} />
          <Metric label="Synthetic Probability" value={fmt(payload.synthetic_probability)} />
          <Metric label="Frames Analysed" value={payload.num_frames_used} />
          <Metric label="Processing Time" value={`${fmt(payload.processing_time, 2)}s`} />
        </div>
      </div>

      {/* Per-branch results (unified pipeline) */}
      {payload.meta?.branches && <BranchPanel branches={payload.meta.branches} />}

      {/* Temporal consistency */}
      <Section title="Temporal Consistency Analysis">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
          <Metric label="Avg P(synth)" value={fmt(t.avg_synthetic)} />
          <Metric label="Max P(synth)" value={fmt(t.max_synthetic)} />
          <Metric label="Min P(synth)" value={fmt(t.min_synthetic)} />
          <Metric label="Variance" value={fmt(t.variance, 4)} />
          <Metric label="Fluctuations" value={t.fluctuations} />
          <Metric label="Suspicious" value={`${t.suspicious_frames}/${t.total_frames} (${t.suspicious_pct}%)`} />
        </div>
        <div className="mt-5">
          {payload.meta?.timeline_source && (
            <div className="mb-2 text-xs text-slate-500">Per-frame scores from: {payload.meta.timeline_source}</div>
          )}
          <TimelineChart frames={payload.frame_predictions} threshold={payload.threshold} />
        </div>
      </Section>

      {/* Frame-level grid */}
      <Section title="Frame-Level Analysis">
        <div className="grid grid-cols-4 gap-2 sm:grid-cols-6 md:grid-cols-8">
          {payload.frame_predictions.map((f) => (
            <button
              key={f.frame_index}
              onClick={() => setZoom(f)}
              className={`rounded-md border p-2 text-left transition hover:border-indigo-500 ${
                f.suspicious ? "border-rose-500/40 bg-rose-500/5" : "border-slate-800 bg-slate-900"
              }`}
              title="Click to inspect"
            >
              <div className="text-[10px] text-slate-500">#{f.frame_index}</div>
              <div className={`text-sm font-semibold ${f.suspicious ? "text-rose-400" : "text-slate-300"}`}>
                {fmt(f.synthetic_probability, 2)}
              </div>
              <div className="text-[10px] text-slate-500">
                {f.timestamp_approx != null ? `≈${f.timestamp_approx}s` : ""}
              </div>
            </button>
          ))}
        </div>
      </Section>

      {/* Explainability */}
      <Section title="Explainability">
        {payload.explainability_score != null ? (
          <div className="text-sm text-slate-300">
            Face-localisation explainability score:{" "}
            <span className="font-semibold text-slate-100">{fmt(payload.explainability_score)}</span>
            <p className="mt-2 text-xs text-slate-500">
              GradCAM heatmaps were generated by the model
              {payload.heatmap_dir ? ` at ${payload.heatmap_dir}` : ""}. Face bounding
              boxes and attention/artifact maps are not produced by the connected
              model and are therefore hidden.
            </p>
          </div>
        ) : (
          <p className="text-sm text-slate-500">
            Explainability output not available for the currently connected model.
          </p>
        )}
      </Section>

      {/* Auto summary — built only from real returned fields */}
      <Section title="Automatic Test Summary">
        <pre className="whitespace-pre-wrap rounded-md bg-slate-950 p-4 text-xs text-slate-300">
{`Video Analysis Summary
Prediction:              ${payload.prediction}
Decision margin:         ${(payload.confidence * 100).toFixed(1)}%
Frames Analysed:         ${payload.num_frames_used}
Suspicious Frames:       ${t.suspicious_frames}/${t.total_frames}
Average Synthetic Prob:  ${fmt(t.avg_synthetic)}
Peak Synthetic Prob:     ${fmt(t.max_synthetic)}
Processing Time:         ${fmt(payload.processing_time, 2)}s${
  payload.meta?.branches
    ? "\n" + Object.entries(payload.meta.branches).map(([k, b]) =>
        `${(k + " branch:").padEnd(25)}${b.status === "ran"
          ? `P(synth) ${fmt(b.prob_synthetic)} vs thr ${b.threshold} -> ${b.fired ? "FIRED" : "clear"}`
          : b.status + (b.reason ? ` (${b.reason})` : "")}`).join("\n")
    : ""
}${
  evaluation && evaluation.correct != null
    ? `\nGround Truth:            ${evaluation.ground_truth}\nClassification Result:   ${evaluation.result}`
    : ""
}${payload.mock ? "\n\n[MOCK MODEL — not a real inference result]" : ""}`}
        </pre>
      </Section>

      {/* Frame zoom modal */}
      {zoom && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6"
          onClick={() => setZoom(null)}
        >
          <div className="card max-w-sm" onClick={(e) => e.stopPropagation()}>
            <div className="card-title">Frame #{zoom.frame_index}</div>
            <div className="mt-3 space-y-1 text-sm">
              <Row k="Timestamp (approx)" v={zoom.timestamp_approx != null ? `≈ ${zoom.timestamp_approx}s` : "—"} />
              <Row k="Predicted class" v={<VerdictBadge value={zoom.predicted_class} />} />
              <Row k="Synthetic prob" v={fmt(zoom.synthetic_probability)} />
              <Row k="Real prob" v={fmt(zoom.real_probability)} />
              <Row k="Decision margin" v={`${(zoom.confidence * 100).toFixed(1)}%`} />
            </div>
            <button className="btn-ghost mt-4 w-full" onClick={() => setZoom(null)}>Close</button>
          </div>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div>
      <div className="text-xs text-slate-500">{label}</div>
      <div className="text-lg font-semibold text-slate-100">{value ?? "—"}</div>
    </div>
  );
}

function Row({ k, v }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-slate-500">{k}</span>
      <span className="text-slate-200">{v}</span>
    </div>
  );
}

const BRANCH_LABEL = {
  general: "General AI-Video Branch",
  face: "Face-Swap Branch",
};

function BranchPanel({ branches }) {
  return (
    <Section title="Detection Branches">
      <div className="grid gap-4 md:grid-cols-2">
        {Object.entries(branches).map(([key, b]) => (
          <BranchCard key={key} name={BRANCH_LABEL[key] || key} b={b} />
        ))}
      </div>
      <p className="mt-3 text-xs text-slate-500">
        Every video runs through the general branch. The face branch runs only when faces
        are found. The final verdict is SYNTHETIC if any branch that ran crosses its threshold.
      </p>
    </Section>
  );
}

function BranchCard({ name, b }) {
  if (b.status !== "ran") {
    return (
      <div className="rounded-md border border-slate-800 bg-slate-900 p-4">
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold text-slate-300">{name}</span>
          <span className="badge-uncertain">{b.status === "not_applicable" ? "NOT APPLICABLE" : "DISABLED"}</span>
        </div>
        {b.reason && <div className="mt-2 text-xs text-slate-500">{b.reason}</div>}
      </div>
    );
  }
  const p = Number(b.prob_synthetic);
  return (
    <div className={`rounded-md border p-4 ${b.fired ? "border-rose-500/50 bg-rose-500/5" : "border-emerald-500/30 bg-emerald-500/5"}`}>
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-slate-200">{name}</span>
        <span className={b.fired ? "badge-synthetic" : "badge-real"}>{b.fired ? "SYNTHETIC" : "REAL"}</span>
      </div>
      <div className="mt-3 text-2xl font-bold text-slate-100">{fmt(p)}</div>
      <div className="text-xs text-slate-500">P(synthetic) · threshold {b.threshold}</div>
      <div className="relative mt-2 h-2 rounded bg-slate-800">
        <div className={`h-2 rounded ${b.fired ? "bg-rose-500" : "bg-emerald-500"}`} style={{ width: `${Math.min(p, 1) * 100}%` }} />
        <div className="absolute top-[-3px] h-[14px] w-[2px] bg-slate-300" style={{ left: `${b.threshold * 100}%` }} title="threshold" />
      </div>
      <div className="mt-3 space-y-1 text-xs">
        {b.detects && <Row k="Detects" v={b.detects} />}
        {b.model && <Row k="Model" v={b.model} />}
        {b.checkpoint && <Row k="Checkpoint" v={b.checkpoint} />}
        {b.face_coverage != null && <Row k="Face coverage" v={`${(b.face_coverage * 100).toFixed(0)}%`} />}
      </div>
    </div>
  );
}

