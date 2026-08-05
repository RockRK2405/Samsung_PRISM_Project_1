import React, { useRef, useState } from "react";
import { api } from "../services/api.js";
import { Section, Field, Spinner } from "../components/common.jsx";
import ResultView from "../components/ResultView.jsx";

const GENERATORS = ["Gemini", "Veo", "Sora", "Runway", "Kling", "Other", "Unknown"];

export default function TestVideo() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [groundTruth, setGroundTruth] = useState("");
  const [generator, setGenerator] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState(null);
  const [drag, setDrag] = useState(false);
  const inputRef = useRef();

  function choose(f) {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setErr(null);
  }

  async function run() {
    if (!file) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await api.detect(file, groundTruth || null, groundTruth === "Synthetic" ? generator : null);
      setResult(res);
    } catch (e) {
      setErr(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-slate-100">Test Video</h2>

      <Section title="Upload">
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); choose(e.dataTransfer.files[0]); }}
          onClick={() => inputRef.current?.click()}
          className={`cursor-pointer rounded-lg border-2 border-dashed p-8 text-center transition ${
            drag ? "border-indigo-500 bg-indigo-500/5" : "border-slate-700 hover:border-slate-600"
          }`}
        >
          <p className="text-sm text-slate-400">
            Drag & drop a video here, or click to select
          </p>
          <p className="mt-1 text-xs text-slate-600">MP4, MOV, AVI, MKV, WebM</p>
          <input
            ref={inputRef}
            type="file"
            accept=".mp4,.mov,.avi,.mkv,.webm"
            className="hidden"
            onChange={(e) => choose(e.target.files[0])}
          />
        </div>

        {preview && (
          <div className="mt-5 grid gap-5 md:grid-cols-2">
            <video src={preview} controls className="w-full rounded-md border border-slate-800" />
            <div className="space-y-4">
              <div className="text-sm text-slate-400">
                <div><span className="text-slate-500">File:</span> {file?.name}</div>
                <div><span className="text-slate-500">Size:</span> {(file?.size / 1e6).toFixed(2)} MB</div>
                <p className="mt-1 text-xs text-slate-600">
                  Duration / resolution / fps / codec are probed server-side on detection.
                </p>
              </div>

              <Field label="Ground Truth (evaluation only — not sent to the model)">
                <select className="input w-full" value={groundTruth} onChange={(e) => setGroundTruth(e.target.value)}>
                  <option value="">Unknown</option>
                  <option value="Real">Real</option>
                  <option value="Synthetic">Synthetic / Deepfake</option>
                </select>
              </Field>

              {groundTruth === "Synthetic" && (
                <Field label="Generator / Source (optional)">
                  <select className="input w-full" value={generator} onChange={(e) => setGenerator(e.target.value)}>
                    <option value="">Select…</option>
                    {GENERATORS.map((g) => <option key={g} value={g}>{g}</option>)}
                  </select>
                </Field>
              )}

              <button className="btn w-full text-base" disabled={busy} onClick={run}>
                {busy ? "Running…" : "RUN DETECTION"}
              </button>
            </div>
          </div>
        )}
      </Section>

      {busy && <Spinner label="Running detection pipeline…" />}
      {err && <p className="text-rose-400">Detection failed: {err}</p>}
      {result && (
        <ResultView
          payload={result.result}
          metadata={result.metadata}
          evaluation={result.evaluation}
        />
      )}
      {result?.metadata && (
        <Section title="Video Metadata">
          <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <Meta k="Duration" v={`${result.metadata.duration_sec}s`} />
            <Meta k="Resolution" v={result.metadata.resolution} />
            <Meta k="FPS" v={result.metadata.fps} />
            <Meta k="Codec" v={result.metadata.codec} />
            <Meta k="Frame count" v={result.metadata.frame_count} />
            <Meta k="Size" v={`${result.metadata.size_mb} MB`} />
          </div>
        </Section>
      )}
    </div>
  );
}

function Meta({ k, v }) {
  return (
    <div>
      <div className="text-xs text-slate-500">{k}</div>
      <div className="text-slate-200">{v}</div>
    </div>
  );
}
