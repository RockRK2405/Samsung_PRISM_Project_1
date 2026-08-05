import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../services/api.js";
import { StatCard, VerdictBadge, Section, Spinner } from "../components/common.jsx";

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api.dashboardSummary().then(setData).catch((e) => setErr(e.message));
  }, []);

  if (err) return <p className="text-rose-400">Failed to load: {err}</p>;
  if (!data) return <Spinner label="Loading dashboard…" />;

  const s = data.summary;
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-slate-100">Dashboard</h2>
        <span className="text-xs text-slate-500">Model: {data.model_version}</span>
      </div>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard title="Total Videos Tested" value={s.total_tested} />
        <StatCard title="Real Videos" value={s.real_videos} />
        <StatCard title="Synthetic / Deepfake" value={s.synthetic_videos} />
        <StatCard title="Avg Decision Margin" value={s.average_confidence != null ? `${(s.average_confidence * 100).toFixed(1)}%` : "—"} />
        <StatCard title="Correct Predictions" value={s.correct_predictions} />
        <StatCard title="Incorrect Predictions" value={s.incorrect_predictions} />
        <StatCard title="Current Model Version" value={data.model_version} />
        <StatCard title="Inference Mode" value={data.mock ? "MOCK" : "REAL"} />
      </div>

      <Section
        title="Recent Test Results"
        right={<Link className="btn-ghost" to="/history">View all</Link>}
      >
        {data.recent.length === 0 ? (
          <p className="text-sm text-slate-500">No tests yet. Head to “Test Video” to run one.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-slate-800">
                  <th className="th">Filename</th>
                  <th className="th">Ground Truth</th>
                  <th className="th">Prediction</th>
                  <th className="th">Synth Prob</th>
                  <th className="th">When</th>
                </tr>
              </thead>
              <tbody>
                {data.recent.map((r) => (
                  <tr key={r.id} className="border-b border-slate-900">
                    <td className="td">
                      <Link className="text-indigo-400 hover:underline" to={`/history?open=${r.id}`}>
                        {r.filename}
                      </Link>
                    </td>
                    <td className="td">{r.ground_truth || "—"}</td>
                    <td className="td"><VerdictBadge value={r.prediction} /></td>
                    <td className="td">{r.synthetic_probability?.toFixed(3)}</td>
                    <td className="td text-slate-500">{new Date(r.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>
    </div>
  );
}
