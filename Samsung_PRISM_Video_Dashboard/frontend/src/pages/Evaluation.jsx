import React, { useEffect, useState } from "react";
import { api } from "../services/api.js";
import { Section, StatCard, Spinner, pct } from "../components/common.jsx";
import { ConfusionMatrix, DistributionChart, SweepChart } from "../charts/Charts.jsx";

export default function Evaluation() {
  const [threshold, setThreshold] = useState(0.5872);
  const [evalData, setEvalData] = useState(null);
  const [sweep, setSweep] = useState(null);
  const [err, setErr] = useState(null);

  function load(t) {
    api.evaluation(t).then(setEvalData).catch((e) => setErr(e.message));
  }

  useEffect(() => {
    load(null); // stored-threshold view first
    api.thresholdSweep().then((d) => setSweep(d.sweep)).catch(() => {});
  }, []);

  if (err) return <p className="text-rose-400">{err}</p>;
  if (!evalData) return <Spinner label="Computing metrics…" />;

  const m = evalData.metrics;
  const targetsMet = {
    f1: m.f1 != null && m.f1 >= 92,
    fpr: m.false_positive_rate != null && m.false_positive_rate < 5,
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-semibold text-slate-100">Evaluation</h2>
      {m.labeled_count === 0 && (
        <p className="text-sm text-amber-400">
          No labeled experiments yet. Run tests with a Ground Truth to populate metrics.
        </p>
      )}

      <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
        <StatCard title="Labeled" value={m.labeled_count} />
        <StatCard title="Accuracy" value={pct(m.accuracy)} />
        <StatCard title="Precision" value={pct(m.precision)} />
        <StatCard title="Recall" value={pct(m.recall)} />
        <StatCard title="F1 Score" value={pct(m.f1)} sub={targetsMet.f1 ? "≥ 92% ✓" : "target ≥ 92%"} />
        <StatCard title="False Positive Rate" value={pct(m.false_positive_rate)} sub={targetsMet.fpr ? "< 5% ✓" : "target < 5%"} />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Section title="Confusion Matrix">
          <ConfusionMatrix cm={m.confusion_matrix} />
        </Section>
        <Section title="Real vs Synthetic Score Distribution">
          <DistributionChart
            real={evalData.score_distributions.real}
            synthetic={evalData.score_distributions.synthetic}
          />
        </Section>
      </div>

      <Section
        title="Threshold Testing"
        right={
          <div className="flex items-center gap-3">
            <input
              type="range" min="0.3" max="0.8" step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              onMouseUp={() => load(threshold)}
              onTouchEnd={() => load(threshold)}
            />
            <span className="w-14 text-sm text-slate-200">{threshold.toFixed(2)}</span>
            <button className="btn-ghost" onClick={() => load(threshold)}>Apply</button>
          </div>
        }
      >
        <p className="mb-4 text-xs text-slate-500">
          Metrics recompute from stored probabilities — no model reruns. Drag the
          slider, release to apply. Metric cards above update to the applied threshold.
        </p>
        {sweep ? <SweepChart sweep={sweep} /> : <Spinner label="Loading sweep…" />}
      </Section>

      {evalData.per_generator.length > 0 && (
        <Section title="Per-Generator Detection Rate">
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-slate-800">
                  <th className="th">Generator</th>
                  <th className="th">Videos Tested</th>
                  <th className="th">Correctly Detected</th>
                  <th className="th">Detection Rate</th>
                </tr>
              </thead>
              <tbody>
                {evalData.per_generator.map((g) => (
                  <tr key={g.generator} className="border-b border-slate-900">
                    <td className="td">{g.generator}</td>
                    <td className="td">{g.videos_tested}</td>
                    <td className="td">{g.correctly_detected}</td>
                    <td className="td">{pct(g.detection_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}
    </div>
  );
}
