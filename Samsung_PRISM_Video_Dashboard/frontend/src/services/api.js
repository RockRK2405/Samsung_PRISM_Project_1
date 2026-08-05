// Central API client. All backend calls go through here so the base URL
// and error handling live in one place.

const BASE = ""; // same-origin (vite proxies /api to :8000 in dev)

async function jsonFetch(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  dashboardSummary: () => jsonFetch("/api/dashboard/summary"),
  modelStatus: () => jsonFetch("/api/model/status"),

  detect: (file, groundTruth, generator) => {
    const fd = new FormData();
    fd.append("file", file);
    if (groundTruth) fd.append("ground_truth", groundTruth);
    if (generator) fd.append("generator", generator);
    return jsonFetch("/api/detect", { method: "POST", body: fd });
  },

  detectBatch: (files, datasetLabel, generator) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    if (datasetLabel) fd.append("dataset_label", datasetLabel);
    if (generator) fd.append("generator", generator);
    return jsonFetch("/api/detect/batch", { method: "POST", body: fd });
  },

  experiments: (limit) =>
    jsonFetch(`/api/experiments${limit ? `?limit=${limit}` : ""}`),
  experiment: (id) => jsonFetch(`/api/experiments/${id}`),
  deleteExperiment: (id) =>
    jsonFetch(`/api/experiments/${id}`, { method: "DELETE" }),

  evaluation: (threshold) =>
    jsonFetch(`/api/evaluation${threshold != null ? `?threshold=${threshold}` : ""}`),
  thresholdSweep: (values) =>
    jsonFetch(`/api/evaluation/threshold-sweep${values ? `?values=${values}` : ""}`),

  exportAllCsvUrl: () => "/api/experiments/export/all.csv",
  exportJsonUrl: (id) => `/api/experiments/${id}/export.json`,
};
