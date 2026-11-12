"use client";
// Shared admin UI primitives + API helper.

export const LABEL_STYLE = {
  low: "bg-teal-100 text-teal-800 border-teal-200",
  moderate: "bg-amber-100 text-amber-800 border-amber-200",
  high: "bg-red-100 text-red-800 border-red-200",
};

export function makeApi(token) {
  return async function api(path, { method = "GET", body, raw = false } = {}) {
    const res = await fetch(`/api/admin/${path}`, {
      method,
      headers: { "X-Admin-Token": token, ...(body ? { "Content-Type": "application/json" } : {}) },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (raw && res.ok) return res;
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(data.error || `Request failed (${res.status})`);
      err.status = res.status;
      throw err;
    }
    return data;
  };
}

export async function downloadFrom(api, path, filename) {
  const res = await api(path, { raw: true });
  const blob = await res.blob();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

export const readFile = (file) =>
  new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = (e) => resolve(e.target.result);
    r.onerror = reject;
    r.readAsText(file);
  });

export const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
export const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "—" : typeof v === "number" ? Number(v.toFixed(d)) : String(v));

export function Card({ title, subtitle, actions, children, className = "" }) {
  return (
    <section className={`bg-white rounded-2xl border border-slate-200 shadow-sm ${className}`}>
      {(title || actions) && (
        <header className="flex flex-wrap items-start justify-between gap-3 px-5 pt-5">
          <div>
            {title && <h2 className="text-base font-black text-slate-900 tracking-tight">{title}</h2>}
            {subtitle && <p className="text-sm text-slate-500 mt-0.5">{subtitle}</p>}
          </div>
          {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Button({ variant = "primary", className = "", ...props }) {
  const styles = {
    primary: "bg-teal-600 hover:bg-teal-500 text-white shadow-sm",
    dark: "bg-slate-900 hover:bg-slate-800 text-white shadow-sm",
    ghost: "bg-white hover:bg-slate-50 text-slate-700 border border-slate-200",
    danger: "bg-white hover:bg-red-50 text-red-600 border border-red-200",
  };
  return (
    <button
      type="button"
      {...props}
      className={`px-4 py-2 rounded-xl text-sm font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed ${styles[variant]} ${className}`}
    />
  );
}

export function LabelBadge({ label }) {
  if (!label) return <span className="px-2 py-0.5 rounded-md text-[11px] font-bold border bg-slate-50 text-slate-400 border-slate-200">unlabelled</span>;
  return <span className={`px-2 py-0.5 rounded-md text-[11px] font-bold border uppercase tracking-wide ${LABEL_STYLE[label]}`}>{label}</span>;
}

export function Notice({ kind = "info", children }) {
  const s = {
    info: "bg-sky-50 border-sky-200 text-sky-800",
    ok: "bg-teal-50 border-teal-200 text-teal-800",
    warn: "bg-amber-50 border-amber-200 text-amber-900",
    error: "bg-red-50 border-red-200 text-red-700",
  }[kind];
  return <div className={`rounded-xl border px-4 py-3 text-sm font-medium ${s}`}>{children}</div>;
}

export function Field({ label, hint, children }) {
  return (
    <label className="block">
      <span className="block text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-1">{label}</span>
      {children}
      {hint && <span className="block text-[11px] text-slate-400 mt-1">{hint}</span>}
    </label>
  );
}

export const inputCls =
  "w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-sm font-semibold text-slate-900 focus:ring-2 focus:ring-teal-400 focus:bg-white outline-none";

export function Stat({ label, value, sub, tone = "text-slate-900" }) {
  return (
    <div className="bg-slate-50 rounded-xl border border-slate-100 p-4">
      <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{label}</p>
      <p className={`text-2xl font-black mt-1 ${tone}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
    </div>
  );
}

export function Bar({ value, max = 1, color = "bg-teal-500" }) {
  const w = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
      <div className={`h-full ${color} rounded-full`} style={{ width: `${w}%` }} />
    </div>
  );
}

export function FilePick({ label, file, onFile, accept = ".csv,.txt", multiple = false }) {
  return (
    <label className="flex items-center gap-3 border-2 border-dashed border-slate-200 hover:border-teal-400 rounded-xl px-4 py-3 cursor-pointer transition-all bg-slate-50 hover:bg-white">
      <input
        type="file"
        accept={accept}
        multiple={multiple}
        className="sr-only"
        onChange={(e) => {
          const files = Array.from(e.target.files || []);
          if (files.length) onFile(multiple ? files : files[0]);
          e.target.value = "";
        }}
      />
      <span className={`w-9 h-9 rounded-lg flex items-center justify-center text-sm font-black ${file ? "bg-teal-600 text-white" : "bg-white text-teal-600 border border-slate-200"}`}>
        {file ? "✓" : "↑"}
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-bold text-slate-800 truncate">{label}</span>
        <span className="block text-xs text-slate-500 truncate">{file || "Click to choose file"}</span>
      </span>
    </label>
  );
}

export function MetricsTable({ candidates, selected }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-[10px] uppercase tracking-widest text-slate-500 border-b border-slate-200">
            <th className="py-2 pr-3">Algorithm</th>
            <th className="py-2 pr-3">Macro F1</th>
            <th className="py-2 pr-3">Balanced acc.</th>
            <th className="py-2 pr-3">ROC AUC</th>
            <th className="py-2 pr-3">At-risk sens.</th>
            <th className="py-2 pr-3">At-risk spec.</th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((c) => (
            <tr key={c.algorithm} className={`border-b border-slate-100 ${c.algorithm === selected ? "bg-teal-50/60" : ""}`}>
              <td className="py-2 pr-3 font-bold text-slate-800">{c.name}{c.algorithm === selected && <span className="ml-2 text-[10px] font-black text-teal-700 uppercase">selected</span>}</td>
              <td className="py-2 pr-3 font-semibold">{pct(c.cv_metrics.macro_f1)}</td>
              <td className="py-2 pr-3">{pct(c.cv_metrics.balanced_accuracy)}</td>
              <td className="py-2 pr-3">{c.cv_metrics.roc_auc == null ? "—" : c.cv_metrics.roc_auc.toFixed(3)}</td>
              <td className="py-2 pr-3">{pct(c.cv_metrics.at_risk_sensitivity)}</td>
              <td className="py-2 pr-3">{pct(c.cv_metrics.at_risk_specificity)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ConfusionMatrix({ cm }) {
  if (!cm) return null;
  const max = Math.max(1, ...cm.matrix.flat());
  return (
    <div className="inline-block">
      <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">Confusion matrix (rows = true, cols = predicted)</p>
      <table className="text-xs">
        <thead>
          <tr>
            <th></th>
            {cm.labels.map((l) => <th key={l} className="px-2 py-1 font-bold text-slate-500 uppercase">{l}</th>)}
          </tr>
        </thead>
        <tbody>
          {cm.matrix.map((row, i) => (
            <tr key={i}>
              <th className="pr-2 py-1 text-right font-bold text-slate-500 uppercase">{cm.labels[i]}</th>
              {row.map((v, j) => (
                <td key={j} className="w-14 h-10 text-center font-black rounded"
                  style={{ background: i === j ? `rgba(20,184,166,${0.12 + 0.6 * (v / max)})` : `rgba(239,68,68,${v ? 0.08 + 0.5 * (v / max) : 0.03})` }}>
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function TrainingReport({ report }) {
  if (!report) return null;
  const sel = report.candidates.find((c) => c.algorithm === report.selected);
  const maxImp = Math.max(1e-9, ...(report.feature_importance || []).map((f) => f.importance));
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat label="Version" value={<span className="text-base">{report.version}</span>} sub={report.data_source} />
        <Stat label="Training feet" value={report.n_samples} sub={`${report.n_patients} patients · ${report.cv_folds}-fold CV`} />
        <Stat label="Macro F1 (CV)" value={pct(sel?.cv_metrics.macro_f1)} tone="text-teal-700" />
        <Stat label="At-risk sensitivity" value={pct(sel?.cv_metrics.at_risk_sensitivity)} sub={`specificity ${pct(sel?.cv_metrics.at_risk_specificity)}`} />
      </div>
      {report.warnings?.map((w) => <Notice key={w} kind="warn">{w}</Notice>)}
      <div>
        <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">Candidates — {report.cv_strategy}</p>
        <MetricsTable candidates={report.candidates} selected={report.selected} />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ConfusionMatrix cm={sel?.cv_metrics.confusion_matrix} />
        <div>
          <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">Top features (permutation importance)</p>
          <ul className="space-y-1.5">
            {(report.feature_importance || []).slice(0, 12).map((f) => (
              <li key={f.feature} className="grid grid-cols-[1fr_120px] gap-3 items-center">
                <span className="text-xs font-semibold text-slate-700 truncate" title={f.feature}>{f.label || f.feature}</span>
                <Bar value={Math.max(0, f.importance)} max={maxImp} />
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
