"use client";
// app/admin/page.js — data & model control room for the SmartSole ML backend.
import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Bar, Button, Card, Field, FilePick, LabelBadge, Notice, Stat, TrainingReport,
  downloadFrom, fmt, inputCls, makeApi, pct, readFile,
} from "./ui";

const TABS = [
  ["overview", "Overview"],
  ["add", "Add data"],
  ["dataset", "Dataset"],
  ["train", "Train model"],
  ["models", "Models"],
  ["predictions", "Prediction log"],
  ["guide", "Data guide"],
];
const TOKEN_KEY = "smartsole_admin_token";

// =============================================================================================
export default function AdminPage() {
  const [token, setToken] = useState("");
  const [tokenInput, setTokenInput] = useState("");
  const [authError, setAuthError] = useState("");
  const [tab, setTab] = useState("overview");
  const [schema, setSchema] = useState(null);
  const [overview, setOverview] = useState(null);

  const api = useMemo(() => makeApi(token), [token]);

  const refresh = useCallback(async () => {
    if (!token) return;
    try {
      setOverview(await api("overview"));
    } catch (e) {
      if (e.status === 401 || e.status === 503) { setToken(""); setAuthError(e.message); }
    }
  }, [api, token]);

  useEffect(() => {
    try {
      const saved = sessionStorage.getItem(TOKEN_KEY);
      // eslint-disable-next-line react-hooks/set-state-in-effect -- read external storage once on mount
      if (saved) setToken(saved);
    } catch { /* storage unavailable */ }
  }, []);

  useEffect(() => {
    if (!token) return;
    (async () => {
      try {
        const [s, o] = await Promise.all([api("schema"), api("overview")]);
        setSchema(s);
        setOverview(o);
        setAuthError("");
        try { sessionStorage.setItem(TOKEN_KEY, token); } catch { /* ignore */ }
      } catch (e) {
        setAuthError(e.message);
        setToken("");
        try { sessionStorage.removeItem(TOKEN_KEY); } catch { /* ignore */ }
      }
    })();
  }, [api, token]);

  if (!token || !schema) {
    return (
      <main className="min-h-screen bg-slate-100 flex items-center justify-center p-4 font-sans">
        <form
          onSubmit={(e) => { e.preventDefault(); setAuthError(""); setToken(tokenInput.trim()); }}
          className="bg-white rounded-3xl shadow-xl border border-slate-200 p-8 w-full max-w-sm space-y-5"
        >
          <div>
            <p className="text-[10px] font-black uppercase tracking-widest text-teal-600">SmartSole</p>
            <h1 className="text-2xl font-black text-slate-900">ML Admin</h1>
            <p className="text-sm text-slate-500 mt-1">Enter the admin token configured on the ML service (<code>ADMIN_TOKEN</code>).</p>
          </div>
          <input type="password" autoFocus value={tokenInput} onChange={(e) => setTokenInput(e.target.value)} className={inputCls} placeholder="Admin token" />
          {authError && <Notice kind="error">{authError}</Notice>}
          <Button type="submit" className="w-full py-3" disabled={!tokenInput.trim()}>{token && !schema ? "Checking…" : "Unlock"}</Button>
          <Link href="/" className="block text-center text-xs font-bold text-slate-400 hover:text-teal-600">← Back to dashboard</Link>
        </form>
      </main>
    );
  }

  const logout = () => {
    try { sessionStorage.removeItem(TOKEN_KEY); } catch { /* ignore */ }
    setToken(""); setSchema(null); setTokenInput("");
  };

  return (
    <main className="min-h-screen bg-slate-100 font-sans text-slate-800">
      <header className="bg-slate-900 text-white">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <span className="w-9 h-9 bg-teal-500 rounded-xl flex items-center justify-center font-black text-sm">ML</span>
            <div>
              <p className="font-black tracking-tight">SmartSole · ML Admin</p>
              <p className="text-xs text-slate-400">
                Active model: <span className="text-teal-300 font-bold">{overview?.active_model?.version || "none"}</span>
                {overview?.active_model && <> · {overview.active_model.algorithm.replaceAll("_", " ")} · {overview.active_model.data_source}</>}
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <Link href="/" className="px-3 py-2 rounded-xl text-sm font-bold bg-white/10 hover:bg-white/20">Dashboard</Link>
            <button onClick={logout} className="px-3 py-2 rounded-xl text-sm font-bold bg-white/10 hover:bg-white/20">Lock</button>
          </div>
        </div>
        <nav className="max-w-7xl mx-auto px-4 md:px-6 flex gap-1 overflow-x-auto">
          {TABS.map(([k, l]) => (
            <button key={k} onClick={() => setTab(k)}
              className={`px-4 py-2.5 text-sm font-bold rounded-t-xl whitespace-nowrap transition-all ${tab === k ? "bg-slate-100 text-slate-900" : "text-slate-300 hover:text-white"}`}>
              {l}
            </button>
          ))}
        </nav>
      </header>

      <div className="max-w-7xl mx-auto px-4 md:px-6 py-6 space-y-6">
        {overview?.storage?.ephemeral && (
          <Notice kind="warn">
            <b>Temporary storage:</b> this ML service runs without a persistent disk (e.g. Render free plan).
            Uploaded sessions, the prediction log and models you train are <b>lost whenever the service restarts or redeploys</b>.
            Use <b>Dataset → Export ML dataset (CSV)</b> to keep a copy of your data. The bundled starter model always comes back automatically.
          </Notice>
        )}
        {tab === "overview" && <Overview overview={overview} goTo={setTab} />}
        {tab === "add" && <AddData api={api} schema={schema} onSaved={refresh} />}
        {tab === "dataset" && <Dataset api={api} schema={schema} onChange={refresh} />}
        {tab === "train" && <Train api={api} schema={schema} overview={overview} onTrained={refresh} />}
        {tab === "models" && <Models api={api} overview={overview} onChange={refresh} />}
        {tab === "predictions" && <Predictions api={api} schema={schema} onChange={refresh} />}
        {tab === "guide" && <Guide schema={schema} api={api} />}
      </div>
    </main>
  );
}

// =============================================================================================
function Overview({ overview, goTo }) {
  if (!overview) return null;
  const r = overview.readiness;
  const m = overview.active_model;
  const maxC = Math.max(1, ...Object.values(r.class_counts));
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <Card title="Dataset" subtitle="Labelled clinical recordings available for training" className="lg:col-span-2">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          <Stat label="Sessions" value={r.sessions} sub={`${r.included_sessions} included`} />
          <Stat label="Labelled feet" value={r.labelled_feet} sub={`${r.unlabelled_feet} unlabelled`} />
          <Stat label="Patients" value={r.patients} />
          <Stat label="Quality warnings" value={r.sessions_with_quality_warnings} tone={r.sessions_with_quality_warnings ? "text-amber-600" : "text-slate-900"} sub="sessions" />
        </div>
        <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">Class balance</p>
        <div className="space-y-2 mb-6">
          {Object.entries(r.class_counts).map(([k, v]) => (
            <div key={k} className="grid grid-cols-[90px_1fr_40px] gap-3 items-center">
              <LabelBadge label={k} />
              <Bar value={v} max={maxC} color={k === "low" ? "bg-teal-500" : k === "moderate" ? "bg-amber-500" : "bg-red-500"} />
              <span className="text-sm font-black text-right">{v}</span>
            </div>
          ))}
        </div>
        <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">Training readiness</p>
        <ul className="space-y-1.5">
          {r.checks.map((c) => (
            <li key={c.id} className="flex items-start gap-2 text-sm">
              <span className={`mt-0.5 w-5 h-5 rounded-full flex items-center justify-center text-[11px] font-black shrink-0 ${c.ok ? "bg-teal-100 text-teal-700" : c.optional ? "bg-slate-100 text-slate-400" : "bg-red-100 text-red-600"}`}>{c.ok ? "✓" : c.optional ? "–" : "✕"}</span>
              <span className={c.ok ? "text-slate-700" : "text-slate-500"}>{c.text}{c.optional && <em className="text-slate-400"> (recommended)</em>}</span>
            </li>
          ))}
        </ul>
        <div className="flex flex-wrap gap-2 mt-6">
          <Button onClick={() => goTo("add")}>+ Add labelled data</Button>
          <Button variant="dark" disabled={!r.trainable} onClick={() => goTo("train")}>Train on real data</Button>
        </div>
      </Card>

      <Card title="Active model" subtitle="Used by the patient dashboard for every prediction">
        {m ? (
          <div className="space-y-4">
            {m.data_source === "synthetic-bootstrap" && (
              <Notice kind="warn">This is the <b>starter model</b>, trained on simulated recordings labelled by clinical rules. Add real labelled data and retrain to replace it.</Notice>
            )}
            <div className="text-sm space-y-1">
              <p><span className="text-slate-500">Version</span> <b className="font-mono">{m.version}</b></p>
              <p><span className="text-slate-500">Algorithm</span> <b>{m.algorithm.replaceAll("_", " ")}</b></p>
              <p><span className="text-slate-500">Trained on</span> <b>{m.n_samples}</b> feet ({m.data_source})</p>
              <p><span className="text-slate-500">Created</span> <b>{new Date(m.created_at).toLocaleString()}</b></p>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Stat label="Macro F1" value={pct(m.metrics.macro_f1)} tone="text-teal-700" />
              <Stat label="ROC AUC" value={m.metrics.roc_auc == null ? "—" : m.metrics.roc_auc.toFixed(3)} />
              <Stat label="At-risk sens." value={pct(m.metrics.at_risk_sensitivity)} />
              <Stat label="At-risk spec." value={pct(m.metrics.at_risk_specificity)} />
            </div>
            <Button variant="ghost" onClick={() => goTo("models")} className="w-full">All models →</Button>
          </div>
        ) : <Notice kind="error">No active model.</Notice>}
      </Card>

      <Card title="How the pipeline works" className="lg:col-span-3">
        <ol className="grid grid-cols-1 md:grid-cols-5 gap-3 text-sm">
          {[
            ["1 · Collect", "Record both insoles in the same session and examine the patient's feet."],
            ["2 · Upload", "Add the two raw logs + profile + a risk label per foot (single or bulk)."],
            ["3 · Featurise", "Each foot becomes ~80 features: zone pressures, peaks, gait, temperature, ΔT, humidity, asymmetry, clinical history."],
            ["4 · Train", "Gradient boosting, random forest and logistic regression are cross-validated by patient; the best is registered."],
            ["5 · Deploy", "Activate a version — the dashboard uses it instantly. Label logged predictions later to keep learning."],
          ].map(([h, t]) => (
            <li key={h} className="bg-slate-50 rounded-xl p-4 border border-slate-100">
              <p className="font-black text-slate-900 mb-1">{h}</p>
              <p className="text-slate-600">{t}</p>
            </li>
          ))}
        </ol>
      </Card>
    </div>
  );
}

// =============================================================================================
function ClinicalInputs({ schema, meta, setMeta }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {schema.clinical_fields.map((f) => (
        <Field key={f.key} label={`${f.label}${f.unit ? ` (${f.unit})` : ""}${f.required ? " *" : ""}`}>
          {f.kind === "number" ? (
            <input type="number" step="any" className={inputCls} value={meta[f.key] ?? ""} onChange={(e) => setMeta({ ...meta, [f.key]: e.target.value })} />
          ) : f.kind === "sex" ? (
            <select className={inputCls} value={meta[f.key] ?? ""} onChange={(e) => setMeta({ ...meta, [f.key]: e.target.value })}>
              <option value="">—</option><option>Male</option><option>Female</option>
            </select>
          ) : (
            <select className={inputCls} value={meta[f.key] === true ? "yes" : meta[f.key] === false ? "no" : ""}
              onChange={(e) => setMeta({ ...meta, [f.key]: e.target.value === "" ? null : e.target.value === "yes" })}>
              <option value="">unknown</option><option value="yes">yes</option><option value="no">no</option>
            </select>
          )}
        </Field>
      ))}
    </div>
  );
}

function LabelSelect({ value, onChange, schema }) {
  return (
    <select className={inputCls} value={value ?? ""} onChange={(e) => onChange(e.target.value || null)}>
      <option value="">unlabelled</option>
      {schema.labels.map((l) => <option key={l} value={l}>{l}</option>)}
    </select>
  );
}

function FeaturePreview({ preview }) {
  if (!preview) return null;
  const { features, feature_labels: labels, quality } = preview;
  const keys = Object.keys(features.left);
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {["left", "right"].map((s) => (
          <div key={s} className="bg-slate-50 rounded-xl p-4 border border-slate-100 text-sm">
            <p className="font-black capitalize mb-1">{s} log</p>
            <p className="text-slate-600">{quality[s].rows} rows · {quality[s].duration_s}s · {quality[s].sampling_hz} Hz</p>
            {quality[s].warnings.map((w) => <p key={w} className="text-amber-700 text-xs mt-1">⚠ {w}</p>)}
          </div>
        ))}
      </div>
      <p className="text-sm text-slate-600">Left/right ΔT: <b>{fmt(features.abs_delta_t)} °C</b></p>
      <div className="max-h-80 overflow-auto border border-slate-100 rounded-xl">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-white">
            <tr className="text-left text-[10px] uppercase tracking-widest text-slate-500">
              <th className="p-2">Feature</th><th className="p-2 text-right">Left</th><th className="p-2 text-right">Right</th>
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => (
              <tr key={k} className="border-t border-slate-100">
                <td className="p-2"><span className="font-semibold text-slate-700">{labels?.[k] || k}</span> <span className="text-slate-400 font-mono">{k}</span></td>
                <td className="p-2 text-right font-mono">{fmt(features.left[k], 3)}</td>
                <td className="p-2 text-right font-mono">{fmt(features.right[k], 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AddData({ api, schema, onSaved }) {
  const empty = { patient_id: "", recorded_on: new Date().toISOString().slice(0, 10), notes: "", left_label: null, right_label: null };
  const [form, setForm] = useState(empty);
  const [meta, setMeta] = useState({ sex: "" });
  const [files, setFiles] = useState({ left: null, right: null });
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState("");
  const [msg, setMsg] = useState(null);

  const pick = (side) => async (file) => {
    const text = await readFile(file);
    setFiles((f) => ({ ...f, [side]: { name: file.name, text } }));
    setPreview(null);
  };

  const doPreview = async () => {
    setBusy("preview"); setMsg(null);
    try {
      setPreview(await api("sessions/preview", { method: "POST", body: { meta, left_csv: files.left.text, right_csv: files.right.text } }));
    } catch (e) { setMsg({ kind: "error", text: e.message }); }
    setBusy("");
  };

  const save = async () => {
    setBusy("save"); setMsg(null);
    try {
      const res = await api("sessions", {
        method: "POST",
        body: { ...form, meta, left_csv: files.left.text, right_csv: files.right.text, left_name: files.left.name, right_name: files.right.name },
      });
      const warns = [...res.quality.left.warnings, ...res.quality.right.warnings];
      setMsg({ kind: warns.length ? "warn" : "ok", text: `Saved session ${res.id}.${warns.length ? ` Quality notes: ${warns.join(" ")}` : ""}` });
      setForm({ ...empty, recorded_on: form.recorded_on });
      setFiles({ left: null, right: null });
      setPreview(null);
      onSaved();
    } catch (e) { setMsg({ kind: "error", text: e.message }); }
    setBusy("");
  };

  const canSave = form.patient_id.trim() && files.left && files.right;

  return (
    <div className="space-y-6">
      <Card title="Add one labelled session" subtitle="Profile + both raw insole logs + the clinician's risk label for each foot.">
        <div className="space-y-5">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Field label="Patient ID *" hint="Same ID for repeat visits — keeps CV honest."><input className={inputCls} value={form.patient_id} onChange={(e) => setForm({ ...form, patient_id: e.target.value })} /></Field>
            <Field label="Recorded on"><input type="date" className={inputCls} value={form.recorded_on} onChange={(e) => setForm({ ...form, recorded_on: e.target.value })} /></Field>
            <Field label="Left foot label"><LabelSelect schema={schema} value={form.left_label} onChange={(v) => setForm({ ...form, left_label: v })} /></Field>
            <Field label="Right foot label"><LabelSelect schema={schema} value={form.right_label} onChange={(v) => setForm({ ...form, right_label: v })} /></Field>
          </div>
          <ClinicalInputs schema={schema} meta={meta} setMeta={setMeta} />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <FilePick label="Left shoe log *" file={files.left?.name} onFile={pick("left")} />
            <FilePick label="Right shoe log *" file={files.right?.name} onFile={pick("right")} />
          </div>
          <Field label="Clinical notes"><textarea rows={2} className={inputCls} value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} placeholder="e.g. callus under 2nd metatarsal head, left" /></Field>
          {msg && <Notice kind={msg.kind}>{msg.text}</Notice>}
          <div className="flex flex-wrap gap-2">
            <Button variant="ghost" disabled={!files.left || !files.right || !!busy} onClick={doPreview}>{busy === "preview" ? "Extracting…" : "Preview features"}</Button>
            <Button disabled={!canSave || !!busy} onClick={save}>{busy === "save" ? "Saving…" : "Save to dataset"}</Button>
          </div>
          <FeaturePreview preview={preview} />
        </div>
      </Card>

      <BulkImport api={api} schema={schema} onSaved={onSaved} />
    </div>
  );
}

function BulkImport({ api, schema, onSaved }) {
  const [manifest, setManifest] = useState(null);
  const [logs, setLogs] = useState([]);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const run = async () => {
    setBusy(true); setError(""); setResult(null);
    try {
      const files = {};
      for (const f of logs) files[f.name] = f.text;
      const res = await api("sessions/bulk", { method: "POST", body: { manifest_csv: manifest.text, files } });
      setResult(res);
      onSaved();
    } catch (e) { setError(e.message); }
    setBusy(false);
  };

  return (
    <Card title="Bulk import" subtitle="One manifest row per session; select all referenced log files at once."
      actions={<>
        <Button variant="ghost" onClick={() => downloadFrom(api, "templates/manifest.csv", "manifest.csv")}>Manifest template</Button>
        <Button variant="ghost" onClick={() => downloadFrom(api, "templates/sensor_log.csv", "sensor_log_example.csv")}>Example log</Button>
      </>}>
      <div className="space-y-4">
        <p className="text-xs text-slate-500">Manifest columns: <code className="font-mono">{schema.manifest_columns.join(", ")}</code>. Labels: {schema.labels.join(" / ")} (blank = unlabelled). Booleans: yes/no.</p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <FilePick label="Manifest CSV" file={manifest?.name} onFile={async (f) => setManifest({ name: f.name, text: await readFile(f) })} />
          <FilePick label="Sensor log files" multiple file={logs.length ? `${logs.length} file(s) selected` : null}
            onFile={async (fs) => setLogs(await Promise.all(fs.map(async (f) => ({ name: f.name, text: await readFile(f) }))))} />
        </div>
        {error && <Notice kind="error">{error}</Notice>}
        <Button disabled={!manifest || !logs.length || busy} onClick={run}>{busy ? "Importing…" : "Import sessions"}</Button>
        {result && (
          <div className="space-y-2">
            <Notice kind={result.failed ? "warn" : "ok"}>Imported {result.imported} session(s){result.failed ? `, ${result.failed} failed` : ""}.</Notice>
            <div className="max-h-64 overflow-auto text-xs border border-slate-100 rounded-xl">
              {result.rows.map((r) => (
                <div key={r.line} className={`px-3 py-2 border-b border-slate-100 ${r.ok ? "" : "bg-red-50"}`}>
                  <b>Line {r.line}</b> · {r.patient_id || "—"} · {r.ok ? `saved ${r.id}` : r.error}
                  {r.warnings?.map((w) => <span key={w} className="block text-amber-700">⚠ {w}</span>)}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

// =============================================================================================
function Dataset({ api, schema, onChange }) {
  const [rows, setRows] = useState(null);
  const [detail, setDetail] = useState(null);
  const [filter, setFilter] = useState("");
  const [msg, setMsg] = useState(null);

  const load = useCallback(async () => {
    try { setRows(await api("sessions")); } catch (e) { setMsg({ kind: "error", text: e.message }); }
  }, [api]);
  // eslint-disable-next-line react-hooks/set-state-in-effect -- fetch on mount; state is set after await
  useEffect(() => { load(); }, [load]);

  const patch = async (id, body) => {
    try {
      await api(`sessions/${id}`, { method: "PATCH", body });
      await load(); onChange();
    } catch (e) { setMsg({ kind: "error", text: e.message }); }
  };
  const remove = async (id) => {
    if (!confirm("Delete this session and its raw logs permanently?")) return;
    await api(`sessions/${id}`, { method: "DELETE" });
    setDetail(null); await load(); onChange();
  };
  const recompute = async () => {
    setMsg({ kind: "info", text: "Recomputing features…" });
    try {
      const r = await api("features/recompute", { method: "POST", body: {} });
      setMsg({ kind: r.failed.length ? "warn" : "ok", text: `Recomputed ${r.recomputed} session(s)${r.failed.length ? `, ${r.failed.length} failed` : ""}.` });
    } catch (e) { setMsg({ kind: "error", text: e.message }); }
  };

  const shown = (rows || []).filter((r) => !filter || r.patient_id.toLowerCase().includes(filter.toLowerCase()));

  return (
    <div className="space-y-6">
      <Card title="Labelled sessions" subtitle="Edit labels inline, exclude bad recordings, inspect extracted features."
        actions={<>
          <input className={`${inputCls} w-44`} placeholder="Filter patient…" value={filter} onChange={(e) => setFilter(e.target.value)} />
          <Button variant="ghost" onClick={recompute}>Recompute features</Button>
          <Button variant="dark" onClick={() => downloadFrom(api, "dataset/export", "smartsole_features.csv")}>Export ML dataset (CSV)</Button>
        </>}>
        {msg && <div className="mb-4"><Notice kind={msg.kind}>{msg.text}</Notice></div>}
        {!rows ? <p className="text-sm text-slate-500">Loading…</p> : !rows.length ? (
          <Notice>No sessions yet — add data from the <b>Add data</b> tab.</Notice>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[10px] uppercase tracking-widest text-slate-500 border-b border-slate-200">
                  <th className="py-2 pr-3">Patient</th><th className="py-2 pr-3">Recorded</th><th className="py-2 pr-3">Profile</th>
                  <th className="py-2 pr-3">Left</th><th className="py-2 pr-3">Right</th><th className="py-2 pr-3">Quality</th>
                  <th className="py-2 pr-3">Use</th><th className="py-2"></th>
                </tr>
              </thead>
              <tbody>
                {shown.map((r) => {
                  const warns = (r.quality?.left?.warnings?.length || 0) + (r.quality?.right?.warnings?.length || 0);
                  return (
                    <tr key={r.id} className={`border-b border-slate-100 ${r.include ? "" : "opacity-50"}`}>
                      <td className="py-2 pr-3 font-bold">{r.patient_id}<span className="block text-[10px] text-slate-400 font-mono">{r.id} · {r.source}</span></td>
                      <td className="py-2 pr-3 text-slate-600">{r.recorded_on || r.created_at.slice(0, 10)}</td>
                      <td className="py-2 pr-3 text-slate-600 text-xs">{r.meta.sex?.[0]}, {r.meta.age}y, {r.meta.weight_kg}kg{r.meta.neuropathy ? " · LOPS" : ""}{r.meta.prior_ulcer ? " · prior ulcer" : ""}</td>
                      <td className="py-2 pr-3 w-32"><LabelSelect schema={schema} value={r.left_label} onChange={(v) => patch(r.id, { left_label: v })} /></td>
                      <td className="py-2 pr-3 w-32"><LabelSelect schema={schema} value={r.right_label} onChange={(v) => patch(r.id, { right_label: v })} /></td>
                      <td className="py-2 pr-3 text-xs">{warns ? <span className="text-amber-700 font-bold">⚠ {warns}</span> : <span className="text-teal-700 font-bold">OK</span>}</td>
                      <td className="py-2 pr-3"><input type="checkbox" className="w-4 h-4 accent-teal-600" checked={r.include} onChange={(e) => patch(r.id, { include: e.target.checked })} /></td>
                      <td className="py-2 text-right whitespace-nowrap">
                        <Button variant="ghost" className="!px-2 !py-1 text-xs" onClick={async () => setDetail(await api(`sessions/${r.id}`))}>Details</Button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {detail && (
        <Card title={`Session ${detail.id} · ${detail.patient_id}`} subtitle={detail.notes || "No notes"}
          actions={<><Button variant="danger" onClick={() => remove(detail.id)}>Delete</Button><Button variant="ghost" onClick={() => setDetail(null)}>Close</Button></>}>
          <div className="flex gap-2 mb-4"><span className="text-sm text-slate-500">Labels:</span><LabelBadge label={detail.left_label} /><LabelBadge label={detail.right_label} /></div>
          {detail.features && <FeaturePreview preview={{ features: detail.features, feature_labels: detail.feature_labels, quality: detail.quality }} />}
        </Card>
      )}
    </div>
  );
}

// =============================================================================================
function Train({ api, schema, overview, onTrained }) {
  const [source, setSource] = useState("real");
  const [algos, setAlgos] = useState(Object.keys(schema.algorithms));
  const [activate, setActivate] = useState(true);
  // Training runs as a background job on the ML service; we start it, then poll its status.
  const [busy, setBusy] = useState(() => overview?.training_job?.status === "running");
  const [jobKind, setJobKind] = useState(() => overview?.training_job?.kind || "admin");
  const [elapsed, setElapsed] = useState(0);
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  const ready = overview?.readiness;

  useEffect(() => {
    if (!busy) return undefined;
    let stopped = false;
    let timer;
    const tick = async () => {
      try {
        const s = await api("train/status");
        if (stopped) return;
        if (s.started_at) setElapsed(Math.max(0, Math.round((Date.now() - Date.parse(s.started_at)) / 1000)));
        if (s.status === "running") { timer = setTimeout(tick, 3000); return; }
        if (s.status === "succeeded") { setReport(s.report); onTrained(); }
        else if (s.status === "failed") setError(s.error || "Training failed.");
        setBusy(false);
      } catch (e) {
        if (!stopped) { setError(e.message); setBusy(false); }
      }
    };
    timer = setTimeout(tick, 1500);
    return () => { stopped = true; clearTimeout(timer); };
  }, [busy, api, onTrained]);

  const run = async () => {
    setError(""); setReport(null); setElapsed(0);
    try {
      await api("train", { method: "POST", body: { source, algorithms: algos, activate } });
      setJobKind("admin");
      setBusy(true);
    } catch (e) { setError(e.message); }
  };

  const SOURCES = [
    ["real", "Real data only", "Recommended once the readiness checks pass. Metrics reflect real clinical performance."],
    ["real+synthetic", "Real + synthetic prior", "Augments a small real dataset with simulated rule-labelled data. Metrics are computed on real samples only."],
    ["synthetic", "Synthetic only (rebuild starter)", "Re-creates the bootstrap model from simulated data. For demos / testing only."],
  ];

  return (
    <div className="space-y-6">
      <Card title="Train a new model" subtitle="All selected algorithms are cross-validated (grouped by patient); the best by macro-F1 is saved as a new version.">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-3">
            <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500">Training data</p>
            {SOURCES.map(([k, l, d]) => (
              <label key={k} className={`flex gap-3 p-3 rounded-xl border cursor-pointer ${source === k ? "border-teal-400 bg-teal-50/50" : "border-slate-200"}`}>
                <input type="radio" name="src" className="mt-1 accent-teal-600" checked={source === k} onChange={() => setSource(k)} />
                <span><span className="block font-bold text-sm">{l}</span><span className="block text-xs text-slate-500">{d}</span></span>
              </label>
            ))}
          </div>
          <div className="space-y-4">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mb-2">Candidate algorithms</p>
              {Object.entries(schema.algorithms).map(([k, l]) => (
                <label key={k} className="flex items-center gap-2 text-sm py-1">
                  <input type="checkbox" className="accent-teal-600" checked={algos.includes(k)} onChange={(e) => setAlgos(e.target.checked ? [...algos, k] : algos.filter((a) => a !== k))} />{l}
                </label>
              ))}
            </div>
            <label className="flex items-center gap-2 text-sm font-semibold">
              <input type="checkbox" className="accent-teal-600" checked={activate} onChange={(e) => setActivate(e.target.checked)} />Activate the new model when done
            </label>
            {source === "real" && ready && !ready.trainable && (
              <Notice kind="warn">Dataset not ready yet: {ready.checks.filter((c) => !c.ok && !c.optional).map((c) => c.text).join(" ")}</Notice>
            )}
            <Button className="w-full py-3" disabled={busy || !algos.length || (source === "real" && !ready?.trainable)} onClick={run}>
              {busy ? `Training & cross-validating… ${elapsed}s` : "Start training"}
            </Button>
          </div>
        </div>
        {busy && (
          <div className="mt-4">
            <Notice kind="info">
              {jobKind === "bootstrap" ? "The starter model is being trained after a fresh start." : "Training is running on the ML service."}{" "}
              It usually takes 1–3 minutes. You can leave this tab — the result will be in the <b>Models</b> tab.
            </Notice>
          </div>
        )}
        {error && <div className="mt-4"><Notice kind="error">{error}</Notice></div>}
      </Card>
      {report && (
        <Card title="Training report" subtitle={`${report.activated ? "Activated" : "Saved (not active)"} · ${report.duration_s}s`}>
          <TrainingReport report={report} />
        </Card>
      )}
    </div>
  );
}

// =============================================================================================
function Models({ api, overview, onChange }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const models = overview?.models || [];

  const act = async (fn) => {
    setError("");
    try { await fn(); onChange(); } catch (e) { setError(e.message); }
  };

  return (
    <div className="space-y-6">
      <Card title="Model registry" subtitle="Every training run is versioned. Roll back by activating an older version.">
        {error && <div className="mb-4"><Notice kind="error">{error}</Notice></div>}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-[10px] uppercase tracking-widest text-slate-500 border-b border-slate-200">
                <th className="py-2 pr-3">Version</th><th className="py-2 pr-3">Algorithm</th><th className="py-2 pr-3">Data</th>
                <th className="py-2 pr-3">Feet</th><th className="py-2 pr-3">Macro F1</th><th className="py-2 pr-3">AUC</th>
                <th className="py-2 pr-3">At-risk sens.</th><th className="py-2"></th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.version} className={`border-b border-slate-100 ${m.is_active ? "bg-teal-50/60" : ""}`}>
                  <td className="py-2 pr-3 font-mono text-xs font-bold">{m.version}{m.is_active && <span className="ml-2 px-1.5 py-0.5 rounded bg-teal-600 text-white text-[9px] font-black">ACTIVE</span>}<span className="block text-[10px] text-slate-400 font-sans font-normal">{new Date(m.created_at).toLocaleString()}</span></td>
                  <td className="py-2 pr-3">{m.algorithm.replaceAll("_", " ")}</td>
                  <td className="py-2 pr-3 text-xs">{m.data_source}</td>
                  <td className="py-2 pr-3">{m.n_samples}</td>
                  <td className="py-2 pr-3 font-semibold">{pct(m.metrics.macro_f1)}</td>
                  <td className="py-2 pr-3">{m.metrics.roc_auc == null ? "—" : m.metrics.roc_auc.toFixed(3)}</td>
                  <td className="py-2 pr-3">{pct(m.metrics.at_risk_sensitivity)}</td>
                  <td className="py-2 text-right whitespace-nowrap space-x-1">
                    <Button variant="ghost" className="!px-2 !py-1 text-xs" onClick={async () => setDetail(await api(`models/${m.version}`))}>Report</Button>
                    {!m.is_active && <Button className="!px-2 !py-1 text-xs" onClick={() => act(() => api(`models/${m.version}/activate`, { method: "POST", body: {} }))}>Activate</Button>}
                    {!m.is_active && <Button variant="danger" className="!px-2 !py-1 text-xs" onClick={() => confirm(`Delete model ${m.version}?`) && act(() => api(`models/${m.version}`, { method: "DELETE" }))}>Delete</Button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      {detail && (
        <Card title={`Report · ${detail.version}`} actions={<Button variant="ghost" onClick={() => setDetail(null)}>Close</Button>}>
          <TrainingReport report={detail.report} />
        </Card>
      )}
    </div>
  );
}

// =============================================================================================
function Predictions({ api, schema, onChange }) {
  const [rows, setRows] = useState(null);
  const [open, setOpen] = useState(null);
  const [form, setForm] = useState({});
  const [msg, setMsg] = useState(null);

  const load = useCallback(async () => {
    try { setRows(await api("predictions")); } catch (e) { setMsg({ kind: "error", text: e.message }); }
  }, [api]);
  // eslint-disable-next-line react-hooks/set-state-in-effect -- fetch on mount; state is set after await
  useEffect(() => { load(); }, [load]);

  const promote = async (id) => {
    try {
      const r = await api(`predictions/${id}/promote`, { method: "POST", body: form });
      setMsg({ kind: "ok", text: `Added to dataset as session ${r.id}.` });
      setOpen(null); await load(); onChange();
    } catch (e) { setMsg({ kind: "error", text: e.message }); }
  };

  return (
    <Card title="Prediction log" subtitle="Every dashboard assessment is stored. When the clinical outcome is known, label it and add it to the training set — the model keeps learning from real use.">
      {msg && <div className="mb-4"><Notice kind={msg.kind}>{msg.text}</Notice></div>}
      {!rows ? <p className="text-sm text-slate-500">Loading…</p> : !rows.length ? <Notice>No predictions logged yet.</Notice> : (
        <div className="space-y-2">
          {rows.map((p) => (
            <div key={p.id} className="border border-slate-200 rounded-xl">
              <div className="flex flex-wrap items-center gap-3 px-4 py-3">
                <span className="text-xs text-slate-500 w-36">{new Date(p.created_at).toLocaleString()}</span>
                <span className="text-xs text-slate-600">{p.meta.sex?.[0]}, {p.meta.age}y, {p.meta.weight_kg}kg</span>
                <span className="text-xs">L <LabelBadge label={p.result.left.riskLevel.toLowerCase()} /> {p.result.left.dfuScore}</span>
                <span className="text-xs">R <LabelBadge label={p.result.right.riskLevel.toLowerCase()} /> {p.result.right.dfuScore}</span>
                <span className="text-[10px] text-slate-400 font-mono">{p.model_version}</span>
                <span className="flex-1" />
                {p.promoted_session_id ? <span className="text-xs font-bold text-teal-700">✓ in dataset ({p.promoted_session_id})</span> : (
                  <Button variant="ghost" className="!px-2 !py-1 text-xs" onClick={() => { setOpen(open === p.id ? null : p.id); setForm({ patient_id: "", left_label: null, right_label: null, notes: "" }); }}>Label outcome</Button>
                )}
              </div>
              {open === p.id && (
                <div className="px-4 pb-4 grid grid-cols-2 md:grid-cols-5 gap-3 items-end">
                  <Field label="Patient ID *"><input className={inputCls} value={form.patient_id} onChange={(e) => setForm({ ...form, patient_id: e.target.value })} /></Field>
                  <Field label="True left label"><LabelSelect schema={schema} value={form.left_label} onChange={(v) => setForm({ ...form, left_label: v })} /></Field>
                  <Field label="True right label"><LabelSelect schema={schema} value={form.right_label} onChange={(v) => setForm({ ...form, right_label: v })} /></Field>
                  <Field label="Notes"><input className={inputCls} value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} /></Field>
                  <Button disabled={!form.patient_id?.trim()} onClick={() => promote(p.id)}>Add to dataset</Button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}

// =============================================================================================
function Guide({ schema, api }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card title="What data the model needs">
        <ul className="text-sm text-slate-700 space-y-3 list-disc pl-5">
          <li><b>One session = both feet, same time.</b> Upload the left and right insole logs recorded together. Bilateral comparison (ΔT, pressure asymmetry) is a core signal.</li>
          <li><b>Recording protocol.</b> Let the insole settle in the shoe for ~5 min, then record <b>2–5 min of normal walking</b> (and optionally 1 min standing). Keep the protocol the same for every patient.</li>
          <li><b>Labels come from a clinician&apos;s foot exam</b>, ideally confirmed at follow-up (e.g. 3–6 months): was an ulcer or pre-ulcerative lesion present / did one develop on that foot?</li>
          <li><b>Patient ID</b> must be stable across visits so the same person never appears in both training and validation folds.</li>
          <li><b>Profile:</b> age, sex and weight are required. Clinical history (neuropathy, PAD, prior ulcer…) is optional but strongly predictive — fill it whenever known.</li>
          <li><b>Volume:</b> minimum ≈ {schema.minimums.per_class} feet per class and {schema.minimums.patients}+ patients to train; aim for {schema.minimums.recommended}+ labelled feet across all three classes for a reliable model.</li>
          <li><b>Balance:</b> high-risk feet are rare — prioritise collecting them (diabetic foot clinics, wound-care follow-ups).</li>
        </ul>
      </Card>

      <Card title="Risk labels (per foot)">
        <div className="space-y-3">
          {schema.labels.map((l) => (
            <div key={l} className="flex gap-3 items-start">
              <div className="w-24 shrink-0"><LabelBadge label={l} /></div>
              <p className="text-sm text-slate-700">{schema.label_guide[l]}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Sensor log format" actions={<Button variant="ghost" onClick={() => downloadFrom(api, "templates/sensor_log.csv", "sensor_log_example.csv")}>Download example</Button>}>
        <pre className="bg-slate-900 text-teal-200 text-xs rounded-xl p-4 overflow-x-auto">{`${schema.sensor_header}
20:45:18.100, 0.95, 1.31, 1.42, 0.41, 30.4, 42.1
20:45:18.200, 1.02, 1.36, 1.10, 0.44, 30.4, 42.2
...`}</pre>
        <ul className="text-xs text-slate-600 mt-3 space-y-1 list-disc pl-5">
          <li>Pressure columns are multiplied by <b>{schema.pressure_scale}</b> to get kPa (set <code>PRESSURE_SCALE</code> on the ML service if your firmware differs).</li>
          <li>Temp in °C, humidity in % RH. Delimiter: comma, semicolon, tab or space. Header optional.</li>
          <li>Time as <code>HH:MM:SS.mmm</code>, seconds or milliseconds. Missing time column → 10 Hz is assumed.</li>
        </ul>
      </Card>

      <Card title={`Features computed per foot (${schema.features.length})`} subtitle={`Feature set v${schema.feature_version} — computed automatically from the raw logs`}>
        <div className="max-h-80 overflow-auto grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1">
          {schema.features.map((f) => (
            <p key={f.name} className="text-xs"><span className="font-semibold text-slate-700">{f.label}</span> <span className="text-slate-400 font-mono">{f.name}</span></p>
          ))}
        </div>
      </Card>

      <Card title="Model & validation" className="lg:col-span-2">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-slate-700">
          <div><p className="font-black mb-1">Why gradient-boosted trees</p><p>The data is tabular (engineered features per foot), small and partly missing — tree ensembles are the strongest, most data-efficient choice there and handle missing clinical fields natively. Logistic regression is kept as an interpretable baseline.</p></div>
          <div><p className="font-black mb-1">Honest evaluation</p><p>Stratified group k-fold CV grouped by patient. Reported: macro-F1, balanced accuracy, ROC-AUC and at-risk sensitivity/specificity (moderate+high vs low) — sensitivity matters most for screening.</p></div>
          <div><p className="font-black mb-1">Explainability & safety</p><p>Each prediction lists its top risk drivers, flags inputs unlike the training data, and always shows evidence-based clinical alerts (ΔT &gt; 2.2 °C, peak pressure &gt; 200 kPa, humidity) independent of the model.</p></div>
        </div>
      </Card>
    </div>
  );
}
