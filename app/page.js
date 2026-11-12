// app/page.js
"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------
// Colour by peak (95th pct) in-shoe pressure. 200 kPa is the IWGDF offloading target.
const PEAK_CRITICAL_KPA = 600;
const getPressureColor = (peakKpa) => {
  if (peakKpa < 200) return "#2dd4bf"; // teal
  if (peakKpa < 300) return "#60a5fa"; // blue
  if (peakKpa < 450) return "#fbbf24"; // amber
  return "#f87171"; // red
};

const RISK_STYLE = {
  LOW: { text: "text-teal-700", bg: "bg-teal-500", soft: "bg-teal-50 border-teal-200", ring: "#14b8a6", label: "Low risk" },
  MODERATE: { text: "text-amber-700", bg: "bg-amber-500", soft: "bg-amber-50 border-amber-200", ring: "#f59e0b", label: "Moderate risk" },
  HIGH: { text: "text-red-700", bg: "bg-red-500", soft: "bg-red-50 border-red-200", ring: "#ef4444", label: "High risk" },
};

const BOOL_FEATURES = new Set(["pt_neuropathy", "pt_pad", "pt_deformity", "pt_prior_ulcer", "pt_prior_amputation"]);
const formatDriver = (d) => {
  if (BOOL_FEATURES.has(d.feature)) return d.value ? "present" : "absent";
  if (d.feature === "pt_sex_male") return d.value ? "male" : "female";
  return `value ${d.value ?? "—"} · typical ${d.typical ?? "—"}`;
};

const ZONE_LABEL = { toe: "Toe", met: "Metatarsal", mid: "Midfoot", heel: "Heel" };

const CLINICAL_TOGGLES = [
  ["neuropathy", "Neuropathy (LOPS)"],
  ["deformity", "Foot deformity"],
  ["pad", "Peripheral artery disease"],
  ["prior_ulcer", "Previous ulcer"],
  ["prior_amputation", "Previous amputation"],
];

const readFile = (file) =>
  new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = (e) => resolve(e.target.result);
    r.onerror = reject;
    r.readAsText(file);
  });

// ---------------------------------------------------------------------------------------------
// Small components (defined at module level so they don't remount on every render)
// ---------------------------------------------------------------------------------------------
const InfoTooltip = ({ title, desc, optimal, danger }) => {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <div className="relative inline-block ml-2 z-40" onMouseEnter={() => setIsOpen(true)} onMouseLeave={() => setIsOpen(false)}>
      <button
        type="button"
        aria-label={`About ${title}`}
        onClick={() => setIsOpen(!isOpen)}
        onBlur={() => setIsOpen(false)}
        className="cursor-help w-5 h-5 rounded-full bg-slate-200 border border-slate-300 text-slate-600 flex items-center justify-center text-[10px] font-black hover:bg-teal-500 hover:text-white focus:bg-teal-500 focus:text-white transition-all shadow-sm focus:outline-none"
      >
        i
      </button>
      <div className={`absolute bottom-full left-1/2 -translate-x-1/2 mb-3 w-64 p-4 bg-slate-900/95 backdrop-blur-xl text-slate-200 text-xs rounded-2xl shadow-2xl transition-all duration-300 text-left border border-slate-700 pointer-events-none ${isOpen ? "opacity-100 visible translate-y-0" : "opacity-0 invisible translate-y-2"}`}>
        <p className="font-black text-teal-400 mb-2 text-sm">{title}</p>
        <p className="mb-3 leading-relaxed text-slate-300">{desc}</p>
        {(optimal || danger) && (
          <div className="space-y-1.5 bg-slate-800/80 p-3 rounded-xl border border-slate-700/50">
            {optimal && <p><span className="text-teal-400 font-bold uppercase tracking-wider text-[10px]">Optimal:</span><br />{optimal}</p>}
            {danger && <p><span className="text-red-400 font-bold uppercase tracking-wider text-[10px]">Danger:</span><br />{danger}</p>}
          </div>
        )}
        <div className="absolute top-full left-1/2 -translate-x-1/2 border-8 border-transparent border-t-slate-900/95"></div>
      </div>
    </div>
  );
};

const FootMapZones = ({ pressures, isLeft }) => {
  const p = pressures || { heel: 100, toe: 100, met: 100, mid: 50 };
  const renderZone = (cx, cy, r, value) => {
    const isCritical = value >= PEAK_CRITICAL_KPA;
    return (
      <g>
        {isCritical && <circle cx={cx} cy={cy} r={r * 1.5} fill="#f87171" className="opacity-50 animate-ping blur-md origin-center" />}
        <circle cx={cx} cy={cy} r={r} fill={getPressureColor(value)} className="opacity-90 blur-[3px] transition-colors duration-1000" />
      </g>
    );
  };
  return (
    <svg viewBox="0 0 100 240" className={`w-full h-full max-h-72 mx-auto drop-shadow-2xl transition-transform duration-700 hover:scale-105 ${isLeft ? "scale-x-[-1]" : ""}`}>
      <path d="M30,10 C10,30 0,70 10,110 C15,140 20,180 20,200 C20,220 35,235 50,235 C65,235 80,220 80,200 C80,180 85,140 90,110 C100,70 90,30 70,10 C60,0 40,0 30,10 Z" fill="rgba(255,255,255,0.6)" stroke="rgba(255,255,255,0.9)" strokeWidth="3" />
      {renderZone(35, 40, 18, p.toe)}
      {renderZone(35, 90, 16, p.met)}
      {renderZone(50, 145, 14, p.mid)}
      {renderZone(50, 200, 20, p.heel)}
    </svg>
  );
};

const RiskGauge = ({ score, level }) => {
  const r = 52;
  const c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, score)) / 100;
  const color = (RISK_STYLE[level] || RISK_STYLE.LOW).ring;
  return (
    <div className="relative w-32 h-32 shrink-0">
      <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
        <circle cx="60" cy="60" r={r} fill="none" stroke="rgba(148,163,184,0.25)" strokeWidth="11" />
        <circle cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="11" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={c * (1 - pct)} style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(.16,1,.3,1)" }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-black text-slate-900 leading-none">{score}</span>
        <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500 mt-1">/ 100</span>
      </div>
    </div>
  );
};

const ProbabilityBars = ({ probabilities }) => (
  <div className="space-y-2 w-full">
    {[["low", "Low", "bg-teal-500"], ["moderate", "Moderate", "bg-amber-500"], ["high", "High", "bg-red-500"]].map(([k, label, color]) => (
      <div key={k}>
        <div className="flex justify-between text-[11px] font-bold text-slate-600 mb-1">
          <span>{label}</span>
          <span>{Math.round((probabilities?.[k] || 0) * 100)}%</span>
        </div>
        <div className="h-2 rounded-full bg-slate-200/70 overflow-hidden">
          <div className={`h-full ${color} rounded-full`} style={{ width: `${(probabilities?.[k] || 0) * 100}%`, transition: "width 1s ease" }} />
        </div>
      </div>
    ))}
  </div>
);

const GlassBackground = () => (
  <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none print:hidden">
    {/* eslint-disable-next-line @next/next/no-img-element */}
    <img src="https://images.unsplash.com/photo-1581594693702-fbdc51b2763b?q=80&w=2070&auto=format&fit=crop" alt="" className="absolute inset-0 w-full h-full object-cover opacity-[0.35] mix-blend-luminosity" />
    <div className="absolute inset-0 bg-gradient-to-br from-teal-50/90 via-slate-50/90 to-blue-50/90 backdrop-blur-[40px]"></div>
    <div className="absolute top-[-10%] left-[-10%] w-[50vw] h-[50vw] rounded-full bg-teal-400/30 blur-[120px] animate-blob"></div>
    <div className="absolute bottom-[-10%] right-[-10%] w-[40vw] h-[40vw] rounded-full bg-blue-500/20 blur-[120px] animate-blob animation-delay-2000"></div>
  </div>
);

const MadeInPecBadge = ({ className = "mb-6" }) => (
  <div className={`glass-card px-5 py-2.5 rounded-full inline-flex items-center gap-3 shadow-sm border border-white/80 animate-fade-in ${className}`}>
    <span className="w-7 h-7 bg-teal-600 text-white rounded-full flex items-center justify-center font-black text-[10px] shadow-sm">PEC</span>
    <span className="text-xs font-black text-slate-800 tracking-widest uppercase">Made In PEC</span>
  </div>
);

const AnimatedHeroVisual = () => (
  <div className="relative w-full h-[500px] hidden lg:flex items-center justify-center animate-fade-in delay-200">
    <div className="absolute w-72 h-72 bg-teal-400/30 rounded-full blur-[80px] animate-pulse"></div>
    <div className="glass-card p-5 rounded-3xl absolute top-10 right-12 z-20 animate-[float_4s_ease-in-out_infinite] shadow-xl border border-white">
      <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest mb-2 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-teal-500 animate-ping"></span> Live Telemetry
      </p>
      <div className="flex gap-2 items-end h-12">
        <div className="w-2 h-6 bg-teal-400 rounded-full animate-[pulse_1s_infinite]"></div>
        <div className="w-2 h-10 bg-blue-400 rounded-full animate-[pulse_1.5s_infinite]"></div>
        <div className="w-2 h-4 bg-teal-300 rounded-full animate-[pulse_1.2s_infinite]"></div>
        <div className="w-2 h-8 bg-blue-500 rounded-full animate-[pulse_0.8s_infinite]"></div>
        <div className="w-2 h-12 bg-teal-500 rounded-full animate-[pulse_1.1s_infinite]"></div>
      </div>
    </div>
    <div className="relative w-48 h-[360px] glass-card rounded-[4rem] border-2 border-white/60 flex flex-col items-center py-10 justify-between animate-[float_6s_ease-in-out_infinite] z-10 shadow-2xl">
      <div className="w-12 h-12 rounded-full border-[4px] border-teal-400 bg-white/50 shadow-[0_0_20px_#2dd4bf] animate-[pulse_2s_infinite]"></div>
      <div className="w-10 h-10 rounded-full border-[4px] border-blue-400 bg-white/50 shadow-[0_0_20px_#60a5fa] animate-[pulse_2.5s_infinite]"></div>
      <div className="w-10 h-10 rounded-full border-[4px] border-amber-400 bg-white/50 shadow-[0_0_20px_#fbbf24] animate-[pulse_3s_infinite]"></div>
      <div className="w-16 h-16 rounded-full border-[4px] border-teal-500 bg-white/50 shadow-[0_0_30px_#14b8a6] animate-[pulse_1.5s_infinite]"></div>
    </div>
    <div className="glass-card px-6 py-4 rounded-2xl shadow-xl flex items-center gap-4 absolute bottom-12 left-0 z-20 animate-[float_5s_ease-in-out_infinite] border border-white">
      <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center text-xl shadow-inner">🧠</div>
      <div>
        <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">Predictive Model</p>
        <p className="text-sm font-black text-slate-800">ML Risk Engine · Gradient-Boosted</p>
      </div>
    </div>
  </div>
);

const UploadTile = ({ side, name, info, onFile }) => {
  const [drag, setDrag] = useState(false);
  return (
    <label
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); }}
      className={`glass-input border-dashed border-2 rounded-3xl p-7 text-center cursor-pointer relative group transition-all hover:bg-white/80 block ${drag ? "border-teal-500 bg-white/80" : name ? "border-teal-400" : "hover:border-teal-400"}`}
    >
      <input type="file" accept=".csv,.txt" onChange={(e) => e.target.files[0] && onFile(e.target.files[0])} className="sr-only" />
      <div className={`w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-3 shadow-sm group-hover:scale-110 transition-transform ${name ? "bg-teal-600 text-white" : "bg-white/90 text-teal-600"}`}>
        <span className="font-black text-xl">{name ? "✓" : side}</span>
      </div>
      <p className="text-slate-800 font-bold truncate">{name || `${side === "L" ? "Left" : "Right"} Shoe Log`}</p>
      <p className="text-xs text-slate-500 font-medium mt-1">{info || "Drop CSV here or click to browse"}</p>
    </label>
  );
};

const FootCard = ({ title, side, data, isLeft }) => {
  const style = RISK_STYLE[data.riskLevel] || RISK_STYLE.LOW;
  return (
    <div className="lg:col-span-6 glass-card rounded-[3rem] p-7 md:p-8 flex flex-col relative shadow-xl break-inside-avoid">
      <div className="w-full flex justify-between items-center mb-6 border-b border-slate-300/40 pb-4">
        <h3 className="text-2xl font-black text-slate-900 tracking-tight">{title}</h3>
        <span className={`px-4 py-2 rounded-xl text-xs font-black uppercase tracking-widest shadow-sm ${data.riskLevel === "HIGH" ? "bg-red-500 text-white animate-pulse" : data.riskLevel === "MODERATE" ? "bg-amber-400 text-slate-900" : "bg-white text-slate-700 border border-slate-200"}`}>
          {style.label}
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-5 lg:grid-cols-1 xl:grid-cols-5 gap-6 items-center mb-6">
        <div className="sm:col-span-2 lg:col-span-1 xl:col-span-2 h-64"><FootMapZones pressures={data.peakPressures || data.pressures} isLeft={isLeft} /></div>
        <div className="sm:col-span-3 lg:col-span-1 xl:col-span-3 space-y-5">
          <div className="flex items-center gap-5">
            <RiskGauge score={data.dfuScore} level={data.riskLevel} />
            <div className="min-w-0">
              <div className="flex items-center"><p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">ML Risk Score</p>
                <InfoTooltip title="ML Risk Score" desc="Probability-weighted risk (0 = certainly low, 100 = certainly high) from the trained model, using pressure, temperature, humidity, left/right asymmetry and clinical history." optimal="0 - 30" danger="> 60" />
              </div>
              <p className="text-xs text-slate-500 font-semibold mt-1">Model confidence {Math.round(data.confidence * 100)}%</p>
            </div>
          </div>
          <ProbabilityBars probabilities={data.probabilities} />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
        <div className="glass-input p-4 rounded-2xl">
          <div className="flex items-center"><p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Skin Temp</p><InfoTooltip title="Local Skin Temp" desc="Mean plantar temperature over the recording." optimal="27 - 34 °C" danger="> 35 °C or > 2.2 °C warmer than other foot" /></div>
          <p className="text-2xl font-black text-orange-500">{data.temp}°</p>
        </div>
        <div className="glass-input p-4 rounded-2xl">
          <div className="flex items-center"><p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Humidity</p><InfoTooltip title="In-shoe Humidity" desc="Prolonged moisture softens skin (maceration) and raises ulcer risk." optimal="< 60 %" danger="> 75 %" /></div>
          <p className="text-2xl font-black text-teal-600">{data.humidity}%</p>
        </div>
        <div className="glass-input p-4 rounded-2xl">
          <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Peak kPa</p>
          <p className="text-2xl font-black text-blue-600">{Math.max(...Object.values(data.peakPressures || data.pressures))}</p>
        </div>
        <div className="glass-input p-4 rounded-2xl">
          <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Cadence</p>
          <p className="text-2xl font-black text-slate-700">{data.cadence ? Math.round(data.cadence) : "—"}<span className="text-xs text-slate-500 font-bold"> /min</span></p>
        </div>
      </div>

      <div className="glass-input p-4 rounded-2xl mb-5">
        <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-3">Zone pressure · mean / peak (kPa)</p>
        <div className="grid grid-cols-4 gap-2 text-center">
          {["toe", "met", "mid", "heel"].map((z) => (
            <div key={z}>
              <div className="w-3 h-3 rounded-full mx-auto mb-1" style={{ background: getPressureColor(data.peakPressures?.[z] ?? data.pressures[z]) }} />
              <p className="text-[10px] font-bold text-slate-500 uppercase">{ZONE_LABEL[z]}</p>
              <p className="text-sm font-black text-slate-800">{data.pressures[z]}<span className="text-slate-400 font-bold"> / {data.peakPressures?.[z] ?? "—"}</span></p>
            </div>
          ))}
        </div>
      </div>

      <div className="glass-input p-5 rounded-2xl mb-5">
        <div className="flex items-center mb-3"><p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Why this score</p>
          <InfoTooltip title="Top risk drivers" desc="Features that push this foot's risk up the most: how much the score would drop if that single measurement looked like a typical low-risk foot in the training data." />
        </div>
        {data.drivers?.length ? (
          <ul className="space-y-2.5">
            {data.drivers.map((d) => (
              <li key={d.feature} className="flex items-center gap-3">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-bold text-slate-800 truncate">{d.label}</p>
                  <p className="text-[11px] text-slate-500 font-medium">{formatDriver(d)}</p>
                </div>
                <div className="w-24 h-2 bg-slate-200/70 rounded-full overflow-hidden"><div className={`h-full ${style.bg}`} style={{ width: `${Math.min(100, d.impact * 2)}%` }} /></div>
                <span className="text-xs font-black text-slate-700 w-12 text-right">+{d.impact}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-slate-500 font-medium">No individual measurement is pushing risk up — readings look typical of low-risk feet.</p>
        )}
      </div>

      <div className="grid grid-cols-4 gap-2 text-center">
        {[["PRS", "Pressure", "text-blue-600"], ["TRS", "Thermal", "text-orange-500"], ["MRS", "Moisture", "text-teal-600"], ["TBI", "Tissue", "text-purple-600"]].map(([k, l, c]) => (
          <div key={k} className="glass-input rounded-xl py-2">
            <p className="text-[9px] font-bold text-slate-500 uppercase tracking-wider">{l} idx</p>
            <p className={`text-lg font-black ${c}`}>{data.scores[k]}</p>
          </div>
        ))}
      </div>
      <p className="text-[10px] text-slate-400 font-semibold mt-2 text-center">Rule-based indices are model inputs, shown for reference ({side}).</p>
    </div>
  );
};

// ---------------------------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------------------------
export default function Home() {
  const [step, setStep] = useState(0);
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("Male");
  const [weight, setWeight] = useState("");
  const [clinical, setClinical] = useState({ height_cm: "", diabetes_years: "", hba1c: "" });
  const [showClinical, setShowClinical] = useState(false);

  const [left, setLeft] = useState({ csv: "", name: "", info: "" });
  const [right, setRight] = useState({ csv: "", name: "", info: "" });

  const [progress, setProgress] = useState(0);
  const [scanMessage, setScanMessage] = useState("Initializing Bio-Sensors...");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [demoLoading, setDemoLoading] = useState("");

  const [sessionID, setSessionID] = useState("---");
  const [currentDate, setCurrentDate] = useState("---");
  const requestRef = useRef(null);

  useEffect(() => {
     
    setSessionID(`PT-${Math.floor(1000 + Math.random() * 9000)}`);
    setCurrentDate(new Date().toLocaleDateString());
  }, []);

  const describe = (text) => {
    const rows = text.trim().split(/\r?\n/).filter(Boolean).length - 1;
    return `${Math.max(rows, 0)} samples`;
  };

  const handleFile = (setter) => async (file) => {
    const csv = await readFile(file);
    setter({ csv, name: file.name, info: describe(csv) });
  };

  const loadDemo = async (scenario) => {
    setDemoLoading(scenario);
    setError("");
    try {
      const res = await fetch(`/api/demo/${scenario}`);
      const d = await res.json();
      if (!res.ok) throw new Error(d.error || "Could not load demo");
      setAge(String(d.meta.age));
      setGender(d.meta.sex);
      setWeight(String(d.meta.weight_kg));
      setClinical({
        height_cm: d.meta.height_cm ?? "", diabetes_years: d.meta.diabetes_years ?? "", hba1c: d.meta.hba1c ?? "",
        neuropathy: d.meta.neuropathy, pad: d.meta.pad, deformity: d.meta.deformity,
        prior_ulcer: d.meta.prior_ulcer, prior_amputation: d.meta.prior_amputation,
      });
      setShowClinical(true);
      setLeft({ csv: d.left_csv, name: `left_${scenario}.csv`, info: describe(d.left_csv) });
      setRight({ csv: d.right_csv, name: `right_${scenario}.csv`, info: describe(d.right_csv) });
    } catch (e) {
      setError(e.message);
    } finally {
      setDemoLoading("");
    }
  };

  const runAnalysis = () => {
    setError("");
    setResult(null);
    setProgress(0);
    setScanMessage("Synchronizing Telemetry Arrays...");
    setStep(2);
    requestRef.current = fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ age, gender, weight, clinical, leftCsvData: left.csv, rightCsvData: right.csv }),
    }).then(async (res) => {
      const data = await res.json().catch(() => ({ error: "Invalid server response" }));
      return res.ok ? data : { error: data.error || `Request failed (${res.status})` };
    }).catch(() => ({ error: "Network error — is the app server running?" }));
  };

  useEffect(() => {
    if (step !== 2) return;
    const messages = [
      [20, "Extracting gait & plantar-pressure features..."],
      [45, "Computing bilateral thermal asymmetry (ΔT)..."],
      [70, "Running ML risk model..."],
      [88, "Explaining risk drivers..."],
    ];
    const interval = setInterval(() => {
      setProgress((old) => {
        const next = Math.min(old + 2, 100);
        const msg = messages.filter(([p]) => next >= p).pop();
        if (msg) setScanMessage(msg[1]);
        if (next >= 100) {
          clearInterval(interval);
          requestRef.current?.then((data) => {
            if (data.error) { setError(data.error); setStep(1); }
            else { setResult(data); setStep(3); }
          });
        }
        return next;
      });
    }, 45);
    return () => clearInterval(interval);
  }, [step]);

  const reset = () => {
    setStep(0);
    setResult(null);
  };

  const overallStyle = result ? RISK_STYLE[(result.overallRisk || "low").toUpperCase()] : RISK_STYLE.LOW;

  return (
    <>
      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes blob { 0%, 100% { transform: translate(0px, 0px) scale(1); } 50% { transform: translate(20px, -30px) scale(1.05); } }
        .animate-blob { animation: blob 15s infinite; }
        .animation-delay-2000 { animation-delay: 2s; }
        .glass-card { background: rgba(255, 255, 255, 0.55); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px); border: 1px solid rgba(255, 255, 255, 0.7); box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.05); }
        .glass-input { background: rgba(255, 255, 255, 0.6); border: 1px solid rgba(255, 255, 255, 0.9); box-shadow: inset 0 2px 10px 0 rgba(0, 0, 0, 0.03); }
        .animate-fade-in { animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards; opacity: 0; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-15px); } }
        .delay-100 { animation-delay: 100ms; } .delay-200 { animation-delay: 200ms; } .delay-300 { animation-delay: 300ms; }
        @keyframes scanline { 0% { transform: translateY(0); opacity: 0; } 10% { opacity: 1; } 90% { opacity: 1; } 100% { transform: translateY(200px); opacity: 0; } }
        .animate-scanner { animation: scanline 2.5s ease-in-out infinite; }
        @media print { .glass-card, .glass-input { background: white !important; backdrop-filter: none !important; box-shadow: none !important; border: 1px solid #e2e8f0 !important; } .animate-fade-in { opacity: 1 !important; animation: none !important; } }
      ` }} />

      <GlassBackground />

      {/* --- SCREEN 0: LANDING --- */}
      {step === 0 && (
        <main className="min-h-screen relative z-10 flex items-center justify-center p-4 md:p-8 font-sans">
          <div className="max-w-7xl w-full grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
            <div className="flex flex-col items-start text-left space-y-8 animate-fade-in z-10">
              <MadeInPecBadge />
              <h1 className="text-6xl md:text-8xl font-black text-slate-900 tracking-tighter leading-tight drop-shadow-sm">
                Smart<span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-500 to-blue-500">Sole</span><br />System.
              </h1>
              <p className="text-slate-600 text-xl font-medium max-w-md leading-relaxed border-l-4 border-teal-400 pl-4">
                Bilateral smart-insole telemetry and a machine-learning risk model for early Diabetic Foot Ulcer detection.
              </p>
              <div className="flex flex-wrap gap-3 text-xs font-bold text-slate-600">
                {["4-zone plantar pressure", "Thermal asymmetry ΔT", "In-shoe humidity", "Explainable ML"].map((t) => (
                  <span key={t} className="glass-card px-3 py-1.5 rounded-full">{t}</span>
                ))}
              </div>
              <div className="flex gap-4 w-full">
                <button onClick={() => setStep(1)} className="w-full sm:w-auto px-10 py-5 bg-slate-900 hover:bg-slate-800 text-white text-lg font-bold rounded-2xl shadow-[0_10px_40px_rgba(0,0,0,0.2)] transition-all transform hover:-translate-y-1">Initiate Assessment &rarr;</button>
              </div>
              <div className="flex items-center justify-between gap-4 pt-6 border-t border-slate-300/50 w-full mt-4">
                <div className="flex items-center gap-4">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src="https://images.unsplash.com/photo-1559839734-2b71ea197ec2?q=80&w=150&auto=format&fit=crop" className="w-12 h-12 rounded-full border-2 border-white shadow-md object-cover" alt="Dr. Jai Mala Gambhir" />
                  <div><p className="text-xs text-slate-500 uppercase tracking-widest font-bold">Module Group • Guided By</p><p className="text-sm font-black text-slate-900">Dr. Jai Mala Gambhir</p></div>
                </div>
                <Link href="/admin" className="text-xs font-bold text-slate-500 hover:text-teal-600 uppercase tracking-widest">Admin ↗</Link>
              </div>
            </div>
            <AnimatedHeroVisual />
          </div>
        </main>
      )}

      {/* --- SCREEN 1: PROFILE + DUAL UPLOAD --- */}
      {step === 1 && (
        <main className="min-h-screen relative z-10 flex items-center justify-center p-4 font-sans">
          <div className="w-full max-w-6xl glass-card p-8 md:p-12 rounded-[3rem] grid grid-cols-1 lg:grid-cols-3 gap-10 animate-fade-in">
            <div className="col-span-1 border-r-0 lg:border-r border-slate-300/30 pr-0 lg:pr-8 flex flex-col justify-between gap-6">
              <div>
                <h2 className="text-3xl font-black text-slate-900 tracking-tight mb-2">Patient Profile</h2>
                <p className="text-slate-600 text-sm font-medium mb-8">Demographics and clinical history personalise the model&apos;s risk estimate.</p>
                <div className="glass-input p-5 rounded-2xl mb-6 text-center">
                  <p className="text-[10px] uppercase font-bold tracking-widest text-slate-500 mb-1">Generated Session ID</p>
                  <p className="text-2xl font-mono font-black text-teal-600">{sessionID}</p>
                </div>
                <div className="rounded-2xl bg-slate-900 text-white p-5 shadow-xl">
                  <p className="text-[10px] uppercase font-bold tracking-widest text-slate-400 mb-3">Try a demo recording</p>
                  <div className="grid grid-cols-3 gap-2">
                    {[["healthy", "Healthy", "bg-teal-500"], ["moderate", "At risk", "bg-amber-500"], ["critical", "Critical", "bg-red-500"]].map(([k, l, c]) => (
                      <button key={k} type="button" onClick={() => loadDemo(k)} disabled={!!demoLoading}
                        className="px-2 py-2.5 rounded-xl bg-white/10 hover:bg-white/20 text-xs font-bold transition-all flex items-center justify-center gap-1.5 disabled:opacity-50">
                        <span className={`w-2 h-2 rounded-full ${c}`} />{demoLoading === k ? "…" : l}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
              <div className="hidden lg:block"><MadeInPecBadge className="" /></div>
            </div>

            <div className="col-span-2 space-y-7">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                <div><label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-2 ml-2">Age</label><input type="number" min="1" max="120" value={age} onChange={(e) => setAge(e.target.value)} className="w-full glass-input text-slate-900 p-4 rounded-2xl focus:ring-2 focus:ring-teal-400 outline-none font-bold text-lg transition-all" /></div>
                <div><label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-2 ml-2">Sex</label><select value={gender} onChange={(e) => setGender(e.target.value)} className="w-full glass-input text-slate-900 p-4 rounded-2xl focus:ring-2 focus:ring-teal-400 outline-none appearance-none font-bold text-lg transition-all"><option value="Male">Male</option><option value="Female">Female</option></select></div>
                <div><label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-2 ml-2">Weight (kg)</label><input type="number" min="20" max="250" value={weight} onChange={(e) => setWeight(e.target.value)} className="w-full glass-input text-slate-900 p-4 rounded-2xl focus:ring-2 focus:ring-teal-400 outline-none font-bold text-lg transition-all" /></div>
              </div>

              <div className="glass-input rounded-2xl">
                <button type="button" onClick={() => setShowClinical(!showClinical)} className="w-full flex items-center justify-between p-4 text-left">
                  <span className="text-xs font-bold text-slate-600 uppercase tracking-widest">Clinical history <span className="normal-case tracking-normal text-slate-400 font-semibold">— optional, improves accuracy</span></span>
                  <span className={`text-slate-500 transition-transform ${showClinical ? "rotate-180" : ""}`}>▾</span>
                </button>
                {showClinical && (
                  <div className="px-4 pb-5 space-y-4">
                    <div className="grid grid-cols-3 gap-3">
                      {[["height_cm", "Height (cm)"], ["diabetes_years", "Diabetes (yrs)"], ["hba1c", "HbA1c (%)"]].map(([k, l]) => (
                        <div key={k}>
                          <label className="block text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1 ml-1">{l}</label>
                          <input type="number" step="0.1" value={clinical[k] ?? ""} onChange={(e) => setClinical({ ...clinical, [k]: e.target.value })} className="w-full bg-white/80 border border-white text-slate-900 p-3 rounded-xl focus:ring-2 focus:ring-teal-400 outline-none font-bold" />
                        </div>
                      ))}
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {CLINICAL_TOGGLES.map(([k, l]) => (
                        <button key={k} type="button" onClick={() => setClinical({ ...clinical, [k]: !clinical[k] })}
                          className={`px-3 py-2 rounded-xl text-xs font-bold border transition-all ${clinical[k] ? "bg-teal-600 text-white border-teal-600" : "bg-white/70 text-slate-600 border-white hover:border-teal-300"}`}>
                          {clinical[k] ? "✓ " : ""}{l}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div>
                <label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-2 ml-2">Bilateral Telemetry Upload</label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <UploadTile side="L" name={left.name} info={left.info} onFile={handleFile(setLeft)} />
                  <UploadTile side="R" name={right.name} info={right.info} onFile={handleFile(setRight)} />
                </div>
                <p className="text-[11px] text-slate-500 font-medium mt-2 ml-2">Format: <code className="font-mono">Time, P1_Toe, P2_Met, P3_Heel, P4_Mid, Temp, Hum</code> — one file per foot, recorded in the same session.</p>
              </div>

              {error && <div className="p-4 rounded-2xl bg-red-50 border border-red-200 text-red-700 text-sm font-bold">{error}</div>}

              <div className="flex justify-end gap-4 pt-2">
                <button onClick={() => setStep(0)} className="px-8 py-4 text-slate-600 font-bold hover:bg-white/80 rounded-2xl transition-all">Cancel</button>
                <button disabled={!age || !weight || !left.csv || !right.csv} onClick={runAnalysis} className="px-10 py-4 bg-teal-600 hover:bg-teal-500 disabled:bg-slate-300 disabled:text-slate-400 text-white font-bold rounded-2xl shadow-xl transition-all">Execute Diagnostics &rarr;</button>
              </div>
            </div>
          </div>
        </main>
      )}

      {/* --- SCREEN 2: BIO-SCANNER --- */}
      {step === 2 && (
        <main className="min-h-screen relative z-10 flex flex-col items-center justify-center p-6 text-center font-sans animate-fade-in">
          <MadeInPecBadge />
          <div className="glass-card p-12 rounded-[3rem] w-full max-w-lg flex flex-col items-center relative overflow-hidden mt-4 shadow-2xl">
            <div className="absolute inset-0 bg-gradient-to-t from-teal-400/20 to-transparent animate-[pulse_2s_infinite]"></div>
            <div className="relative w-32 h-40 mb-8 z-10">
              <svg viewBox="0 0 100 240" className="w-full h-full opacity-40 drop-shadow-xl"><path d="M30,10 C10,30 0,70 10,110 C15,140 20,180 20,200 C20,220 35,235 50,235 C65,235 80,220 80,200 C80,180 85,140 90,110 C100,70 90,30 70,10 C60,0 40,0 30,10 Z" fill="#334155" /></svg>
              <div className="absolute top-0 left-[-10%] w-[120%] h-1 bg-teal-400 shadow-[0_0_20px_#2dd4bf] animate-scanner"></div>
            </div>
            <h2 className="text-3xl font-black text-slate-900 mb-2 relative z-10 tracking-tight">Processing Telemetry</h2>
            <p className="text-slate-600 font-bold mb-10 h-6 relative z-10">{scanMessage}</p>
            <div className="w-full bg-white/70 border border-white p-1 rounded-full shadow-inner relative z-10">
              <div className="h-3 bg-teal-500 rounded-full transition-all duration-200" style={{ width: `${progress}%` }}></div>
            </div>
          </div>
        </main>
      )}

      {/* --- SCREEN 3: DASHBOARD --- */}
      {step === 3 && result && (
        <main className="min-h-screen relative z-10 text-slate-800 p-4 md:p-8 font-sans pb-20 animate-fade-in">
          <div className="max-w-[1400px] mx-auto">
            <div className="glass-card p-6 rounded-[2rem] flex flex-col md:flex-row justify-between items-center gap-4 mb-8 shadow-md">
              <div className="flex items-center gap-5">
                <div className="hidden sm:block"><MadeInPecBadge className="" /></div>
                <div>
                  <h1 className="text-3xl font-black text-slate-900 tracking-tight">Clinical Assessment</h1>
                  <p className="text-sm font-bold text-slate-600">Subject: {sessionID} • {result.gender}, {result.age} y • {result.weight} kg • {currentDate}</p>
                </div>
              </div>
              <div className="flex gap-3 print:hidden">
                <button onClick={() => window.print()} className="px-6 py-3.5 bg-white/70 hover:bg-white text-slate-700 font-bold rounded-2xl shadow-sm transition-all border border-slate-200">Print report</button>
                <button onClick={reset} className="px-6 py-3.5 bg-slate-900 hover:bg-slate-800 text-white font-bold rounded-2xl shadow-sm transition-all">Close Session</button>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              <div className={`lg:col-span-12 glass-card rounded-[3rem] p-8 md:p-10 flex flex-col md:flex-row items-center justify-between gap-6 overflow-hidden relative border-l-8 shadow-xl`} style={{ borderLeftColor: overallStyle.ring }}>
                <div className="relative z-10">
                  <div className="flex items-center mb-2">
                    <p className="text-sm font-bold uppercase tracking-widest text-slate-500">Bilateral Diagnosis</p>
                    <InfoTooltip title="Bilateral Diagnosis" desc="Overall status is driven by the higher-risk foot as predicted by the ML model." optimal="Healthy Distribution" danger="DFU WARNING or CRITICAL ALERT" />
                  </div>
                  <h2 className={`text-5xl md:text-6xl font-black tracking-tighter ${overallStyle.text}`}>{result.overallStatus}</h2>
                  <div className="flex flex-wrap items-center gap-2 mt-4">
                    <span className="px-3 py-1.5 rounded-full bg-white/80 border border-white text-[11px] font-bold text-slate-600">
                      🧠 {result.model?.algorithm?.replaceAll("_", " ")} · {result.model?.version}
                    </span>
                    {result.model?.metrics?.macro_f1 != null && (
                      <span className="px-3 py-1.5 rounded-full bg-white/80 border border-white text-[11px] font-bold text-slate-600">CV macro-F1 {(result.model.metrics.macro_f1 * 100).toFixed(0)}%</span>
                    )}
                    {result.model?.isBootstrap && (
                      <span className="px-3 py-1.5 rounded-full bg-amber-100 border border-amber-200 text-[11px] font-bold text-amber-800">Starter model · trained on simulated data</span>
                    )}
                  </div>
                </div>
                <div className="text-center md:text-right relative z-10">
                  <div className="flex items-center justify-center md:justify-end mb-1">
                    <p className="text-sm font-bold uppercase tracking-widest text-slate-500">Thermal Variance (ΔT)</p>
                    <InfoTooltip title="Thermal Variance (ΔT)" desc="Difference in mean temperature between the left and right foot — the strongest early indicator of inflammation." optimal="0.0 °C to 1.0 °C" danger="> 2.2 °C" />
                  </div>
                  <p className={`text-5xl font-black drop-shadow-sm ${parseFloat(result.abs_deltaT) > 2.2 ? "text-red-500" : "text-orange-500"}`}>{result.abs_deltaT}°C</p>
                </div>
              </div>

              {(result.clinicalFlags?.length > 0 || result.warnings?.length > 0) && (
                <div className="lg:col-span-12 grid grid-cols-1 md:grid-cols-2 gap-4">
                  {result.clinicalFlags?.map((f, i) => (
                    <div key={`f${i}`} className={`glass-card rounded-2xl p-4 flex gap-3 items-start border-l-4 ${f.severity === "high" ? "border-l-red-500" : f.severity === "moderate" ? "border-l-amber-500" : "border-l-sky-400"}`}>
                      <span className="text-lg">{f.severity === "high" ? "🔴" : f.severity === "moderate" ? "🟠" : "🔵"}</span>
                      <p className="text-sm font-semibold text-slate-700">{f.message}</p>
                    </div>
                  ))}
                  {result.warnings?.map((w, i) => (
                    <div key={`w${i}`} className="glass-card rounded-2xl p-4 flex gap-3 items-start border-l-4 border-l-slate-400">
                      <span className="text-lg">⚠️</span>
                      <p className="text-sm font-semibold text-slate-600">{w}</p>
                    </div>
                  ))}
                </div>
              )}

              <FootCard title="Left Foot" side="left" data={result.left} isLeft />
              <FootCard title="Right Foot" side="right" data={result.right} isLeft={false} />

              <p className="lg:col-span-12 text-center text-xs text-slate-500 font-semibold max-w-3xl mx-auto">
                SmartSole is a screening aid, not a diagnosis. Results must be interpreted by a qualified clinician together with a physical foot examination.
              </p>
            </div>
          </div>
        </main>
      )}
    </>
  );
}
