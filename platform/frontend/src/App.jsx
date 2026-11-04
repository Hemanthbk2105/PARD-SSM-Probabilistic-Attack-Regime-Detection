import { useEffect, useMemo, useState } from "react";
import {
  Activity, AlertTriangle, ArrowDownRight, ArrowUpRight, BarChart3, Bell,
  BookOpen, BrainCircuit, ChevronDown, ChevronRight, CircleHelp, Clock3,
  Command, Database, Download, FileBarChart2, Fingerprint, Gauge, Github,
  LayoutDashboard, Menu, Network, Play, Radio, RefreshCw, Search, Settings2,
  Shield, ShieldAlert, ShieldCheck, SlidersHorizontal, Sparkles, UploadCloud,
  X, Zap
} from "lucide-react";
import {
  Area, AreaChart, CartesianGrid, Cell, Line, LineChart, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis, BarChart, Bar
} from "recharts";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const REGIME = [
  { name: "Normal", color: "#47d7ac" },
  { name: "Reconnaissance", color: "#f5bd55" },
  { name: "Intrusion", color: "#ff8a65" },
  { name: "DoS / DDoS", color: "#ff5f72" },
  { name: "Exfiltration", color: "#a78bfa" }
];
const NAV = [
  { group: "MONITOR", items: [
    { id: "overview", label: "Command center", icon: LayoutDashboard },
    { id: "threats", label: "Threat intelligence", icon: ShieldAlert },
    { id: "regimes", label: "Regime explorer", icon: Activity }
  ]},
  { group: "DATA & MODELS", items: [
    { id: "datasets", label: "Dataset management", icon: Database },
    { id: "models", label: "Model configuration", icon: BrainCircuit },
    { id: "network", label: "Network explorer", icon: Network },
    { id: "reports", label: "Security reports", icon: FileBarChart2 }
  ]}
];

function cx(...items) { return items.filter(Boolean).join(" "); }
function fmt(value) { return Number(value || 0).toLocaleString("en-IN"); }
function pct(value) { return `${(Number(value || 0) * 100).toFixed(1)}%`; }
function shortDate(value) {
  if (!value) return "Not run";
  return new Date(value).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function App() {
  const [page, setPage] = useState("overview");
  const [mobileNav, setMobileNav] = useState(false);
  const [dataset, setDataset] = useState("cicids2017");
  const [datasets, setDatasets] = useState([]);
  const [result, setResult] = useState(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [busy, setBusy] = useState("");
  const [toast, setToast] = useState("");
  const [files, setFiles] = useState([]);
  const [sampleSize, setSampleSize] = useState(50000);
  const [config, setConfig] = useState({ regimes: 4, state_dim: 8, obs_dim: 10, em_iterations: 8 });
  const [search, setSearch] = useState("");

  const activeDataset = datasets.find(d => d.id === dataset);
  const hasResult = Boolean(result);
  const pageTitle = NAV.flatMap(g => g.items).find(i => i.id === page)?.label || "Command center";

  async function request(path, options = {}) {
    const response = await fetch(`${API}${path}`, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "Request failed");
    return data;
  }

  async function refresh() {
    try {
      await request("/api/health");
      setApiOnline(true);
      const data = await request("/api/datasets");
      setDatasets(data.datasets || []);
      const selected = data.datasets?.find(d => d.id === dataset);
      if (selected?.analysed) {
        const latest = await request(`/api/results/${dataset}`);
        setResult(latest);
      } else setResult(null);
    } catch {
      setApiOnline(false);
      setDatasets([]);
      setResult(null);
    }
  }

  useEffect(() => { refresh(); }, [dataset]);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(""), 4200);
    return () => clearTimeout(timer);
  }, [toast]);

  async function upload() {
    if (!files.length) return setToast("Choose one or more CSV files first.");
    const form = new FormData();
    files.forEach(file => form.append("files", file));
    setBusy("upload");
    try {
      const data = await request(`/api/upload/${dataset}`, { method: "POST", body: form });
      setToast(data.message);
      setFiles([]);
      await refresh();
    } catch (error) { setToast(error.message); }
    finally { setBusy(""); }
  }

  async function prepare() {
    setBusy("prepare");
    try {
      const data = await request(`/api/prepare/${dataset}?sample_size=${sampleSize}`, { method: "POST" });
      setToast(`Prepared ${fmt(data.samples)} rows with ${data.features} features.`);
      await refresh();
    } catch (error) { setToast(error.message); }
    finally { setBusy(""); }
  }

  async function analyse() {
    setBusy("analyse");
    try {
      const params = new URLSearchParams(config);
      const data = await request(`/api/analyse/${dataset}?${params}`, { method: "POST" });
      setResult(data.result);
      setToast("Experiment completed. Results are now available.");
      await refresh();
    } catch (error) { setToast(error.message); }
    finally { setBusy(""); }
  }

  async function loadResults() {
    try { setResult(await request(`/api/results/${dataset}`)); }
    catch { setResult(null); }
  }

  useEffect(() => { loadResults(); }, [dataset]);

  const chartData = useMemo(() => {
    if (!result) return [];
    return (result.regime_uncertainty || []).map((v, i) => ({
      step: i + 1,
      uncertainty: Number(v),
      regime: result.predicted_regimes?.[i] ?? 0,
      truth: result.true_regimes?.[i] ?? 0
    }));
  }, [result]);

  const distribution = useMemo(() => {
    if (!result) return [];
    return Object.entries(result.ground_truth_distribution || {}).map(([id, count]) => ({
      name: `Dataset class ${id}`,
      value: count,
      color: REGIME[Number(id)]?.color || "#8092a8"
    }));
  }, [result]);

  function go(id) { setPage(id); setMobileNav(false); }

  return (
    <div className="app-shell">
      <aside className={cx("sidebar", mobileNav && "sidebar-open")}>
        <div className="brand">
          <div className="brand-mark"><Fingerprint size={21}/></div>
          <div><div className="brand-name">PARD<span> / </span>SSM</div><div className="brand-sub">INTELLIGENCE PLATFORM</div></div>
          <button className="icon-btn close-nav" onClick={() => setMobileNav(false)}><X size={17}/></button>
        </div>
        <div className="workspace-switch">
          <div className="workspace-icon"><Shield size={16}/></div>
          <div className="workspace-copy"><strong>Security workspace</strong><span>Private environment</span></div>
          <ChevronDown size={14} className="muted"/>
        </div>
        <nav className="nav">
          {NAV.map(group => <div className="nav-group" key={group.group}>
            <div className="nav-caption">{group.group}</div>
            {group.items.map(item => {
              const Icon = item.icon;
              return <button key={item.id} className={cx("nav-item", page === item.id && "active")} onClick={() => go(item.id)}>
                <Icon size={17}/><span>{item.label}</span>{page === item.id && <span className="nav-active-dot"/>}
              </button>;
            })}
          </div>)}
        </nav>
        <div className="sidebar-bottom">
          <div className="help-card"><div className="help-icon"><BookOpen size={16}/></div><div><strong>Platform guide</strong><span>Understand the model</span></div><ChevronRight size={14}/></div>
          <div className="profile-row"><div className="avatar">PS</div><div className="profile-copy"><strong>PARD Security</strong><span>Local instance</span></div><Settings2 size={16} className="muted"/></div>
        </div>
      </aside>
      {mobileNav && <button className="scrim" onClick={() => setMobileNav(false)} aria-label="Close navigation"/>}
      <main className="main">
        <header className="topbar">
          <div className="top-left"><button className="icon-btn mobile-menu" onClick={() => setMobileNav(true)}><Menu size={19}/></button><div className="breadcrumb">Workspace <ChevronRight size={13}/> <strong>{pageTitle}</strong></div></div>
          <div className="top-actions">
            <div className={cx("system-status", apiOnline ? "online" : "offline")}><span className="pulse-dot"/>{apiOnline ? "API connected" : "API offline"}</div>
            <div className="top-divider"/>
            <button className="icon-btn" title="Refresh data" onClick={refresh}><RefreshCw size={16}/></button>
            <button className="icon-btn notification-btn" title="Notifications"><Bell size={16}/><i/></button>
            <div className="top-avatar">H</div>
          </div>
        </header>

        <div className="page-wrap">
          <div className="page-heading">
            <div><div className="eyebrow"><span className="eyebrow-line"/>{page === "overview" ? "NETWORK DEFENCE / OVERVIEW" : `PARD-SSM / ${pageTitle.toUpperCase()}`}</div><h1>{page === "overview" ? "Command center" : pageTitle}<span className="heading-period">.</span></h1><p>{page === "overview" ? "A probabilistic view of network behaviour, attack regimes and model confidence." : pageDescriptions[page]}</p></div>
            <div className="heading-controls">
              <div className="select-wrap"><Database size={14}/><select value={dataset} onChange={e => setDataset(e.target.value)}><option value="cicids2017">CICIDS2017</option><option value="unswnb15">UNSW-NB15</option></select><ChevronDown size={13}/></div>
              <button className="primary-btn" onClick={() => go("datasets")}><UploadCloud size={15}/> Add dataset</button>
            </div>
          </div>

          {!apiOnline && <div className="notice notice-warn"><AlertTriangle size={17}/><div><strong>Backend not connected</strong><span>Start the FastAPI service to upload datasets and run real experiments. The interface will not invent or display sample metrics as live results.</span></div><code>uvicorn main:app --reload</code></div>}

          {page === "overview" && <Overview result={result} dataset={dataset} activeDataset={activeDataset} chartData={chartData} distribution={distribution} go={go} apiOnline={apiOnline} />}
          {page === "threats" && <Threats result={result} chartData={chartData} go={go}/>}
          {page === "regimes" && <Regimes result={result} distribution={distribution} chartData={chartData}/>}
          {page === "datasets" && <Datasets dataset={dataset} datasets={datasets} activeDataset={activeDataset} files={files} setFiles={setFiles} upload={upload} prepare={prepare} busy={busy} sampleSize={sampleSize} setSampleSize={setSampleSize} apiOnline={apiOnline}/>}
          {page === "models" && <Models result={result} config={config} setConfig={setConfig} analyse={analyse} busy={busy} activeDataset={activeDataset} apiOnline={apiOnline} go={go}/>}
          {page === "network" && <NetworkExplorer result={result} search={search} setSearch={setSearch}/>}
          {page === "reports" && <Reports result={result} dataset={dataset} apiOnline={apiOnline}/>}
          <footer className="footer"><span><span className="footer-mark">◈</span> PARD-SSM Intelligence <span className="footer-sep">/</span> Security operations</span><span>Probabilistic Attack Regime Detection <span className="footer-sep">·</span> v1.0.0</span></footer>
        </div>
      </main>
      {toast && <div className="toast"><span className="toast-dot"/>{toast}<button onClick={() => setToast("")}><X size={14}/></button></div>}
    </div>
  );
}

const pageDescriptions = {
  threats: "Inspect attack probability signals and the moments where network behaviour shifts.",
  regimes: "Explore the latent states inferred by the Switching State-Space Model.",
  datasets: "Bring your CICIDS2017 or UNSW-NB15 traffic data into the detection workflow.",
  models: "Configure an experiment and evaluate the probabilistic state-space model.",
  network: "Explore model outputs at the observation level without treating flow rows as live packets.",
  reports: "Review and export the latest reproducible experiment output."
};

function EmptyState({ icon: Icon = Activity, title, text, action, onAction }) {
  return <div className="empty-state"><div className="empty-icon"><Icon size={22}/></div><h3>{title}</h3><p>{text}</p>{action && <button className="secondary-btn" onClick={onAction}>{action}<ChevronRight size={14}/></button>}</div>;
}

function MetricCard({ label, value, suffix, detail, icon: Icon, tone = "mint", trend }) {
  return <div className="metric-card"><div className="metric-top"><span>{label}</span><span className={cx("metric-icon", tone)}><Icon size={16}/></span></div><div className="metric-value">{value ?? "—"}<small>{suffix}</small></div><div className="metric-foot">{trend && <span className="metric-trend"><ArrowUpRight size={13}/>{trend}</span>}<span>{detail}</span></div></div>;
}

function Panel({ title, subtitle, action, children, className = "" }) {
  return <section className={cx("panel", className)}><div className="panel-head"><div><h3>{title}</h3>{subtitle && <p>{subtitle}</p>}</div>{action}</div>{children}</section>;
}

function Overview({ result, dataset, activeDataset, chartData, distribution, go, apiOnline }) {
  return <>
    <div className="status-strip"><div className="status-main"><span className={cx("status-indicator", apiOnline ? "good" : "idle")}/><strong>{result ? "Experiment results loaded" : "Awaiting first experiment"}</strong><span>{result ? `Last run ${shortDate(result.created_at)}` : "Upload a dataset to initialise the detection workflow"}</span></div><div className="status-meta"><span className="mono">DATA SOURCE</span><strong>{dataset.toUpperCase()}</strong></div><div className="status-meta"><span className="mono">MODEL</span><strong>SWITCHING SSM</strong></div></div>
    <div className="metric-grid">
      <MetricCard label="Samples analysed" value={result ? fmt(result.samples) : "—"} detail={result ? `${fmt(result.evaluated_steps)} evaluation steps` : "No experiment data"} icon={Database} tone="blue"/>
      <MetricCard label="Adjusted Rand index" value={result ? Number(result.adjusted_rand_index).toFixed(3) : "—"} detail="Permutation-invariant cluster agreement" icon={Gauge} tone="mint"/>
      <MetricCard label="Prediction error" value={result ? Number(result.prediction_mse).toFixed(4) : "—"} detail="Mean squared observation error" icon={Activity} tone="amber"/>
      <MetricCard label="Log likelihood" value={result ? Number(result.log_likelihood).toFixed(1) : "—"} detail="Mean test-window likelihood" icon={BrainCircuit} tone="violet"/>
    </div>
    <div className="content-grid overview-grid">
      <Panel title="Regime uncertainty" subtitle="One minus the highest filtered latent-state probability" action={<span className="panel-tag"><span className="live-dot"/> MODEL OUTPUT</span>}>
        {result ? <div className="chart-area"><div className="chart-legend"><span><i className="legend-mint"/>Regime uncertainty</span><span className="legend-note">Evaluation sequence · sampled points</span></div><ResponsiveContainer width="100%" height={250}><AreaChart data={chartData}><defs><linearGradient id="attackFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ff6e75" stopOpacity={0.28}/><stop offset="100%" stopColor="#ff6e75" stopOpacity={0.01}/></linearGradient></defs><CartesianGrid stroke="#202c39" strokeDasharray="3 5" vertical={false}/><XAxis dataKey="step" tick={{fill:"#718196",fontSize:11}} axisLine={false} tickLine={false}/><YAxis domain={[0,1]} tick={{fill:"#718196",fontSize:11}} axisLine={false} tickLine={false} tickFormatter={v => `${Math.round(v*100)}%`}/><Tooltip contentStyle={{background:"#111c27",border:"1px solid #2a3948",borderRadius:10,color:"#e9f0f7"}} formatter={v => [pct(v),"Regime uncertainty"]}/><Area type="monotone" dataKey="uncertainty" stroke="#ff7880" strokeWidth={2} fill="url(#attackFill)"/></AreaChart></ResponsiveContainer></div> : <EmptyState icon={Activity} title="No signal to plot yet" text="Run your first experiment and the regime-uncertainty curve will appear here." action="Open dataset management" onAction={() => go("datasets")}/>}
      </Panel>
      <Panel title="Regime distribution" subtitle="Ground-truth labels in the evaluated segment" action={<button className="text-action" onClick={() => go("regimes")}>Explore <ChevronRight size={13}/></button>}>
        {result ? <div className="distribution-wrap"><div className="donut-wrap"><ResponsiveContainer width="100%" height={190}><PieChart><Pie data={distribution} dataKey="value" nameKey="name" innerRadius={58} outerRadius={80} paddingAngle={3} stroke="none">{distribution.map((entry,i) => <Cell key={i} fill={entry.color}/>)}</Pie><Tooltip contentStyle={{background:"#111c27",border:"1px solid #2a3948",borderRadius:10}}/></PieChart></ResponsiveContainer><div className="donut-center"><strong>{fmt(result.evaluated_steps)}</strong><span>OBSERVATIONS</span></div></div><div className="regime-legend">{distribution.map((item,i) => <div key={item.name}><span><i style={{background:item.color}}/>{item.name}</span><strong>{fmt(item.value)}</strong></div>)}</div></div> : <EmptyState icon={Fingerprint} title="Regimes not inferred" text="Regime composition will be calculated from the held-out experiment sequence."/>}
      </Panel>
    </div>
    <div className="content-grid lower-grid">
      <Panel title="Detection workflow" subtitle="From raw traffic to probabilistic regime inference">
        <div className="pipeline">{[{n:"01",t:"Dataset ingestion",s:"CSV traffic records",ic:Database},{n:"02",t:"Feature preparation",s:"Cleaning · scaling · PCA",ic:SlidersHorizontal},{n:"03",t:"Switching SSM",s:"Latent state inference",ic:BrainCircuit},{n:"04",t:"Evaluation",s:"Metrics · regime outputs",ic:BarChart3}].map((item,i)=>{const Icon=item.ic;return <div className="pipeline-step" key={item.n}><div className="pipeline-icon"><Icon size={16}/></div><div><span>{item.n} / STAGE</span><strong>{item.t}</strong><small>{item.s}</small></div>{i<3&&<ChevronRight className="pipeline-arrow" size={15}/>}</div>})}</div>
      </Panel>
      <Panel title="Workspace readiness" subtitle="Current data and service status">
        <div className="readiness-list"><div><span><i className={cx("readiness-dot",apiOnline&&"ready")}/>API service</span><strong>{apiOnline?"Connected":"Offline"}</strong></div><div><span><i className={cx("readiness-dot",activeDataset?.file_count>0&&"ready")}/>Raw CSV files</span><strong>{activeDataset?.file_count||0} files</strong></div><div><span><i className={cx("readiness-dot",activeDataset?.processed&&"ready")}/>Processed arrays</span><strong>{activeDataset?.processed?"Ready":"Pending"}</strong></div><div><span><i className={cx("readiness-dot",result&&"ready")}/>Latest experiment</span><strong>{result?"Available":"Not run"}</strong></div></div>
        <button className="wide-link" onClick={() => go("datasets")}>Manage datasets <ChevronRight size={14}/></button>
      </Panel>
    </div>
    {result && <div className="method-note"><AlertTriangle size={15}/><span>{result.evaluation_note}</span></div>}
  </>;
}

function Threats({ result, chartData, go }) {
  if (!result) return <EmptyState icon={ShieldAlert} title="Threat intelligence is waiting for data" text="The threat timeline is built from the SSSM posterior output. Upload your traffic dataset and run an experiment to populate it." action="Manage datasets" onAction={() => go("datasets")}/>;
  return <><div className="metric-grid"><MetricCard label="Evaluated observations" value={fmt(result.evaluated_steps)} detail="Held-out sequence" icon={Activity}/><MetricCard label="Peak regime uncertainty" value={pct(Math.max(...(result.regime_uncertainty||[0])))} detail="Maximum posterior uncertainty" icon={ShieldAlert} tone="rose"/><MetricCard label="Predicted regime changes" value={fmt((result.predicted_regimes||[]).filter((v,i,a)=>i>0&&v!==a[i-1]).length)} detail="Changes in sampled output" icon={Zap} tone="amber"/><MetricCard label="Latest run" value={shortDate(result.created_at).split(",")[0]} detail="Local experiment record" icon={Clock3} tone="blue"/></div><Panel title="Regime uncertainty timeline" subtitle="Posterior uncertainty across the evaluation sequence"><div className="chart-area tall"><ResponsiveContainer width="100%" height={340}><AreaChart data={chartData}><defs><linearGradient id="threatFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ff6874" stopOpacity={0.3}/><stop offset="100%" stopColor="#ff6874" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="#202c39" strokeDasharray="3 5" vertical={false}/><XAxis dataKey="step" tick={{fill:"#718196",fontSize:11}} axisLine={false}/><YAxis domain={[0,1]} tick={{fill:"#718196",fontSize:11}} axisLine={false} tickFormatter={v=>pct(v)}/><Tooltip contentStyle={{background:"#111c27",border:"1px solid #2a3948",borderRadius:10}} formatter={v=>[pct(v),"Regime uncertainty"]}/><Area dataKey="uncertainty" stroke="#ff6f7b" strokeWidth={2} fill="url(#threatFill)"/></AreaChart></ResponsiveContainer></div></Panel><div className="method-note"><AlertTriangle size={15}/><span>This chart shows uncertainty in the inferred latent regime. It is not an attack score, calibrated alert, or proof of a specific attack.</span></div></>;
}

function Regimes({ result, distribution, chartData }) {
  if (!result) return <EmptyState icon={Fingerprint} title="No latent regimes available" text="Run the Switching SSM experiment to inspect inferred regime probabilities and decoded state transitions."/>;
  const probs = result.regime_probabilities || [];
  const series = probs.map((row,i)=>({step:i+1,...Object.fromEntries(row.map((v,k)=>[`r${k}`,v]))}));
  return <><div className="regime-cards">{distribution.map((r,i)=><div className="regime-card" key={r.name}><span className="regime-swatch" style={{background:r.color}}/><span className="regime-index">STATE {String(i).padStart(2,"0")}</span><strong>{r.name}</strong><b>{fmt(r.value)}</b><small>Ground-truth observations</small></div>)}</div><Panel title="Posterior regime probabilities" subtitle="P(state at time t | observations up to t)"><ResponsiveContainer width="100%" height={350}><AreaChart data={series}><CartesianGrid stroke="#202c39" strokeDasharray="3 5" vertical={false}/><XAxis dataKey="step" tick={{fill:"#718196",fontSize:11}} axisLine={false}/><YAxis domain={[0,1]} tick={{fill:"#718196",fontSize:11}} axisLine={false}/><Tooltip contentStyle={{background:"#111c27",border:"1px solid #2a3948",borderRadius:10}} formatter={v=>[pct(v)]}/>{(probs[0]||[]).map((_,i)=><Area key={i} type="monotone" dataKey={`r${i}`} name={REGIME[i]?.name||`Regime ${i}`} stackId="1" stroke={REGIME[i]?.color||"#9aa"} fill={REGIME[i]?.color||"#9aa"} fillOpacity={0.42}/>)}</AreaChart></ResponsiveContainer></Panel><Panel title="Decoded sequence" subtitle="Most likely latent state at each sampled time step"><div className="sequence-strip">{(result.predicted_regimes||[]).map((v,i)=><div key={i} title={`Step ${i+1}: latent state ${v}`} style={{background:REGIME[v]?.color||"#718196"}}/>)}</div><div className="sequence-caption"><span>EARLIER OBSERVATIONS</span><span>LATER OBSERVATIONS</span></div></Panel><div className="method-note"><AlertTriangle size={15}/><span>Latent state IDs are learned states. They are not automatically equivalent to the semantic labels in the dataset. Interpret state names only after validating the model-to-label mapping.</span></div></>;
}

function Datasets({ dataset, datasets, activeDataset, files, setFiles, upload, prepare, busy, sampleSize, setSampleSize, apiOnline }) {
  return <><div className="dataset-intro"><div className="dataset-intro-icon"><Database size={24}/></div><div><h2>Connect your network traffic data.</h2><p>Upload the original CSV files. PARD-SSM will clean, encode and prepare the selected dataset using the repository's preprocessing pipeline.</p></div></div><div className="dataset-grid"><div className="dataset-choice"><div className="choice-top"><span className="choice-icon cicids">C</span><span className="choice-label">SUPPORTED DATA SOURCE / 01</span></div><h3>CICIDS2017</h3><p>Network flows covering benign activity and multiple attack families, including scanning, brute force and denial-of-service.</p><div className="choice-meta"><span>Canadian Institute for Cybersecurity</span><span>CSV collection</span></div><a href="https://www.unb.ca/cic/datasets/ids-2017.html" target="_blank" rel="noreferrer">Dataset source <ChevronRight size={13}/></a><div className="choice-state">{datasets.find(d=>d.id==="cicids2017")?.processed?<><ShieldCheck size={14}/> Prepared</>:<><Clock3 size={14}/> Awaiting upload</>}</div></div><div className="dataset-choice"><div className="choice-top"><span className="choice-icon unsw">U</span><span className="choice-label">SUPPORTED DATA SOURCE / 02</span></div><h3>UNSW-NB15</h3><p>Modern network traffic with normal activity and labelled attack categories for security analytics and intrusion detection evaluation.</p><div className="choice-meta"><span>UNSW Canberra Cyber</span><span>CSV collection</span></div><a href="https://research.unsw.edu.au/projects/unsw-nb15-dataset" target="_blank" rel="noreferrer">Dataset source <ChevronRight size={13}/></a><div className="choice-state">{datasets.find(d=>d.id==="unswnb15")?.processed?<><ShieldCheck size={14}/> Prepared</>:<><Clock3 size={14}/> Awaiting upload</>}</div></div></div><Panel title="Dataset ingestion" subtitle={`Selected source: ${dataset.toUpperCase()}`}><div className="upload-zone"><input id="csv-upload" type="file" accept=".csv,text/csv" multiple onChange={e=>setFiles(Array.from(e.target.files||[]))}/><label htmlFor="csv-upload"><div className="upload-icon"><UploadCloud size={24}/></div><strong>Drop CSV files here or browse</strong><span>Multiple files supported · CSV format only</span><small>For CICIDS2017, select the extracted MachineLearningCSV files. For UNSW-NB15, select the four dataset CSV files.</small></label></div>{files.length>0&&<div className="selected-files">{files.map(f=><span key={f.name}><Database size={13}/>{f.name}<small>{(f.size/1024/1024).toFixed(1)} MB</small></span>)}</div>}<div className="upload-actions"><button className="primary-btn" disabled={!apiOnline||busy==="upload"} onClick={upload}>{busy==="upload"?"Uploading…": "Upload selected files"} <UploadCloud size={15}/></button><div className="sample-control"><label>Preparation sample limit</label><select value={sampleSize} onChange={e=>setSampleSize(Number(e.target.value))}><option value="10000">10,000 rows · quick test</option><option value="50000">50,000 rows · recommended</option><option value="100000">100,000 rows · larger run</option><option value="0">All available rows</option></select></div><button className="secondary-btn" disabled={!apiOnline||busy==="prepare"||!activeDataset?.file_count} onClick={prepare}>{busy==="prepare"?"Preparing…":"Prepare dataset"} <Zap size={14}/></button></div></Panel><Panel title="Ingestion status" subtitle="Files stored locally in the platform data directory"><div className="file-table"><div className="file-table-head"><span>DATASET</span><span>FILES</span><span>ROWS</span><span>STATUS</span></div>{datasets.map(d=><div className="file-table-row" key={d.id}><strong>{d.name}</strong><span>{d.file_count}</span><span>{d.processed?fmt(d.samples):"—"}</span><span className={d.processed?"table-ready":"table-pending"}>{d.processed?"Prepared":"Pending"}</span></div>)}</div></Panel><div className="method-note"><AlertTriangle size={15}/><span>Evaluation limitation: the repository's CICIDS2017 and UNSW-NB15 loaders randomly sample rows before the train/test split. This platform surfaces that limitation and does not label the resulting split as chronological.</span></div></>;
}

function Models({ result, config, setConfig, analyse, busy, activeDataset, apiOnline, go }) {
  const change = (key,value)=>setConfig(prev=>({...prev,[key]:Number(value)}));
  return <><div className="model-banner"><div className="model-banner-icon"><BrainCircuit size={27}/></div><div><span className="eyebrow">CORE INFERENCE ENGINE</span><h2>Switching State-Space Model</h2><p>Probabilistic latent-state estimation using regime-conditioned dynamics and sequential observations.</p></div><span className="model-version">PARD / SSSM</span></div><div className="content-grid model-grid"><Panel title="Experiment configuration" subtitle="Tune the inference pipeline before execution"><div className="config-list"><ConfigControl label="Number of latent regimes" value={config.regimes} min={2} max={5} onChange={v=>change("regimes",v)} hint="Latent states inferred by the switching model."/><ConfigControl label="Hidden state dimension" value={config.state_dim} min={2} max={20} onChange={v=>change("state_dim",v)} hint="Size of the continuous hidden state vector."/><ConfigControl label="PCA observation dimensions" value={config.obs_dim} min={2} max={20} onChange={v=>change("obs_dim",v)} hint="Number of principal components passed to the SSSM."/><ConfigControl label="EM iterations" value={config.em_iterations} min={1} max={30} onChange={v=>change("em_iterations",v)} hint="Expectation-maximisation training iterations."/><button className="primary-btn run-btn" disabled={!apiOnline||!activeDataset?.processed||Boolean(busy)} onClick={analyse}>{busy==="analyse"?"Running experiment…": "Run PARD-SSM experiment"} <Play size={15}/></button>{!activeDataset?.processed&&<button className="text-action" onClick={()=>go("datasets")}>Open dataset management first <ChevronRight size={13}/></button>}</div></Panel><Panel title="Experiment lifecycle" subtitle="Execution stages and output artefacts"><div className="lifecycle">{[{t:"Load prepared observations",s:"Features and ground-truth labels",ic:Database},{t:"Temporal holdout split",s:"80% train / 20% test",ic:SlidersHorizontal},{t:"Fit latent dynamics",s:"EM iterations on up to 5,000 training observations",ic:BrainCircuit},{t:"Filter held-out sequences",s:"Posterior probabilities and Viterbi path",ic:Activity},{t:"Evaluate and persist",s:"Cluster metrics and JSON report",ic:FileBarChart2}].map((v,i)=>{const Icon=v.ic;return <div className="lifecycle-row" key={v.t}><div className="lifecycle-number">{String(i+1).padStart(2,"0")}</div><div className="lifecycle-icon"><Icon size={15}/></div><div><strong>{v.t}</strong><span>{v.s}</span></div><span className="lifecycle-state">{result?"AVAILABLE":"READY"}</span></div>})}</div></Panel></div>{result&&<Panel title="Latest experiment" subtitle={`Completed ${shortDate(result.created_at)}`}><div className="result-summary"><div><span>ADJUSTED RAND INDEX</span><strong>{Number(result.adjusted_rand_index).toFixed(3)}</strong></div><div><span>NORMALISED MUTUAL INFO</span><strong>{Number(result.normalized_mutual_info).toFixed(3)}</strong></div><div><span>PREDICTION MSE</span><strong>{Number(result.prediction_mse).toFixed(4)}</strong></div><div><span>LOG LIKELIHOOD</span><strong>{Number(result.log_likelihood).toFixed(2)}</strong></div><div><span>TEST STEPS</span><strong>{fmt(result.evaluated_steps)}</strong></div></div></Panel>}</>;
}

function ConfigControl({ label, value, min, max, step=1, onChange, hint }) {
  return <div className="config-control"><div><label>{label}</label><span>{hint}</span></div><div className="config-input"><input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(e.target.value)}/><strong>{value}</strong></div></div>;
}

function NetworkExplorer({ result, search, setSearch }) {
  if (!result) return <EmptyState icon={Network} title="No observation stream loaded" text="The explorer uses sampled SSSM inference outputs. It does not claim to show live packets or reconstruct source and destination hosts from aggregate feature arrays."/>;
  const rows = (result.predicted_regimes||[]).map((regime,i)=>({id:i+1,step:i+1,regime,name:result.regime_names?.[String(regime)]||`Latent state ${regime}`,truth:result.true_regimes?.[i]??"—",prob:result.regime_uncertainty?.[i]??0})).filter(r=>`${r.name} ${r.step} ${r.regime}`.toLowerCase().includes(search.toLowerCase()));
  return <Panel title="Observation explorer" subtitle="Sampled model outputs · not raw packet captures" action={<div className="search-box"><Search size={14}/><input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Filter observations…"/></div>}><div className="observation-table"><div className="observation-head"><span>STEP</span><span>INFERRED STATE</span><span>GROUND TRUTH</span><span>ATTACK POSTERIOR</span></div>{rows.slice(0,100).map(row=><div className="observation-row" key={row.id}><span className="mono">#{String(row.step).padStart(4,"0")}</span><span><i className="table-regime" style={{background:REGIME[row.regime]?.color||"#718196"}}/>{row.name}</span><span className="mono">{row.truth}</span><strong>{pct(row.prob)}</strong></div>)}</div><div className="table-foot">Showing {Math.min(rows.length,100)} of {rows.length} sampled observations.</div></Panel>;
}

function Reports({ result, dataset, apiOnline }) {
  return <><div className="report-hero"><div className="report-icon"><FileBarChart2 size={25}/></div><div><span className="eyebrow">EXPERIMENT ARCHIVE</span><h2>Security analysis, clearly documented.</h2><p>Export the exact result payload generated by the latest local experiment.</p></div></div>{result?<><div className="report-card"><div className="report-card-top"><div><span className="report-type">PARD-SSM / EVALUATION</span><h3>{dataset.toUpperCase()} experiment report</h3><p>Generated {shortDate(result.created_at)}</p></div><span className="report-ready"><ShieldCheck size={14}/> Result saved</span></div><div className="result-summary"><div><span>ADJUSTED RAND INDEX</span><strong>{Number.isFinite(Number(result.adjusted_rand_index)) ? Number(result.adjusted_rand_index).toFixed(3) : "—"}</strong></div><div><span>NORMALISED MUTUAL INFO</span><strong>{Number.isFinite(Number(result.normalized_mutual_info)) ? Number(result.normalized_mutual_info).toFixed(3) : "—"}</strong></div><div><span>OBSERVATIONS</span><strong>{fmt(result.evaluated_steps)}</strong></div><div><span>FEATURES</span><strong>{fmt(result.feature_count)}</strong></div><div><span>REGIMES</span><strong>{result.regimes}</strong></div></div><a className="primary-btn download-btn" href={`${API}/api/reports/${dataset}`}><Download size={15}/> Download JSON report</a></div><Panel title="Evaluation caveat" subtitle="Read before using the result in a paper or presentation"><div className="method-note inside"><AlertTriangle size={16}/><span>{result.evaluation_note}</span></div></Panel></>:<EmptyState icon={FileBarChart2} title="No report generated yet" text="Complete a model run to save a result report for this dataset."/>}<Panel title="Model transparency" subtitle="What this dashboard does and does not claim"><div className="integrity-grid"><div><ShieldCheck size={17}/><strong>Measured outputs</strong><p>Metrics and plots are populated from actual API experiment results.</p></div><div><AlertTriangle size={17}/><strong>Visible limitations</strong><p>Sampling, latent-label alignment and evaluation caveats are shown alongside results.</p></div><div><Fingerprint size={17}/><strong>Latent states</strong><p>Inferred state IDs are not silently renamed as semantic attack classes.</p></div></div></Panel></>;
}

export default App;
