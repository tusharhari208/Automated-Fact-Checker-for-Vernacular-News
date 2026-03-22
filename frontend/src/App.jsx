import { useState, useRef } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "https://thelma-casketlike-cully.ngrok-free.dev";

function VerdictBadge({ verdict }) {
  const map = {
    TRUE: { label: "VERIFIED TRUE", cls: "badge-true", icon: "✓" },
    FALSE: { label: "VERIFIED FALSE", cls: "badge-false", icon: "✗" },
    UNVERIFIABLE: { label: "UNVERIFIABLE", cls: "badge-unknown", icon: "?" },
  };
  const v = map[verdict] || map["UNVERIFIABLE"];
  return (
    <span className={`badge ${v.cls}`}>
      <span className="badge-icon">{v.icon}</span>
      {v.label}
    </span>
  );
}

function ConfidenceBar({ value }) {
  const pct = Math.min(100, Math.max(0, value));
  const color = pct > 70 ? "#00e5a0" : pct > 40 ? "#f5c842" : "#ff4d6d";
  return (
    <div className="conf-wrap">
      <div className="conf-bar">
        <div className="conf-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="conf-label" style={{ color }}>{pct.toFixed(1)}%</span>
    </div>
  );
}

function ResultCard({ result, index }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className={`result-card verdict-${result.verdict?.toLowerCase()}`} style={{ animationDelay: `${index * 80}ms` }}>
      <div className="result-header" onClick={() => setExpanded(!expanded)}>
        <div className="result-top">
          <span className="result-num">#{index + 1}</span>
          <VerdictBadge verdict={result.verdict} />
          <span className="expand-btn">{expanded ? "▲" : "▼"}</span>
        </div>
        <p className="result-claim">"{result.claim}"</p>
        <ConfidenceBar value={result.confidence || 0} />
      </div>
      {expanded && (
        <div className="result-body">
          {result.source && (
            <div className="meta-row">
              <span className="meta-label">Source</span>
              <span className="meta-val">{result.source}</span>
            </div>
          )}
          {result.evidence && (
            <div className="meta-row">
              <span className="meta-label">Evidence</span>
              <span className="meta-val evidence-text">"{result.evidence}"</span>
            </div>
          )}
          {result.explanation && (
            <div className="meta-row">
              <span className="meta-label">Explanation</span>
              <span className="meta-val">{result.explanation}</span>
            </div>
          )}
          {result.error && (
            <div className="meta-row error-row">
              <span className="meta-label">Error</span>
              <span className="meta-val">{result.error}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function StatsBar({ stats }) {
  if (!stats) return null;
  const trueCount = stats.results?.filter(r => r.verdict === "TRUE").length || 0;
  const falseCount = stats.results?.filter(r => r.verdict === "FALSE").length || 0;
  const unknownCount = stats.results?.filter(r => r.verdict === "UNVERIFIABLE").length || 0;
  return (
    <div className="stats-bar">
      <div className="stat-pill stat-true"><span>{trueCount}</span> True</div>
      <div className="stat-pill stat-false"><span>{falseCount}</span> False</div>
      <div className="stat-pill stat-unknown"><span>{unknownCount}</span> Unverifiable</div>
      {stats.elapsed_seconds && (
        <div className="stat-pill stat-time"><span>{stats.elapsed_seconds}s</span> Elapsed</div>
      )}
      {stats.throughput && (
        <div className="stat-pill stat-speed"><span>{stats.throughput}</span> posts/min</div>
      )}
    </div>
  );
}

export default function App() {
  const [mode, setMode] = useState("single"); // single | batch
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [batchStats, setBatchStats] = useState(null);
  const [error, setError] = useState("");
  const [apiStatus, setApiStatus] = useState(null); // null | ok | fail
  const textRef = useRef();

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`, {
        headers: { "ngrok-skip-browser-warning": "true" }
      });
      const data = await res.json();
      setApiStatus(data.status === "running" ? "ok" : "fail");
    } catch {
      setApiStatus("fail");
    }
  }

  async function runSingle() {
    if (!input.trim()) return;
    setLoading(true);
    setError("");
    setResults([]);
    setBatchStats(null);
    try {
      const res = await fetch(`${API_BASE}/fact-check`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "ngrok-skip-browser-warning": "true" },
        body: JSON.stringify({ claim: input.trim() }),
      });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data = await res.json();
      setResults([{ claim: input.trim(), ...data }]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function runBatch() {
    const lines = input.split("\n").map(l => l.replace(/^\d+\.\s*/, "").trim()).filter(Boolean);
    if (!lines.length) return;
    setLoading(true);
    setError("");
    setResults([]);
    setBatchStats(null);
    try {
      const res = await fetch(`${API_BASE}/fact-check/batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "ngrok-skip-browser-warning": "true" },
        body: JSON.stringify({ claims: lines }),
      });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data = await res.json();
      setBatchStats(data);
      setResults(data.results || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const handleSubmit = () => (mode === "single" ? runSingle() : runBatch());

  const sampleSingle = "Vaccines cause autism";
  const sampleBatch = `1. Vaccines cause autism
2. India won the 2011 Cricket World Cup
3. The Eiffel Tower is in London
4. Neil Armstrong walked on the moon`;

  return (
    <div className="app">
      {/* Bg grid */}
      <div className="bg-grid" />
      <div className="bg-glow" />

      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo-wrap">
            <div className="logo-icon">
              <svg viewBox="0 0 40 40" fill="none">
                <circle cx="20" cy="20" r="18" stroke="url(#lg)" strokeWidth="2"/>
                <path d="M13 20l5 5 9-10" stroke="url(#lg)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                <defs>
                  <linearGradient id="lg" x1="0" y1="0" x2="40" y2="40">
                    <stop stopColor="#00e5a0"/>
                    <stop offset="1" stopColor="#0098f7"/>
                  </linearGradient>
                </defs>
              </svg>
            </div>
            <div>
              <h1 className="logo-title">FactShield</h1>
              <p className="logo-sub">AI-Powered Fact Checker · Intel Unnati</p>
            </div>
          </div>
          <button
            className={`health-btn ${apiStatus === "ok" ? "health-ok" : apiStatus === "fail" ? "health-fail" : ""}`}
            onClick={checkHealth}
          >
            <span className={`dot ${apiStatus === "ok" ? "dot-ok" : apiStatus === "fail" ? "dot-fail" : "dot-idle"}`} />
            {apiStatus === null ? "Check API" : apiStatus === "ok" ? "API Online" : "API Offline"}
          </button>
        </div>
      </header>

      <main className="main">
        {/* Intro */}
        <section className="intro">
          <h2 className="intro-title">Detect Misinformation<br /><span className="grad-text">Instantly & Accurately</span></h2>
          <p className="intro-desc">
            Multi-layer AI verification using Classical ML · FAISS Semantic Search · Wikipedia · Google Gemini
          </p>
          <div className="tech-pills">
            {["multilingual-e5-small", "FAISS", "FEVER Dataset", "Gemini 1.5 Flash", "NLI Model"].map(t => (
              <span key={t} className="tech-pill">{t}</span>
            ))}
          </div>
        </section>

        {/* Input panel */}
        <section className="panel">
          <div className="mode-tabs">
            <button className={`tab ${mode === "single" ? "tab-active" : ""}`} onClick={() => { setMode("single"); setResults([]); setError(""); }}>
              Single Claim
            </button>
            <button className={`tab ${mode === "batch" ? "tab-active" : ""}`} onClick={() => { setMode("batch"); setResults([]); setError(""); }}>
              Batch Check
            </button>
          </div>

          <div className="input-area">
            <textarea
              ref={textRef}
              className="textarea"
              rows={mode === "batch" ? 7 : 4}
              placeholder={mode === "single"
                ? "Enter a news claim to verify…"
                : "Enter numbered claims:\n1. First claim here\n2. Second claim here\n3. Third claim here"}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && e.ctrlKey) handleSubmit(); }}
            />
            <div className="input-actions">
              <button className="sample-btn" onClick={() => setInput(mode === "single" ? sampleSingle : sampleBatch)}>
                Load Sample
              </button>
              <button className="clear-btn" onClick={() => { setInput(""); setResults([]); setError(""); setBatchStats(null); }}>
                Clear
              </button>
              <button className="submit-btn" onClick={handleSubmit} disabled={loading || !input.trim()}>
                {loading ? <span className="spinner" /> : null}
                {loading ? "Verifying…" : mode === "single" ? "Verify Claim →" : "Verify Batch →"}
              </button>
            </div>
          </div>
          <p className="tip">Tip: Press Ctrl+Enter to submit</p>
        </section>

        {/* Error */}
        {error && (
          <div className="error-box">
            <span>⚠ {error}</span>
          </div>
        )}

        {/* Results */}
        {results.length > 0 && (
          <section className="results-section">
            <div className="results-header">
              <h3 className="results-title">{results.length === 1 ? "Fact Check Result" : `Results · ${results.length} claims`}</h3>
              <StatsBar stats={batchStats} />
            </div>
            <div className="results-list">
              {results.map((r, i) => <ResultCard key={i} result={r} index={i} />)}
            </div>
          </section>
        )}

        {/* Architecture */}
        <section className="arch-section">
          <h3 className="section-title">Verification Pipeline</h3>
          <div className="arch-flow">
            {[
              { icon: "⚡", label: "ML Models", sub: "Passive Aggressive + Logistic Regression" },
              { icon: "🔍", label: "FAISS Search", sub: "100K+ FEVER verified claims" },
              { icon: "🧠", label: "NLI Model", sub: "BART-large-mnli entailment check" },
              { icon: "📡", label: "Wikipedia", sub: "Live fact cross-reference" },
              { icon: "✨", label: "Gemini LLM", sub: "Final reasoning & explanation" },
            ].map((s, i) => (
              <div key={i} className="arch-step">
                <div className="arch-icon">{s.icon}</div>
                <div className="arch-label">{s.label}</div>
                <div className="arch-sub">{s.sub}</div>
                {i < 4 && <div className="arch-arrow">→</div>}
              </div>
            ))}
          </div>
        </section>
      </main>

      <footer className="footer">
        <p>Intel Unnati Industrial Training · Fake News Detection Project</p>
        <p className="footer-note">For educational purposes only. Verify important news from trusted sources.</p>
      </footer>
    </div>
  );
}
