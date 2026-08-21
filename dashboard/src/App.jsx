import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  ShieldAlert, Activity, Cpu, Layers, Play, Pause, FastForward, 
  CheckCircle, AlertTriangle, XCircle, BarChart3, Database, 
  Terminal, Sliders, RefreshCw, Eye, Sparkles
} from 'lucide-react';
import { 
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Cell, Legend, ScatterChart, Scatter, ZAxis
} from 'recharts';

const isLocal = typeof window !== 'undefined' && (
  window.location.hostname === 'localhost' || 
  window.location.hostname === '127.0.0.1'
);
const API_BASE = isLocal ? 'http://127.0.0.1:8000' : '';

export default function App() {
  const [activeTab, setActiveTab] = useState('monitor'); // 'monitor' | 'xai' | 'benchmarks' | 'models'
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(true);
  const [streamDelay, setStreamDelay] = useState(250);
  
  const [latestPacket, setLatestPacket] = useState(null);
  const [telemetryHistory, setTelemetryHistory] = useState([]);
  const [recentAlerts, setRecentAlerts] = useState([]);
  const [selectedAlert, setSelectedAlert] = useState(null);
  
  const [benchmarkData, setBenchmarkData] = useState([]);
  const [modelComparisonData, setModelComparisonData] = useState([]);
  const [perAttackData, setPerAttackData] = useState([]);
  const [stats, setStats] = useState({
    total_inspected: 0,
    passed_alerts: 0,
    suppressed_alerts: 0,
    normal_traffic: 0
  });

  const eventSourceRef = useRef(null);

  // Fetch static data (benchmarks & model comparisons)
  const fetchStaticData = async () => {
    try {
      const [benchRes, compRes, perAttackRes, statsRes] = await Promise.all([
        axios.get(`${API_BASE}/api/benchmarks`),
        axios.get(`${API_BASE}/api/models/comparison`),
        axios.get(`${API_BASE}/api/models/per-attack`).catch(() => ({ data: [] })),
        axios.get(`${API_BASE}/api/stats`)
      ]);
      setBenchmarkData(benchRes.data);
      setModelComparisonData(compRes.data);
      if (perAttackRes.data && perAttackRes.data.length > 0) {
        setPerAttackData(perAttackRes.data);
      }
      if (statsRes.data.filter_stats) {
        setStats(statsRes.data.filter_stats);
      }
      setIsConnected(true);
    } catch (err) {
      console.warn("Backend not yet connected:", err.message);
      setIsConnected(false);
    }
  };

  useEffect(() => {
    fetchStaticData();
    const interval = setInterval(fetchStaticData, 5000);
    return () => clearInterval(interval);
  }, []);

  // Connect to SSE stream
  useEffect(() => {
    let es;
    const connectSSE = () => {
      try {
        es = new EventSource(`${API_BASE}/api/stream/sse`);
        eventSourceRef.current = es;

        es.onopen = () => {
          setIsConnected(true);
        };

        es.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.error) return;

            setLatestPacket(data);

            // Update rolling telemetry history for chart
            setTelemetryHistory((prev) => {
              const activeName = data.active_metric || 'tcp.len';
              const actualVal = (data.actual_signal !== undefined && data.actual_signal !== null)
                ? data.actual_signal
                : (data.features[activeName] ?? data.features['tcp.len'] ?? data.features['sensor_telemetry'] ?? 50.0);
                
              const predictedVal = (data.twin_signal !== undefined && data.twin_signal !== null)
                ? data.twin_signal
                : (data.predicted_state?.[activeName] ?? data.predicted_state?.['tcp.len'] ?? data.predicted_state?.['sensor_telemetry'] ?? actualVal);
                
              const devVal = (data.mean_deviation !== undefined && data.mean_deviation !== null)
                ? data.mean_deviation
                : Math.abs(actualVal - predictedVal);

              const nextPoint = {
                time: new Date(data.timestamp * 1000).toLocaleTimeString([], { hour12: false, minute: '2-digit', second: '2-digit' }),
                actual: Number(Number(actualVal).toFixed(2)),
                predicted: Number(Number(predictedVal).toFixed(2)),
                deviation: Number(Number(devVal).toFixed(2)),
                label: data.ground_truth,
                predClass: data.prediction?.predicted_class || 'Normal'
              };
              const updated = [...prev, nextPoint];
              return updated.slice(-30); // keep rolling 30 points
            });

            // Update alerts feed
            if (data.filter && data.filter.decision !== "NORMAL") {
              const newAlert = {
                id: data.index,
                time: new Date(data.timestamp * 1000).toLocaleTimeString(),
                attack_type: data.prediction.predicted_class,
                ground_truth: data.ground_truth,
                confidence: Math.round(data.prediction.confidence * 100),
                filter_decision: data.filter.decision,
                reason: data.filter.reason,
                shap_explanation: data.shap_explanation,
                mean_deviation: data.mean_deviation
              };
              setRecentAlerts((prev) => [newAlert, ...prev].slice(0, 30));
              setSelectedAlert((curr) => curr === null ? newAlert : curr);
            }

            // Update counter stats
            setStats((prev) => ({
              total_inspected: prev.total_inspected + 1,
              passed_alerts: prev.passed_alerts + (data.filter?.decision === 'PASS' ? 1 : 0),
              suppressed_alerts: prev.suppressed_alerts + (data.filter?.decision === 'SUPPRESS' ? 1 : 0),
              normal_traffic: prev.normal_traffic + (data.filter?.decision === 'NORMAL' ? 1 : 0)
            }));
          } catch (e) {
            console.error("SSE parse error", e);
          }
        };

        es.onerror = () => {
          setIsConnected(false);
          es.close();
        };
      } catch (err) {
        setIsConnected(false);
      }
    };

    connectSSE();
    return () => {
      if (es) es.close();
    };
  }, []);

  const toggleStreaming = async () => {
    const nextState = !isStreaming;
    setIsStreaming(nextState);
    try {
      await axios.get(`${API_BASE}/api/stream/config`, {
        params: { delay_ms: streamDelay, streaming: nextState }
      });
    } catch (e) {
      console.error(e);
    }
  };

  const handleStep = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/stream/step`);
      setLatestPacket(res.data);
      if (res.data.filter?.decision !== "NORMAL") {
        setSelectedAlert({
          id: res.data.index,
          time: new Date().toLocaleTimeString(),
          attack_type: res.data.prediction.predicted_class,
          ground_truth: res.data.ground_truth,
          confidence: Math.round(res.data.prediction.confidence * 100),
          filter_decision: res.data.filter.decision,
          reason: res.data.filter.reason,
          shap_explanation: res.data.shap_explanation,
          mean_deviation: res.data.mean_deviation
        });
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleDelayChange = async (e) => {
    const newDelay = parseInt(e.target.value);
    setStreamDelay(newDelay);
    try {
      await axios.get(`${API_BASE}/api/stream/config`, {
        params: { delay_ms: newDelay, streaming: isStreaming }
      });
    } catch (err) {
      console.error(err);
    }
  };

  const suppressionRate = stats.total_inspected > 0 
    ? ((stats.suppressed_alerts / (stats.passed_alerts + stats.suppressed_alerts || 1)) * 100).toFixed(1)
    : "0.0";

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Navigation Bar */}
      <header style={{
        borderBottom: '1px solid var(--border-subtle)',
        background: 'rgba(7, 9, 14, 0.85)',
        backdropFilter: 'blur(12px)',
        position: 'sticky',
        top: 0,
        zIndex: 50,
        padding: '0.85rem 2rem',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            background: 'linear-gradient(135deg, #38bdf8 0%, #a855f7 100%)',
            padding: '8px',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            boxShadow: '0 0 20px rgba(56, 189, 248, 0.4)'
          }}>
            <ShieldAlert size={24} color="#07090e" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em', background: 'linear-gradient(to right, #f8fafc, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              Twin-Guided X-IDS
            </h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Industrial IoT Intrusion Detection • Digital Twin Deviation & SHAP XAI
            </p>
          </div>
        </div>

        {/* Tab Controls */}
        <div style={{ display: 'flex', gap: '0.5rem', background: 'rgba(15, 23, 42, 0.8)', padding: '4px', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
          {[
            { id: 'monitor', label: 'Live Monitor', icon: Activity },
            { id: 'xai', label: 'SHAP Explainability', icon: Eye },
            { id: 'benchmarks', label: 'Edge Benchmarks', icon: Cpu },
            { id: 'models', label: 'IDS Comparison', icon: BarChart3 }
          ].map(tab => {
            const Icon = tab.icon;
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  padding: '8px 16px',
                  borderRadius: '8px',
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  transition: 'all 0.2s',
                  background: active ? 'rgba(56, 189, 248, 0.15)' : 'transparent',
                  color: active ? '#38bdf8' : '#94a3b8',
                  boxShadow: active ? '0 0 15px rgba(56, 189, 248, 0.2)' : 'none'
                }}
              >
                <Icon size={16} />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Status Indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '6px 14px',
            borderRadius: '20px',
            background: isConnected ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
            border: `1px solid ${isConnected ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`
          }}>
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: isConnected ? '#10b981' : '#ef4444',
              boxShadow: `0 0 10px ${isConnected ? '#10b981' : '#ef4444'}`
            }} />
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: isConnected ? '#10b981' : '#ef4444' }}>
              {isConnected ? 'LIVE BACKEND' : 'OFFLINE'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main style={{ flex: 1, padding: '1.5rem 2rem', maxWidth: '1600px', margin: '0 auto', width: '100%' }}>
        
        {/* KPI Banner */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.25rem', marginBottom: '1.5rem' }}>
          <div className="glass-panel" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>TOTAL INGESTED</span>
              <Activity size={18} color="#38bdf8" />
            </div>
            <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#f8fafc' }}>
              {stats.total_inspected.toLocaleString()}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              Edge-IIoTset Live Replay
            </div>
          </div>

          <div className="glass-panel" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>HIGH-FIDELITY ALERTS</span>
              <AlertTriangle size={18} color="#ef4444" />
            </div>
            <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#ef4444' }}>
              {stats.passed_alerts.toLocaleString()}
            </div>
            <div style={{ fontSize: '0.75rem', color: '#10b981', marginTop: '4px' }}>
              Passed by Confidence Filter
            </div>
          </div>

          <div className="glass-panel" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>FALSE ALARM SUPPRESSION</span>
              <ShieldAlert size={18} color="#10b981" />
            </div>
            <div style={{ fontSize: '1.75rem', fontWeight: 800, color: '#10b981' }}>
              {suppressionRate}%
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              {stats.suppressed_alerts} Ambiguous alerts filtered
            </div>
          </div>

          <div className="glass-panel" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>TWIN RESIDUAL STATUS</span>
              <Sparkles size={18} color="#a855f7" />
            </div>
            <div style={{ fontSize: '1.75rem', fontWeight: 800, color: latestPacket?.prediction?.predicted_class !== 'Normal' ? '#f59e0b' : '#38bdf8' }}>
              {latestPacket ? latestPacket.prediction.predicted_class : 'Synchronizing...'}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              Residual: {latestPacket ? latestPacket.mean_deviation.toFixed(2) : '0.00'}
            </div>
          </div>
        </div>

        {/* TAB 1: LIVE MONITOR */}
        {activeTab === 'monitor' && (
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem' }}>
            {/* Left Column: Visualizer & Stream Control */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              
              {/* Controls Bar */}
              <div className="glass-panel" style={{ padding: '1rem 1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <button
                    onClick={toggleStreaming}
                    style={{
                      background: isStreaming ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                      color: isStreaming ? '#ef4444' : '#10b981',
                      border: `1px solid ${isStreaming ? '#ef4444' : '#10b981'}`,
                      borderRadius: '8px',
                      padding: '8px 16px',
                      fontWeight: 700,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      cursor: 'pointer'
                    }}
                  >
                    {isStreaming ? <Pause size={16} /> : <Play size={16} />}
                    {isStreaming ? 'PAUSE STREAM' : 'RESUME STREAM'}
                  </button>

                  <button
                    onClick={handleStep}
                    disabled={isStreaming}
                    style={{
                      background: 'rgba(56, 189, 248, 0.1)',
                      color: isStreaming ? '#64748b' : '#38bdf8',
                      border: '1px solid rgba(56, 189, 248, 0.3)',
                      borderRadius: '8px',
                      padding: '8px 14px',
                      fontWeight: 600,
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      cursor: isStreaming ? 'not-allowed' : 'pointer'
                    }}
                  >
                    <FastForward size={16} />
                    STEP (1 PKT)
                  </button>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <Sliders size={16} color="var(--text-muted)" />
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Delay: {streamDelay}ms</span>
                  <input
                    type="range"
                    min="50"
                    max="1000"
                    step="50"
                    value={streamDelay}
                    onChange={handleDelayChange}
                    style={{ width: '120px', accentColor: '#38bdf8' }}
                  />
                </div>
              </div>

              {/* Real-time Dual-Trace Telemetry Chart */}
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '4px' }}>
                      <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Continuous Digital Twin State Tracking</h2>
                      <span style={{
                        background: 'rgba(56, 189, 248, 0.15)',
                        color: '#38bdf8',
                        border: '1px solid rgba(56, 189, 248, 0.3)',
                        padding: '2px 8px',
                        borderRadius: '6px',
                        fontSize: '0.7rem',
                        fontWeight: 700
                      }}>
                        Active Signal: {latestPacket?.active_metric || 'tcp.len'} (Bytes)
                      </span>
                    </div>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                      Blue: Actual Ingested Telemetry (y_t) • Amber: Digital Twin Healthy Forecast (ŷ_t) • Residual: |y_t - ŷ_t|
                    </p>
                  </div>
                  <span style={{
                    padding: '4px 10px',
                    borderRadius: '6px',
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    background: latestPacket?.prediction?.predicted_class !== 'Normal' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                    color: latestPacket?.prediction?.predicted_class !== 'Normal' ? '#ef4444' : '#10b981'
                  }}>
                    {latestPacket?.prediction?.predicted_class !== 'Normal' ? 'ANOMALY DETECTED' : 'HEALTHY TWIN'}
                  </span>
                </div>

                <div style={{ height: '320px', width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={telemetryHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 11 }} />
                      <YAxis 
                        stroke="#64748b" 
                        tick={{ fontSize: 11 }} 
                        domain={[
                          (dataMin) => Math.max(0, Math.floor(dataMin * 0.85)),
                          (dataMax) => Math.max(100, Math.min(Math.ceil(dataMax * 1.15), 3500))
                        ]} 
                      />
                      <Tooltip 
                        contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                        labelStyle={{ color: '#94a3b8' }}
                        formatter={(value, name) => [`${value} Bytes`, name]}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="actual" name="Actual Sensor Value" stroke="#38bdf8" strokeWidth={2.5} dot={{ r: 2, fill: '#38bdf8' }} isAnimationActive={false} />
                      <Line type="monotone" dataKey="predicted" name="Twin Forecast (Expected)" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 5" dot={false} isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Deviation Magnitude Chart */}
              <div className="glass-panel" style={{ padding: '1.5rem' }}>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.5rem' }}>Residual Deviation Spikes |y - ŷ|</h3>
                <div style={{ height: '140px', width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={telemetryHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 10 }} />
                      <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                      <Bar dataKey="deviation" name="Mean Deviation Vector">
                        {telemetryHistory.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.predClass === 'Normal' ? '#10b981' : '#ef4444'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Right Column: Live Alerts & Filter Decisions */}
            <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', height: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Live Threat Alert Stream</h2>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Latest 30</span>
              </div>

              <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.75rem', maxHeight: '680px' }}>
                {recentAlerts.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-muted)' }}>
                    <CheckCircle size={36} color="#10b981" style={{ marginBottom: '12px' }} />
                    <p>No active anomalies detected yet.</p>
                  </div>
                ) : (
                  recentAlerts.map((alert, idx) => {
                    const isPassed = alert.filter_decision === 'PASS';
                    const isSelected = selectedAlert?.id === alert.id;
                    return (
                      <div
                        key={idx}
                        onClick={() => setSelectedAlert(alert)}
                        style={{
                          padding: '12px 14px',
                          borderRadius: '10px',
                          cursor: 'pointer',
                          background: isSelected ? 'rgba(56, 189, 248, 0.15)' : 'rgba(30, 41, 59, 0.5)',
                          border: `1px solid ${isSelected ? '#38bdf8' : isPassed ? 'rgba(239, 68, 68, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`,
                          transition: 'all 0.15s ease'
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                          <span style={{ fontWeight: 700, fontSize: '0.9rem', color: isPassed ? '#ef4444' : '#f59e0b' }}>
                            {alert.attack_type}
                          </span>
                          <span style={{
                            fontSize: '0.7rem',
                            fontWeight: 700,
                            padding: '2px 8px',
                            borderRadius: '4px',
                            background: isPassed ? '#ef4444' : 'rgba(245, 158, 11, 0.2)',
                            color: isPassed ? '#fff' : '#f59e0b'
                          }}>
                            {alert.filter_decision}
                          </span>
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', justifyContent: 'space-between' }}>
                          <span>Confidence: {alert.confidence}%</span>
                          <span className="font-mono">{alert.time}</span>
                        </div>
                        <p style={{ fontSize: '0.7rem', color: '#94a3b8', marginTop: '6px', lineHeight: 1.3 }}>
                          {alert.reason}
                        </p>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: SHAP EXPLAINABILITY */}
        {activeTab === 'xai' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: '1.5rem' }}>
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <h2 style={{ fontSize: '1.2rem', fontWeight: 700 }}>
                  Local SHAP Feature Attributions
                </h2>
                <span style={{
                  background: 'rgba(56, 189, 248, 0.15)',
                  color: '#38bdf8',
                  border: '1px solid rgba(56, 189, 248, 0.3)',
                  padding: '3px 10px',
                  borderRadius: '6px',
                  fontSize: '0.72rem',
                  fontWeight: 700
                }}>
                  ● LIVE LOCAL SHAP ATTRIBUTION
                </span>
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
                Real-time feature attributions computed per incoming alert by SHAP TreeExplainer.
              </p>

              {selectedAlert?.shap_explanation ? (
                <div style={{ height: '350px', width: '100%' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      layout="vertical"
                      data={selectedAlert.shap_explanation.top_features}
                      margin={{ top: 10, right: 30, left: 80, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis type="number" stroke="#64748b" tick={{ fontSize: 11 }} />
                      <YAxis dataKey="feature" type="category" stroke="#64748b" tick={{ fontSize: 11 }} />
                      <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                      <Bar dataKey="shap_value" name="SHAP Impact Score">
                        {selectedAlert.shap_explanation.top_features.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.shap_value > 0 ? '#ef4444' : '#10b981'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--text-muted)' }}>
                  <Eye size={40} style={{ marginBottom: '12px' }} />
                  <p>Select an alert from the Live Monitor to inspect its SHAP breakdown.</p>
                </div>
              )}
            </div>

            {/* Explanation Details Card */}
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '1rem' }}>Attack Signature & Filter Diagnostic</h3>
              {selectedAlert ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px', borderRadius: '8px' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>PREDICTED THREAT</div>
                    <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#38bdf8' }}>{selectedAlert.attack_type}</div>
                  </div>

                  <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px', borderRadius: '8px' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>FILTER DECISION</div>
                    <div style={{ fontSize: '1.1rem', fontWeight: 700, color: selectedAlert.filter_decision === 'PASS' ? '#ef4444' : '#f59e0b' }}>
                      {selectedAlert.filter_decision}
                    </div>
                    <p style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '4px' }}>{selectedAlert.reason}</p>
                  </div>

                  <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px', borderRadius: '8px' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px' }}>TOP CONTRIBUTING SENSORS</div>
                    {selectedAlert.shap_explanation?.top_features.map((f, i) => (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', padding: '4px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                        <span style={{ color: '#f8fafc' }}>{f.feature}</span>
                        <span style={{ color: f.shap_value > 0 ? '#ef4444' : '#10b981', fontWeight: 600 }}>{f.shap_value.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <p style={{ color: 'var(--text-muted)' }}>No alert selected.</p>
              )}
            </div>
          </div>
        )}

        {/* TAB 3: EDGE BENCHMARKS */}
        {activeTab === 'benchmarks' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div className="glass-panel" style={{ padding: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 800 }}>
                  Resource-Aware Edge Trade-Off Benchmarks
                </h2>
                <span style={{
                  background: 'rgba(245, 158, 11, 0.15)',
                  color: '#f59e0b',
                  border: '1px solid rgba(245, 158, 11, 0.3)',
                  padding: '3px 10px',
                  borderRadius: '6px',
                  fontSize: '0.72rem',
                  fontWeight: 700
                }}>
                  📊 OFFLINE EMPIRICAL BENCHMARK (5-Run Profile)
                </span>
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
                Experimental evaluation across 4 Edge-IIoT deployment configurations measuring Latency (ms), Storage Size (KB), and Macro-F1.
              </p>

              <div style={{ height: '340px', width: '100%', marginBottom: '2rem' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={benchmarkData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="Configuration" stroke="#64748b" tick={{ fontSize: 10 }} />
                    <YAxis yAxisId="left" orientation="left" stroke="#38bdf8" label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft', fill: '#38bdf8' }} />
                    <YAxis yAxisId="right" orientation="right" stroke="#10b981" label={{ value: 'Accuracy (%)', angle: 90, position: 'insideRight', fill: '#10b981' }} />
                    <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                    <Legend />
                    <Bar yAxisId="left" dataKey="Avg Latency (ms/sample)" name="Inference Latency (ms)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                    <Bar yAxisId="right" dataKey="Accuracy (%)" name="Accuracy (%)" fill="#10b981" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Master Benchmark Table */}
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                      <th style={{ padding: '12px' }}>CONFIGURATION</th>
                      <th style={{ padding: '12px' }}>FEATURE SPACE</th>
                      <th style={{ padding: '12px' }}>ACCURACY</th>
                      <th style={{ padding: '12px' }}>MACRO-F1</th>
                      <th style={{ padding: '12px' }}>LATENCY (MS)</th>
                      <th style={{ padding: '12px' }}>THROUGHPUT (SAMPLES/S)</th>
                      <th style={{ padding: '12px' }}>STORAGE (KB)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkData.map((row, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: idx % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent' }}>
                        <td style={{ padding: '12px', fontWeight: 600, color: '#f8fafc' }}>{row["Configuration"]}</td>
                        <td style={{ padding: '12px', color: '#94a3b8' }}>{row["Feature Space"]}</td>
                        <td style={{ padding: '12px', color: '#10b981', fontWeight: 700 }}>{row["Accuracy (%)"]}%</td>
                        <td style={{ padding: '12px', color: '#38bdf8', fontWeight: 700 }}>{row["Macro-F1"]}</td>
                        <td style={{ padding: '12px', color: '#f59e0b' }}>{row["Avg Latency (ms/sample)"]} ms</td>
                        <td style={{ padding: '12px' }}>{row["Throughput (samples/sec)"]}</td>
                        <td style={{ padding: '12px', color: '#a855f7' }}>{row["Total Footprint (KB)"]} KB</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: IDS ARCHITECTURE COMPARISON */}
        {activeTab === 'models' && (
          <div className="glass-panel" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 800 }}>
                Baseline vs. Twin-Deviation vs. Twin-Augmented IDS Comparison
              </h2>
              <span style={{
                background: 'rgba(168, 85, 247, 0.15)',
                color: '#a855f7',
                border: '1px solid rgba(168, 85, 247, 0.3)',
                padding: '3px 10px',
                borderRadius: '6px',
                fontSize: '0.72rem',
                fontWeight: 700
              }}>
                📈 OFFLINE TEST EVALUATION (13,999 Samples)
              </span>
            </div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
              Direct experimental comparison across 6 classifiers trained on raw telemetry vs. residual deviation spaces on held-out test data.
            </p>

            <div style={{ height: '340px', width: '100%', marginBottom: '2rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={modelComparisonData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="Model Architecture" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis domain={[30, 100]} stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                  <Legend />
                  <Bar dataKey="Accuracy (%)" name="Accuracy (%)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Macro-F1" name="Macro-F1 (Ratio)" fill="#10b981" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Weighted-F1" name="Weighted-F1" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div style={{ overflowX: 'auto', marginBottom: '2.5rem' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '12px' }}>MODEL ARCHITECTURE</th>
                    <th style={{ padding: '12px' }}>ACCURACY (%)</th>
                    <th style={{ padding: '12px' }}>MACRO-F1</th>
                    <th style={{ padding: '12px' }}>WEIGHTED-F1</th>
                    <th style={{ padding: '12px' }}>MACRO-PRECISION</th>
                    <th style={{ padding: '12px' }}>MACRO-RECALL</th>
                  </tr>
                </thead>
                <tbody>
                  {modelComparisonData.map((row, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <td style={{ padding: '12px', fontWeight: 600, color: '#f8fafc' }}>{row["Model Architecture"]}</td>
                      <td style={{ padding: '12px', color: '#10b981', fontWeight: 700 }}>{row["Accuracy (%)"]}%</td>
                      <td style={{ padding: '12px', color: '#38bdf8', fontWeight: 700 }}>{row["Macro-F1"]}</td>
                      <td style={{ padding: '12px', color: '#f59e0b' }}>{row["Weighted-F1"]}</td>
                      <td style={{ padding: '12px' }}>{row["Macro-Precision"]}</td>
                      <td style={{ padding: '12px' }}>{row["Macro-Recall"]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Sub-Panel: 15-Class Fine-Grained Per-Attack Breakdown */}
            <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '2rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div>
                  <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: '#f8fafc' }}>
                    15-Class Fine-Grained Threat Performance Breakdown
                  </h3>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Granular classification F1-scores across all 15 attack types evaluated on 13,999 stratified test samples.
                  </p>
                </div>
                <div style={{
                  background: 'rgba(56, 189, 248, 0.12)',
                  border: '1px solid rgba(56, 189, 248, 0.3)',
                  padding: '6px 14px',
                  borderRadius: '8px',
                  fontSize: '0.8rem',
                  color: '#38bdf8',
                  fontWeight: 700
                }}>
                  Exact/Statistical Parity on 11 of 15 Classes
                </div>
              </div>

              {/* Grouped Comparison Bar Chart */}
              {perAttackData.length > 0 && (
                <div style={{ height: '360px', width: '100%', marginBottom: '2rem' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={perAttackData} margin={{ top: 20, right: 30, left: 20, bottom: 40 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="Attack Class" stroke="#64748b" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" />
                      <YAxis domain={[0.5, 1.05]} stroke="#64748b" tick={{ fontSize: 11 }} />
                      <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                      <Legend verticalAlign="top" height={36} />
                      <Bar dataKey="XGB-Raw F1" name="XGB-Raw Baseline (34 Features)" fill="#64748b" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="XGB-Twin-v2 F1" name="XGB-Twin-Augmented-v2 (43 Features)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="RF-Twin-v2 F1" name="RF-Twin-Augmented-v2 (43 Features)" fill="#a855f7" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Per-Attack Interactive Table */}
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.82rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                      <th style={{ padding: '10px' }}>ATTACK CLASS</th>
                      <th style={{ padding: '10px' }}>CATEGORY</th>
                      <th style={{ padding: '10px' }}>SUPPORT</th>
                      <th style={{ padding: '10px' }}>XGB-RAW F1</th>
                      <th style={{ padding: '10px' }}>XGB-TWIN-V2 F1</th>
                      <th style={{ padding: '10px' }}>DELTA (ΔF1)</th>
                      <th style={{ padding: '10px' }}>RF-TWIN-V2 F1</th>
                      <th style={{ padding: '10px' }}>OUTCOME</th>
                    </tr>
                  </thead>
                  <tbody>
                    {perAttackData.map((row, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', background: idx % 2 === 0 ? 'rgba(255,255,255,0.01)' : 'transparent' }}>
                        <td style={{ padding: '10px', fontWeight: 700, color: '#f8fafc' }}>{row["Attack Class"]}</td>
                        <td style={{ padding: '10px', color: '#94a3b8' }}>{row["Category"]}</td>
                        <td style={{ padding: '10px', color: 'var(--text-muted)' }}>{row["Support"]}</td>
                        <td style={{ padding: '10px', color: '#94a3b8' }}>{Number(row["XGB-Raw F1"]).toFixed(4)}</td>
                        <td style={{ padding: '10px', color: '#38bdf8', fontWeight: 700 }}>{Number(row["XGB-Twin-v2 F1"]).toFixed(4)}</td>
                        <td style={{
                          padding: '10px',
                          fontWeight: 700,
                          color: row["XGB F1 Delta"] >= 0 ? '#10b981' : (Math.abs(row["XGB F1 Delta"]) <= 0.005 ? '#38bdf8' : '#ef4444')
                        }}>
                          {row["XGB F1 Delta"] >= 0 ? `+${Number(row["XGB F1 Delta"]).toFixed(4)}` : Number(row["XGB F1 Delta"]).toFixed(4)}
                        </td>
                        <td style={{ padding: '10px', color: '#a855f7' }}>{Number(row["RF-Twin-v2 F1"]).toFixed(4)}</td>
                        <td style={{ padding: '10px' }}>
                          <span style={{
                            padding: '3px 8px',
                            borderRadius: '4px',
                            fontSize: '0.72rem',
                            fontWeight: 700,
                            background: row["Outcome"] === 'Exact Parity' ? 'rgba(16, 185, 129, 0.2)' : (row["Outcome"] === 'Statistical Parity' ? 'rgba(56, 189, 248, 0.2)' : 'rgba(100, 116, 139, 0.2)'),
                            color: row["Outcome"] === 'Exact Parity' ? '#10b981' : (row["Outcome"] === 'Statistical Parity' ? '#38bdf8' : '#94a3b8')
                          }}>
                            {row["Outcome"]}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
