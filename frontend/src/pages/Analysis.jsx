import { useEffect, useState } from 'react';
import { BrainCircuit, CheckCircle2, RefreshCw } from 'lucide-react';
import PageHeader from '../components/PageHeader';
import Panel from '../components/Panel';
import MetricCard from '../components/MetricCard';
import { apiRequest, STAGES } from '../services/projectData';

const MITRE = [
  { stage: 'Benign', technique: 'No active technique', label: 'Routine traffic' },
  { stage: 'Reconnaissance', technique: 'Network service scanning · T1046', label: 'Looking for weak spots' },
  { stage: 'Initial Access', technique: 'Exploit public-facing service · T1190', label: 'Breaking in through an open door' },
  { stage: 'Lateral Movement', technique: 'Remote services / SMB · T1021', label: 'Moving between devices' },
  { stage: 'Command & Control', technique: 'Ingress tool transfer · T1105', label: 'Taking control of a device' },
  { stage: 'Exfiltration', technique: 'Exfiltration over C2 · T1041', label: 'Stealing data out' },
];

export default function Analysis({ onNotice }) {
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = () => {
    setBusy(true);
    apiRequest('/api/analysis', { method: 'POST', body: JSON.stringify({ context_index: 0, k_steps: 10, model_type: 'transformer' }) })
      .then(setData)
      .catch(e => onNotice?.(e.message))
      .finally(() => setBusy(false));
  };
  useEffect(run, []);

  const probs = data?.probabilities?.length ? data.probabilities : [0.22, 0.28, 0.31, 0.38, 0.46, 0.63, 0.55, 0.49, 0.41, 0.36];
  const peak = Math.max(...probs);
  const stageIdx = data?.stages?.[probs.indexOf(peak)] ?? 0;
  const stageName = data?.stage_names?.[stageIdx] ?? STAGES[Math.min(stageIdx, STAGES.length - 1)];
  const mitre = MITRE.find(m => m.stage === stageName) || MITRE[1];

  const drivers = data?.shap?.top_features || [['SYN Flag Count', 0.82], ['Flow Pkts/s', 0.58], ['Flow IAT Mean', 0.31], ['Pkt Len Std', 0.19]];
  const attention = data?.attention?.temporal_attention?.length
    ? data.attention.temporal_attention
    : Array.from({ length: 16 }, (_, i) => 0.3 + ((i * 7) % 40) / 160);

  return (
    <>
      <PageHeader
        eyebrow="INTELLIGENCE / ANALYSIS"
        title="Explain the forecast"
        description="Inspect the predicted trajectory and the signals that moved the model."
        action={
          <button className="button" onClick={run} disabled={busy}>
            <RefreshCw size={16} /> {busy ? 'Analysing…' : 'Re-run analysis'}
          </button>
        }
      />
      {!data && !busy && <div className="empty-state"><p>Waiting for the analysis endpoint…</p></div>}
      <div className="metric-grid">
        <MetricCard label="Peak infiltration" value={`${(peak * 100).toFixed(1)}%`} note={`Step +${probs.indexOf(peak)}`} tone={peak > 0.6 ? 'warn' : 'safe'} />
        <MetricCard label="Dominant stage" value={stageName} note={mitre.label} />
        <MetricCard label="Top driver" value={drivers[0]?.[0] ?? '—'} note="Positive contribution" />
        <MetricCard label="Ground truth" value={data?.ground_truth_stage != null ? STAGES[data.ground_truth_stage] ?? `Stage ${data.ground_truth_stage}` : '—'} note="Context label" />
      </div>
      <div className="two-col">
        <Panel title="Feature drivers" kicker="SHAP ATTRIBUTIONS">
          {drivers.map(([name, importance]) => {
            const abs = Math.min(Math.abs(importance), 1);
            const tone = importance >= 0 ? 'danger' : 'safe';
            return <Driver key={name} label={name} value={`${importance >= 0 ? '+' : '−'}${abs.toFixed(2)}`} width={`${(abs * 100).toFixed(0)}%`} tone={tone} />;
          })}
          {data?.shap_error && <div className="chart-foot">SHAP unavailable ({data.shap_error}).</div>}
        </Panel>
        <Panel title="Temporal attention" kicker="MODEL FOCUS">
          <div className="attention-grid">
            {attention.map((w, i) => <span key={i} style={{ opacity: 0.18 + Math.min(w, 1) * 0.82 }} />)}
          </div>
          <div className="chart-foot">Each cell is one observed context window — brighter means it pulled more of the model's focus.</div>
          {data?.attention_error && <div className="chart-foot">Attention unavailable ({data.attention_error}).</div>}
        </Panel>
      </div>
      <Panel title="MITRE ATT&CK mapping" kicker="STAGE CONTEXT">
        <div className="mapping-row">
          <span className="icon-box small"><BrainCircuit size={18} /></span>
          <div><strong>{mitre.stage}</strong><p>{mitre.technique}</p></div>
          <CheckCircle2 className="check" size={19} />
        </div>
      </Panel>
    </>
  );
}

function Driver({ label, value, width, tone }) {
  return (
    <div className="driver">
      <div><span>{label}</span><strong className={tone}>{value}</strong></div>
      <div className="driver-track"><i className={tone} style={{ width }} /></div>
    </div>
  );
}