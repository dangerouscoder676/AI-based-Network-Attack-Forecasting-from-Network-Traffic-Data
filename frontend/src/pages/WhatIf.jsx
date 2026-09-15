import { useState } from 'react';
import { FlaskConical, Play, ShieldCheck, ShieldOff, ShieldX, Unplug } from 'lucide-react';
import PageHeader from '../components/PageHeader';
import Panel from '../components/Panel';
import MetricCard from '../components/MetricCard';
import ChartBars from '../components/ChartBars';
import { apiRequest } from '../services/projectData';

const PRESETS = [
  { key: 'isolate_attacker', label: 'Isolate the attacker source', spec: { type: 'isolate_attacker', source: 'auto' }, Icon: ShieldX,
    copy: 'Block the external host at your gateway — the single action that truly shrinks the threat.' },
  { key: 'block_port', label: 'Block the attacker door', spec: { type: 'block_port', port: 'auto' }, Icon: ShieldOff,
    copy: 'Close the single most-hit network port the attacker probed.' },
  { key: 'disconnect_host', label: 'Disconnect a device', spec: { type: 'disconnect_host', ip: 'auto' }, Icon: Unplug,
    copy: 'Take the most-targeted machine off the network immediately.' },
  { key: 'slow_logins', label: 'Slow login attempts', spec: { type: 'slow_logins', factor: 0.3 }, Icon: FlaskConical,
    copy: 'Force attackers to retry logins 70% more slowly — throttle brute force.' },
];

export default function WhatIf({ onNotice }) {
  const [selected, setSelected] = useState(PRESETS[0]);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);

  const run = (preset = selected) => {
    setBusy(true);
    apiRequest('/api/whatif', { method: 'POST', body: JSON.stringify({ seed: 7, k_steps: 10, intervention: preset.spec }) })
      .then(r => {
        setSelected(preset);
        setResult(r);
        const pct = r.reduction != null ? Math.round(r.reduction * 100) : 0;
        onNotice?.(pct > 0
          ? `What-if: ${preset.label} cuts sustained intrusion confidence by ${pct}%.`
          : 'What-if: this action showed no material change.');
      })
      .catch(e => onNotice?.(e.message))
      .finally(() => setBusy(false));
  };

  const baseAvg = result?.baseline?.mean ?? result?.baseline?.peak ?? 0;
  const ctrAvg = result?.counterfactual?.mean ?? result?.counterfactual?.peak ?? 0;
  const basePeak = result?.baseline?.peak ?? 0;
  const ctrPeak = result?.counterfactual?.peak ?? 0;
  const baseProbs = result?.baseline?.probabilities?.length ? result.baseline.probabilities : [0.22, 0.28, 0.31, 0.38, 0.46, 0.63, 0.55, 0.49, 0.41, 0.36];
  const ctrProbs = result?.counterfactual?.probabilities?.length ? result.counterfactual.probabilities : baseProbs;

  return (
    <>
      <PageHeader
        eyebrow="INTELLIGENCE / WHAT-IF"
        title="Counterfactual defence planner"
        description="Ask the AI what happens if you actually do something to stop the attack."
        action={
          <button className="button primary" onClick={() => run()} disabled={busy}>
            <Play size={16} /> {busy ? 'Simulating…' : result ? 'Re-run what-if' : 'Run what-if'}
          </button>
        }
      />
      <div className="section-grid">
        {PRESETS.map(p => (
          <button key={p.key} className={`action-card ${selected.key === p.key ? 'metric-card' : ''}`} onClick={() => run(p)} disabled={busy}>
            <span className="icon-box small"><p.Icon size={19} /></span>
            <strong>{p.label}</strong>
            <p>{p.copy}</p>
            <span className="text-link">{selected.key === p.key ? 'Selected · click to run' : 'Run this action'}</span>
          </button>
        ))}
      </div>
      {result?.intervention && (
        <div className="metric-grid">
          <MetricCard label="Average confidence" value={`${(baseAvg * 100).toFixed(0)}%`} note={`Peak ${(basePeak * 100).toFixed(0)}% · ${result.baseline.stage_name}`} tone={baseAvg > 0.6 ? 'warn' : 'safe'} />
          <MetricCard label={`After: ${result.intervention.label}`} value={`${(ctrAvg * 100).toFixed(0)}%`} note="Counterfactual average" tone={ctrAvg > 0.6 ? 'warn' : 'safe'} />
          <MetricCard label="Threat reduction" value={`${Math.round(result.reduction * 100)}%`} note="Sustained confidence removed" tone={result.recommended ? 'safe' : 'neutral'} />
          <MetricCard label="Verdict" value={result.recommended ? 'ACT NOW' : 'ASSESS'} note="Recommended defence" tone={result.recommended ? 'safe' : 'warn'} />
        </div>
      )}
      {result && (
        <>
          <div className="two-col">
            <Panel title="Forecast with no action" kicker="BASELINE">
              <div className="chart-summary"><div><span>Average intrusion confidence</span><strong>{(baseAvg * 100).toFixed(1)}%</strong></div></div>
              <ChartBars values={baseProbs} labels={baseProbs.map((_, i) => i === 0 ? 'Now' : `+${i}`)} color="#b33a3a" />
              <div className="chart-foot">{result.baseline.story[1]}</div>
            </Panel>
            <Panel title={`After: ${result.intervention.label}`} kicker="COUNTERFACTUAL">
              <div className="chart-summary"><div><span>Average intrusion confidence</span><strong>{(ctrAvg * 100).toFixed(1)}%</strong></div></div>
              <ChartBars values={ctrProbs} labels={ctrProbs.map((_, i) => i === 0 ? 'Now' : `+${i}`)} color="#287a5a" />
              <div className="chart-foot">{result.counterfactual.story[1]}</div>
            </Panel>
          </div>
          <Panel title="The AI's recommendation" kicker="PLAIN LANGUAGE">
            <div className="callout safe"><ShieldCheck size={17} /><span>{result.verdict.story.join(' ')}</span></div>
          </Panel>
        </>
      )}
      {!result && (
        <Panel title="How it works" kicker="PROACTIVE DEFENCE">
          <p className="body-copy">
            NetSeer re-runs its attack simulation with your chosen defence in place — blocking a
            service port, disconnecting a machine, or slowing brute-force logins — and shows you
            the change in breach confidence. Pick an action above and hit <strong>Run what-if</strong>.
          </p>
        </Panel>
      )}
    </>
  );
}