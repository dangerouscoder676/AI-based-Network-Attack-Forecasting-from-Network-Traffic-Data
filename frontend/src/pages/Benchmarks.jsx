import { useEffect, useState } from 'react';
import { Info } from 'lucide-react';
import PageHeader from '../components/PageHeader';
import Panel from '../components/Panel';
import ChartBars from '../components/ChartBars';
import { apiRequest, benchmarkModels } from '../services/projectData';

export default function Benchmarks() {
  const [realData, setRealData] = useState(null);
  const [realErr, setRealErr] = useState(null);

  useEffect(() => {
    apiRequest('/api/benchmarks/real')
      .then(setRealData)
      .catch(e => setRealErr(e.message));
  }, []);

  const synthNames = benchmarkModels.map(m => m.name);
  const synthF1 = benchmarkModels.map(m => m.metrics.F1 || 0);
  const realResults = realData?.results ? Object.values(realData.results) : [];
  const realNames = realResults.map(r => r.name);
  const realF1 = realResults.map(r => r.metrics.F1 || 0);

  return (
    <>
      <PageHeader
        eyebrow="INTELLIGENCE / BENCHMARKS"
        title="Model performance"
        description="Metrics loaded from the repository's benchmark_results.json artifact."
      />
      <Panel title="Synthetic comparison matrix" kicker="SYSTEM CHECK">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Model</th><th>F1</th><th>Precision</th><th>Recall</th><th>AUC-ROC</th><th>Accuracy</th></tr></thead>
            <tbody>
              {benchmarkModels.map(m => (
                <tr key={m.key}><td><strong>{m.name}</strong></td><td>{fmt(m.metrics.F1)}</td><td>{fmt(m.metrics.Precision)}</td><td>{fmt(m.metrics.Recall)}</td><td>{fmt(m.metrics['AUC-ROC'])}</td><td>{fmt(m.metrics.Accuracy)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <div className="two-col">
        <Panel title="F1 comparison" kicker="MODEL QUALITY">
          <ChartBars values={synthF1} labels={synthNames.map(n => n.split(' ')[0])} color="#287a5a" />
          <div className="chart-foot">Higher is better · evaluation artifact</div>
        </Panel>
        <Panel title="Interpretation" kicker="READ THE NUMBERS">
          <p className="body-copy">
            The shipped synthetic benchmark validates pipeline behavior on overlapping temporal windows. Treat it as a system check, not a production accuracy claim.
          </p>
          <div className="callout neutral">Benchmark source: <code>results/benchmark_results.json</code></div>
        </Panel>
      </div>

      {realData && (
        <>
          <div className="callout safe">
            <Info size={17} />
            <span>
              Real-data benchmark loaded ({realData.data_dir}). {' '}
              Split: {realData.split}. Test contexts: {realData.test_contexts}.
              {realData.fine_tuned && ' Fine-tuned on real training portion.'}
            </span>
          </div>
          <Panel title="Real-data results" kicker="CREDIBLE EVALUATION">
            <div className="table-wrap">
              <table>
                <thead><tr><th>Model</th><th>F1</th><th>Precision</th><th>Recall</th><th>FPR</th><th>AUC-ROC</th><th>Attack Rate</th></tr></thead>
                <tbody>
                  {realResults.map((r, i) => (
                    <tr key={i}>
                      <td><strong>{r.name}</strong></td>
                      <td>{fmt(r.metrics.F1)}</td>
                      <td>{fmt(r.metrics.Precision)}</td>
                      <td>{fmt(r.metrics.Recall)}</td>
                      <td>{fmt(r.metrics.FPR)}</td>
                      <td>{fmt(r.metrics['AUC-ROC'])}</td>
                      <td>{fmt(r.metrics.attack_rate ?? null)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="chart-foot" style={{ marginTop: 18 }}>
              Run: <code>python scripts/benchmark_real.py --data-dir &lt;path&gt;</code> to update this report.
            </div>
          </Panel>
          {realNames.length > 0 && (
            <Panel title="Real-data F1 comparison" kicker="CREDIBLE EVALUATION">
              <ChartBars values={realF1} labels={realNames.map(n => n.split(' ')[0])} color="#245b78" />
              <div className="chart-foot">Chronological split — no train/test leakage.</div>
            </Panel>
          )}
        </>
      )}
      {!realData && !realErr && (
        <Panel title="Real-data evaluation" kicker="PROVING CREDIBILITY">
          <p className="body-copy">
            Running the real-data benchmark on CIC-IDS-2018 (or a similar public dataset) will replace the
            synthetic system check with credible, no-leakage evaluation numbers.
          </p>
          <div className="callout neutral">Run: <code>python scripts/benchmark_real.py --data-dir data/raw/cic-ids-2018</code></div>
        </Panel>
      )}
    </>
  );
}

const fmt = v => typeof v === 'number' ? v.toFixed(3) : '—';