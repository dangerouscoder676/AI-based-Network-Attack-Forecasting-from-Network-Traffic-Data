import { useEffect, useState } from 'react';
import { GitBranch, Radar } from 'lucide-react';
import PageHeader from '../components/PageHeader';
import Panel from '../components/Panel';
import StatusBadge from '../components/StatusBadge';
import { apiRequest } from '../services/projectData';

const LAYOUT = { external: 0.08, exposed: 0.35, database: 0.68, internal: 0.92 };
const SVG_W = 820;
const SVG_H = 480;
const NODE_R = 18;
const RISK_COLORS = { CRITICAL: '#ff3860', HIGH: '#f85149', MEDIUM: '#d29922', LOW: '#3fb950', NONE: '#287a5a' };

function layoutNodes(nodes) {
  const byRole = {};
  nodes.forEach(n => { const r = n.type || 'internal'; (byRole[r] ??= []).push(n); });
  const positioned = [];
  Object.entries(byRole).forEach(([role, group]) => {
    const xPct = LAYOUT[role] ?? 0.5;
    group.forEach((node, i) => {
      const yPct = 0.15 + ((i + 0.5) / Math.max(group.length, 1)) * 0.7;
      positioned.push({ ...node, x: xPct * SVG_W, y: yPct * SVG_H, role });
    });
  });
  return positioned;
}

export default function AttackPath({ onNotice }) {
  const [network, setNetwork] = useState('');
  const [busy, setBusy] = useState(false);
  const [graph, setGraph] = useState(null);

  const run = () => {
    if (!network.trim()) { onNotice?.('Enter a network CIDR first.'); return; }
    setBusy(true);
    apiRequest('/api/attack-path', { method: 'POST', body: JSON.stringify({ network, method: 'auto', timeout_s: 0.8 }) })
      .then(setGraph)
      .catch(e => onNotice?.(e.message))
      .finally(() => setBusy(false));
  };

  const nodes = layoutNodes(graph?.nodes || []);
  const edges = graph?.edges || [];
  const attackPath = new Set(graph?.attack_path || []);
  const nodeById = {};
  nodes.forEach(n => { nodeById[n.id] = n; });

  const inPath = (id) => attackPath.has(id);

  return (
    <>
      <PageHeader
        eyebrow="INTELLIGENCE / ATTACK PATH"
        title="Predict where the attack moves"
        description="Visualise the most likely path an attacker takes across your network — before it happens."
        action={
          <button className="button primary" onClick={run} disabled={busy}>
            <Radar size={16} /> {busy ? 'Analysing…' : 'Analyse attack path'}
          </button>
        }
      />
      <Panel title="Scan target" kicker="GRAPH INPUT">
        <div className="form-grid">
          <label>Network (CIDR)<input value={network} onChange={e => setNetwork(e.target.value)} placeholder="192.168.1.0/24" /></label>
          <label>Method
            <select value="auto" disabled>
              <option value="auto">Auto</option>
            </select>
          </label>
          <label>Timeout (s)<input type="number" min="0.1" max="10" step="0.1" defaultValue="0.8" disabled /></label>
        </div>
      </Panel>
      {graph && (
        <Panel title="Network topology + predicted attack path" kicker="ATTACK GRAPH">
          <div style={{ background: '#dfe4ea', borderRadius: 12, padding: 16, overflowX: 'auto' }}>
            <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ width: '100%', height: 'auto', minHeight: 340 }}>
              <defs>
                <marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                  <polygon points="0 0, 8 3, 0 6" fill="#94a3b8" />
                </marker>
                <marker id="arrow-red" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                  <polygon points="0 0, 8 3, 0 6" fill="#b33a3a" />
                </marker>
              </defs>

              {/* edges */}
              {edges.map((e, i) => {
                const from = nodeById[e.source];
                const to = nodeById[e.target];
                if (!from || !to) return null;
                const isAttackEdge = inPath(e.source) && inPath(e.target);
                return (
                  <g key={i}>
                    <line
                      x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                      stroke={isAttackEdge ? '#b33a3a' : '#94a3b8'}
                      strokeWidth={isAttackEdge ? 2.5 : 1.2}
                      strokeDasharray={isAttackEdge ? '0' : '6 4'}
                      markerEnd={isAttackEdge ? 'url(#arrow-red)' : 'url(#arrow)'}
                      opacity={isAttackEdge ? 1 : 0.45}
                    />
                    <text
                      x={(from.x + to.x) / 2}
                      y={(from.y + to.y) / 2 - 6}
                      textAnchor="middle"
                      fontSize="10"
                      fill={isAttackEdge ? '#b33a3a' : '#7f8d9a'}
                      fontFamily="'DM Sans', sans-serif"
                    >
                      {e.label}
                    </text>
                  </g>
                );
              })}

              {/* nodes */}
              {nodes.map(n => {
                const fill = RISK_COLORS[n.risk] || '#7f8d9a';
                const isAtt = n.type === 'external';
                return (
                  <g key={n.id}>
                    <circle
                      cx={n.x} cy={n.y} r={isAtt ? NODE_R + 4 : NODE_R}
                      fill={isAtt ? '#1a1a2e' : '#e4e9ef'}
                      stroke={fill}
                      strokeWidth={inPath(n.id) ? 3 : 1.5}
                      style={{ filter: inPath(n.id) ? 'drop-shadow(0 0 6px rgba(248,81,73,0.5))' : 'none' }}
                    />
                    <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize="11" fontWeight="700"
                      fill={isAtt ? '#fff' : '#18232d'}
                      fontFamily="'DM Sans', sans-serif">
                      {isAtt ? '⚡' : n.label.slice(-5)}
                    </text>
                    <text x={n.x} y={n.y + NODE_R + 14} textAnchor="middle" fontSize="10" fontWeight="600"
                      fill={fill}
                      fontFamily="'Space Grotesk', sans-serif">
                      {isAtt ? 'Attacker' : n.label}
                    </text>
                    {!isAtt && n.ports && (
                      <text x={n.x} y={n.y - NODE_R - 6} textAnchor="middle" fontSize="9"
                        fill="#7f8d9a" fontFamily="'DM Sans', sans-serif">
                        {n.services?.split(',').slice(0, 3).join(', ') || n.ports.split(',').slice(0, 3).join(',')}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
          <div className="chart-foot" style={{ marginTop: 12, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            {Object.entries(RISK_COLORS).map(([k, c]) => (
              <span key={k} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: '0.72rem' }}>
                <span style={{ width: 10, height: 10, borderRadius: '50%', background: c, display: 'inline-block' }} />{k}
              </span>
            ))}
          </div>
        </Panel>
      )}
      {graph?.summary && (
        <>
          <div className="metric-grid">
            <div className="metric-card"><div className="metric-label">Entry point</div><div className="metric-value">{graph.summary.entry_point || '—'}</div></div>
            <div className="metric-card"><div className="metric-label">Target</div><div className="metric-value">{graph.summary.target || '—'}</div></div>
            <div className="metric-card"><div className="metric-label">Path length</div><div className="metric-value">{(graph.attack_path?.length || 0) - 1}</div></div>
            <div className="metric-card"><div className="metric-label">Total hosts</div><div className="metric-value">{Math.max(0, (graph.nodes?.length || 1) - 1)}</div></div>
          </div>
          <Panel title="AI analysis" kicker="PLAIN LANGUAGE">
            <p className="body-copy">{graph.summary.description}</p>
          </Panel>
          <Panel title="Discovered assets" kicker="HOST INVENTORY">
            <div className="table-wrap">
              <table>
                <thead><tr><th>Host</th><th>Role</th><th>Open ports</th><th>Risk</th></tr></thead>
                <tbody>
                  {(graph.nodes || []).filter(n => n.type !== 'external').map(n => (
                    <tr key={n.id}>
                      <td>{n.label}</td>
                      <td><span style={{ fontSize: '0.75rem', color: '#7f8d9a' }}>{n.type}</span></td>
                      <td>{n.ports || '—'}</td>
                      <td><StatusBadge tone={n.risk === 'HIGH' || n.risk === 'CRITICAL' ? 'warn' : 'safe'}>{n.risk}</StatusBadge></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}
      {!graph && (
        <Panel title="How the attack-path map works" kicker="THREE LAYERS">
          <p className="body-copy">
            Enter an authorized CIDR and the AI will discover every device, map each one's open services, then
            predict the most likely route an attacker would take — from the entry point to the most sensitive
            host. The highlighted path shows exactly where the attack is heading, <em>before it happens</em>.
          </p>
        </Panel>
      )}
    </>
  );
}