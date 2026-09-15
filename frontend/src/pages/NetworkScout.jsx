import { useState } from 'react';
import { Globe2, Radar, Search, ShieldAlert } from 'lucide-react';
import PageHeader from '../components/PageHeader';
import Panel from '../components/Panel';
import StatusBadge from '../components/StatusBadge';
import { apiRequest } from '../services/projectData';

const METHODS = { auto: 'Auto', arp: 'ARP only', icmp: 'ICMP only' };

export default function NetworkScout({ onNotice }) {
  const [network, setNetwork] = useState('');
  const [method, setMethod] = useState('auto');
  const [timeout_s, setTimeoutS] = useState('0.8');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);

  const run = () => {
    if (!network.trim()) {
      onNotice?.('Enter a network CIDR first.');
      return;
    }
    setBusy(true);
    apiRequest('/api/scout', {
      method: 'POST',
      body: JSON.stringify({ network, method, timeout_s: Number(timeout_s) || 0.8 }),
    })
      .then(r => {
        setResult(r);
        onNotice?.(`Scan complete: ${r.summary.hosts} hosts, ${r.summary.open_ports} open ports.`);
      })
      .catch(e => onNotice?.(e.message))
      .finally(() => setBusy(false));
  };

  return (
    <>
      <PageHeader
        eyebrow="OPERATIONS / NETWORK SCOUT"
        title="Map the attack surface"
        description="Discover hosts and exposed services on a controlled subnet."
        action={
          <button className="button primary" onClick={run} disabled={busy}>
            <Radar size={16} /> {busy ? 'Scanning…' : 'Scan network'}
          </button>
        }
      />
      <Panel title="Scan target" kicker="CONTROLLED DISCOVERY">
        <div className="form-grid">
          <label>Network (CIDR)<input value={network} onChange={e => setNetwork(e.target.value)} placeholder="192.168.1.0/24" /></label>
          <label>Method
            <select value={method} onChange={e => setMethod(e.target.value)}>
              {Object.entries(METHODS).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </label>
          <label>Timeout (s)<input type="number" min="0.1" max="10" step="0.1" value={timeout_s} onChange={e => setTimeoutS(e.target.value)} /></label>
        </div>
        <div className="scan-note"><Globe2 size={18} /><span><strong>Safe by design.</strong> The existing scanner uses ARP/ICMP discovery and a curated TCP connect list.</span></div>
      </Panel>
      <div className="metric-grid">
        <div className="metric-card"><div className="metric-label">Hosts found</div><div className="metric-value">{result?.summary.hosts ?? '—'}</div></div>
        <div className="metric-card"><div className="metric-label">Open services</div><div className="metric-value">{result?.summary.open_ports ?? '—'}</div></div>
        <div className="metric-card"><div className="metric-label">Exposed hosts</div><div className="metric-value">{result?.summary.exposed_hosts ?? '—'}</div></div>
        <div className="metric-card"><div className="metric-label">Attack surface</div><div className="metric-value">{result?.summary.level ?? '—'}</div></div>
      </div>
      <Panel title="Host inventory" kicker="DISCOVERED ASSETS">
        {result ? (
          <div className="table-wrap">
            <table>
              <thead><tr><th>IP</th><th>Device</th><th>Ports</th><th>Risk</th></tr></thead>
              <tbody>
                {result.hosts.map(h => (
                  <tr key={h.ip}>
                    <td>{h.ip}</td>
                    <td>{h.vendor || h.os_guess || 'unknown'}</td>
                    <td>{h.open_ports || 'None'}</td>
                    <td><StatusBadge tone={h.risk_label === 'HIGH' || h.risk_label === 'CRITICAL' ? 'warn' : 'safe'}>{h.risk_label}</StatusBadge></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state large"><Search size={27} /><strong>No scan results yet</strong><p>Enter an authorized CIDR and run the scan.</p></div>
        )}
      </Panel>
      <div className="callout neutral"><ShieldAlert size={17} /> <span>Uses existing <code>discover_hosts()</code> / <code>scan_ports()</code> logic; large-CIDR sweeps are refused for safety.</span></div>
    </>
  );
}