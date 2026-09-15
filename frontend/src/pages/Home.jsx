import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  BarChart3,
  FileSearch,
  FlaskConical,
  GitBranch,
  Network,
  Play,
  ShieldCheck,
  Sparkles,
  Upload,
} from "lucide-react";
import PageHeader from "../components/PageHeader";
import Panel from "../components/Panel";
import { apiRequest } from "../services/projectData";

export default function Home({ onNotice }) {
  const go = useNavigate();
  const [busy, setBusy] = useState(false);
  const runSimulation = () => {
    setBusy(true);
    apiRequest("/api/simulate", {
      method: "POST",
      body: JSON.stringify({ seed: 7, k_steps: 10 }),
    })
      .then((r) =>
        onNotice(
          `Simulation complete: peak infiltration ${(r.peak * 100).toFixed(1)}%.`,
        ),
      )
      .catch((e) => onNotice(e.message))
      .finally(() => setBusy(false));
  };
  return (
    <>
      <PageHeader
        eyebrow="CONTROL CENTER"
        title="A clearer view of what happens next."
        description="NetSeer turns network telemetry into a calm, explainable forecast of attacker progression."
        action={
          <button className="button primary" onClick={() => go("/dashboard")}>
            <Activity size={17} /> Open dashboard
          </button>
        }
      />
      <div className="hero-grid">
        <Panel className="hero-panel">
          <div className="hero-copy">
            <h2>
              Watch the network.
              <br />
              <em>Know the next move.</em>
            </h2>
            <p>
              Run a safe scan, play a simulated attack, or connect the live
              monitor when you are ready.
            </p>
            <div className="button-row">
              <button className="button primary" onClick={() => go("/scout")}>
                <Network size={17} /> Scan network
              </button>
              <button
                className="button"
                onClick={runSimulation}
                disabled={busy}
              >
                <Play size={17} /> {busy ? "Running…" : "Simulate attack"}
              </button>
            </div>
          </div>
          <div className="hero-orbit">
            <div className="orbit-ring ring-one" />
            <div className="orbit-ring ring-two" />
            <div className="orbit-core">
              <ShieldCheck size={34} />
              <span>
                NET
                <br />
                SAFE
              </span>
            </div>
          </div>
        </Panel>
        <Panel title="System posture" kicker="CURRENT STATE">
          <div className="posture">
            <div className="posture-score">
              A<span>-</span>
            </div>
            <div>
              <StatusLine
                label="Telemetry pipeline"
                value="Ready"
                tone="safe"
              />
              <StatusLine label="Model weights" value="Available" tone="safe" />
              <StatusLine label="Live capture" value="Standby" tone="muted" />
            </div>
          </div>
        </Panel>
      </div>
      <div className="section-grid">
        <ActionCard
          icon={Network}
          title="Network Scout"
          copy="Discover hosts, exposed ports and likely attack paths on your local subnet."
          action="Start a scan"
          onClick={() => go("/scout")}
        />
        <ActionCard
          icon={Activity}
          title="Live Monitor"
          copy="Keep a rolling eye on traffic and surface a forecast when behavior shifts."
          action="Open monitor"
          onClick={() => go("/live")}
        />
        <ActionCard
          icon={FileSearch}
          title="Analyze data"
          copy="Explore forecasts, attention, SHAP drivers and stage-level context."
          action="Open analysis"
          onClick={() => go("/analysis")}
        />
      </div>
      <div className="section-grid">
        <ActionCard
          icon={FlaskConical}
          title="What-if planner"
          copy="Ask the AI what happens if you actually block a port or disconnect a device."
          action="Try a counterfactual"
          onClick={() => go("/whatif")}
        />
        <ActionCard
          icon={GitBranch}
          title="Attack path map"
          copy="See exactly which machine gets attacked next, plotted live on your network graph."
          action="View attack path"
          onClick={() => go("/attack-path")}
        />
        <ActionCard
          icon={BarChart3}
          title="Benchmarks"
          copy="Check model performance on synthetic and real-world (CIC-IDS-2018) data."
          action="Open benchmarks"
          onClick={() => go("/benchmarks")}
        />
      </div>
      <Panel title="How NetSeer works" kicker="THREE LAYERS">
        <div className="steps">
          <Step
            n="01"
            title="Observe"
            copy="Flow telemetry is normalized into temporal network states."
          />
          <Step
            n="02"
            title="Forecast"
            copy="The World Model rolls forward across the ATT&CK chain."
          />
          <Step
            n="03"
            title="Explain"
            copy="Alerts carry confidence, stage and contributing features."
          />
        </div>
      </Panel>
    </>
  );
}
function StatusLine({ label, value, tone }) {
  return (
    <div className="status-line">
      <span>{label}</span>
      <strong className={tone}>{value}</strong>
    </div>
  );
}
function ActionCard({ icon: Icon, title, copy, action, onClick }) {
  return (
    <button className="action-card" onClick={onClick}>
      <span className="icon-box small">
        <Icon size={19} />
      </span>
      <strong>{title}</strong>
      <p>{copy}</p>
      <span className="text-link">
        {action}
        <ArrowRight size={15} />
      </span>
    </button>
  );
}
function Step({ n, title, copy }) {
  return (
    <div className="step">
      <span className="step-num">{n}</span>
      <div>
        <strong>{title}</strong>
        <p>{copy}</p>
      </div>
    </div>
  );
}
