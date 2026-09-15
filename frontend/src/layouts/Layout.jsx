import { NavLink } from 'react-router-dom';
import { Activity, BarChart3, BookOpen, BrainCircuit, ChevronRight, GitBranch, Info, LayoutDashboard, Network, Search, ShieldCheck, TestTubes, X } from 'lucide-react';

const nav = [
  ['/', 'Home', ShieldCheck], ['/', 'Operations', null], ['/live', 'Live Monitor', Activity], ['/scout', 'Network Scout', Search],
  ['/', 'Intelligence', null], ['/dashboard', 'Dashboard', LayoutDashboard], ['/analysis', 'Analysis', BrainCircuit],
  ['/whatif', 'What-if Planner', TestTubes], ['/attack-path', 'Attack Path', GitBranch], ['/benchmarks', 'Benchmarks', BarChart3],
  ['/', 'Reference', null], ['/about', 'About', Info],
];

export default function Layout({ children, notice, clearNotice }) {
  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark"><Network size={20}/></span><span>NetSeer</span></div>
      <div className="brand-sub">NETWORK DEFENSE / 01</div>
      <nav>{nav.map(([to, label, Icon], i) => Icon ? <NavLink key={label} to={to} className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}><Icon size={17}/><span>{label}</span>{label === 'Home' && <ChevronRight size={14} className="nav-arrow"/>}</NavLink> : <div key={`${label}-${i}`} className="nav-label">{label}</div>)}</nav>
      <div className="sidebar-status"><span className="status-dot"/> <span>Local workspace</span><span className="status-pill">READY</span></div>
    </aside>
    <main className="main"><header className="topbar"><div><span className="eyebrow">SECURITY OPERATIONS CENTER</span><span className="topbar-title">Network attack forecasting</span></div><div className="topbar-right"><span className="sync-dot"/> Offline-first runtime <span className="avatar">NS</span></div></header>
      <div className="content">{notice && <div className="notice"><Info size={17}/><span>{notice}</span><button onClick={clearNotice}><X size={15}/></button></div>}{children}</div>
    </main>
  </div>;
}
