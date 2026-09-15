import { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './layouts/Layout';
import Home from './pages/Home';
import LiveMonitor from './pages/LiveMonitor';
import NetworkScout from './pages/NetworkScout';
import Dashboard from './pages/Dashboard';
import Analysis from './pages/Analysis';
import Benchmarks from './pages/Benchmarks';
import WhatIf from './pages/WhatIf';
import AttackPath from './pages/AttackPath';
import About from './pages/About';

export default function App() {
  const [notice, setNotice] = useState(null);
  return <Layout notice={notice} clearNotice={() => setNotice(null)}>
    <Routes>
      <Route path="/" element={<Home onNotice={setNotice} />} />
      <Route path="/live" element={<LiveMonitor onNotice={setNotice} />} />
      <Route path="/scout" element={<NetworkScout onNotice={setNotice} />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/analysis" element={<Analysis />} />
      <Route path="/whatif" element={<WhatIf onNotice={setNotice} />} />
      <Route path="/attack-path" element={<AttackPath onNotice={setNotice} />} />
      <Route path="/benchmarks" element={<Benchmarks />} />
      <Route path="/about" element={<About />} />
    </Routes>
  </Layout>;
}
