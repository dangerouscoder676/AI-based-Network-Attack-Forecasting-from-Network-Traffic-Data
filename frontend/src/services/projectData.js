import benchmarkData from '../../../backend/results/benchmark_results.json';

export const STAGES = ['Benign', 'Reconnaissance', 'Initial Access', 'Lateral Movement', 'Command & Control', 'Exfiltration'];
export const benchmarkModels = Object.entries(benchmarkData).map(([key, value]) => ({ key, ...value }));
export const getBenchmarkData = () => benchmarkData;

export const backendUrl = import.meta.env.VITE_NETSEER_API_URL || '';
export async function apiRequest(path, options) {
  if (!backendUrl) throw new Error('Set VITE_NETSEER_API_URL to connect to FastAPI.');
  const response = await fetch(`${backendUrl}${path}`, { headers: { 'Content-Type': 'application/json' }, ...options });
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}
