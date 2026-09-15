"""HTTP adapter: React/Vite -> FastAPI -> existing NetSeer functions."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from session import (
    get_config,
    load_benchmarks,
    load_benchmarks_real,
    load_model,
    load_synthetic_data,
    run_attention,
    run_rollout,
    run_shap,
)
from src.data.live_capture import LiveTrafficEngine, list_interfaces  # noqa: E402
from src.data.network_scanner import (  # noqa: E402
    COMMON_PORTS,
    discover_hosts,
    guess_local_network,
    host_risk,
    network_risk_summary,
    scan_ports,
)
from src.prediction.attack_simulator import build_phase_plan, simulate_attack, simulate_intervention  # noqa: E402
from src.prediction.attack_path import build_attack_path  # noqa: E402

app = FastAPI(title="NetSeer API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_live_engine: Optional[LiveTrafficEngine] = None


class ForecastRequest(BaseModel):
    context_index: int = Field(0, ge=0)
    k_steps: int = Field(10, ge=1, le=30)
    model_type: str = Field("transformer", pattern="^(transformer|lstm)$")
    model_path: Optional[str] = None


class ScoutRequest(BaseModel):
    network: str = Field(..., min_length=7, max_length=64)
    method: str = Field("auto", pattern="^(auto|arp|icmp)$")
    timeout_s: float = Field(0.8, gt=0, le=10)
    ports: Optional[list[int]] = None


class AttackPathRequest(BaseModel):
    network: str = Field(..., min_length=7, max_length=64)
    method: str = Field("auto", pattern="^(auto|arp|icmp)$")
    timeout_s: float = Field(0.8, gt=0, le=10)
    ports: Optional[list[int]] = None


class LiveStartRequest(BaseModel):
    interface: Optional[str] = None


class SimulationRequest(BaseModel):
    seed: int = Field(7, ge=0, le=99)
    k_steps: int = Field(10, ge=1, le=30)


class WhatIfRequest(BaseModel):
    seed: int = Field(7, ge=0, le=99)
    k_steps: int = Field(10, ge=1, le=30)
    intervention: dict = Field(default_factory=lambda: {"type": "do_nothing"})


def _json(value: Any) -> Any:
    """Convert numpy/Pandas values from existing functions to JSON-safe values."""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, pd.DataFrame):
        return value.replace({np.nan: None}).to_dict(orient="records")
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _forecast_payload(request: ForecastRequest, include_explainability: bool = False) -> dict:
    data = load_synthetic_data(int(get_config()["data"]["context_length"]))
    if not data or data[0] is None:
        raise HTTPException(404, "Synthetic data is unavailable")
    contexts, seq_stages, feature_names, _states, _timeline = data
    if request.context_index >= len(contexts):
        raise HTTPException(422, f"context_index must be below {len(contexts)}")

    model = load_model(request.model_type, request.model_path, family="offline")
    if model is None:
        raise HTTPException(503, "No compatible model checkpoint was found")
    context = contexts[request.context_index]
    rollout = run_rollout(model, context, request.k_steps)
    result = {
        "context_index": request.context_index,
        "context_count": len(contexts),
        "feature_names": feature_names,
        "ground_truth_stage": int(seq_stages[request.context_index]),
        "probabilities": rollout["probabilities"],
        "stages": rollout["stages"],
        "stage_names": rollout["stage_names"],
        "confidence": rollout["confidence"],
        "predicted_states": rollout["predicted_states"],
    }
    if include_explainability:
        try:
            attention = run_attention(model, context)
            result["attention"] = {"temporal_attention": attention.get("temporal_attention", [])}
        except Exception as exc:  # explainability is optional to the forecast
            result["attention_error"] = str(exc)
        try:
            shap_result = run_shap(model, contexts, context, feature_names)
            result["shap"] = {"top_features": shap_result.get("top_features", [])}
        except Exception as exc:
            result["shap_error"] = str(exc)
    return _json(result)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "service": "netseer-api"}


@app.get("/api/config")
def config() -> dict:
    cfg = get_config()
    return _json({"context_length": cfg["data"]["context_length"], "k_steps": cfg["prediction"]["k_steps"], "stages": ["Benign", "Reconnaissance", "Initial Access", "Lateral Movement", "Command & Control", "Exfiltration"], "common_ports": COMMON_PORTS})


@app.get("/api/benchmarks")
def benchmarks() -> dict:
    result = load_benchmarks()
    if result is None:
        raise HTTPException(404, "Benchmark results are unavailable")
    return _json(result)


@app.get("/api/benchmarks/real")
def benchmarks_real() -> dict:
    """Real-data benchmark results (run ``scripts/benchmark_real.py`` first)."""
    result = load_benchmarks_real()
    if result is None:
        raise HTTPException(404, "Real-data benchmark not yet available. Run: python scripts/benchmark_real.py --data-dir <path>")
    return _json(result)


@app.post("/api/forecast")
def forecast(request: ForecastRequest) -> dict:
    return _forecast_payload(request)


@app.post("/api/analysis")
def analysis(request: ForecastRequest) -> dict:
    return _forecast_payload(request, include_explainability=True)


@app.post("/api/simulate")
def simulate(request: SimulationRequest) -> dict:
    model = load_model("transformer", family="live")
    if model is None:
        raise HTTPException(503, "No live model checkpoint was found")
    try:
        return _json(simulate_attack(model, build_phase_plan(), seed=request.seed, k_steps=request.k_steps))
    except Exception as exc:
        raise HTTPException(502, f"Simulation failed: {exc}") from exc


@app.post("/api/whatif")
def whatif(request: WhatIfRequest) -> dict:
    """Counterfactual defence planner: baseline vs a chosen intervention."""
    model = load_model("transformer", family="live")
    if model is None:
        raise HTTPException(503, "No live model checkpoint was found")
    try:
        return _json(simulate_intervention(model, request.intervention,
                                           seed=request.seed, k_steps=request.k_steps))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"What-if simulation failed: {exc}") from exc


@app.post("/api/scout")
def scout(request: ScoutRequest) -> dict:
    try:
        hosts = discover_hosts(request.network, prefer=request.method, timeout=request.timeout_s)
        scan_ports(hosts, ports=request.ports, timeout_s=request.timeout_s)
        inventory = host_risk(hosts)
        summary = network_risk_summary(inventory, total_addressed=len(hosts))
        return _json({"network": request.network, "summary": summary, "hosts": inventory})
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Network scan failed: {exc}") from exc


@app.post("/api/attack-path")
def attack_path(request: AttackPathRequest) -> dict:
    """Compute a predicted attack-path graph over the discovered network."""
    try:
        hosts = discover_hosts(request.network, prefer=request.method, timeout=request.timeout_s)
        scan_ports(hosts, ports=request.ports, timeout_s=request.timeout_s)
        inventory = host_risk(hosts)
        graph = build_attack_path(inventory)
        return _json(graph)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Attack path analysis failed: {exc}") from exc


@app.get("/api/live/interfaces")
def live_interfaces() -> dict:
    return {"interfaces": list_interfaces()}


@app.get("/api/live/status")
def live_status() -> dict:
    return _json(_live_engine.status() if _live_engine else {"running": False, "mode": None, "source": "", "packets": 0, "flows": 0, "error": ""})


@app.post("/api/live/start")
def live_start(request: LiveStartRequest) -> dict:
    global _live_engine
    if _live_engine and _live_engine.running:
        raise HTTPException(409, "Live capture is already running")
    _live_engine = LiveTrafficEngine()
    try:
        _live_engine.start_interface(request.interface)
        return _json(_live_engine.status())
    except Exception as exc:
        _live_engine = None
        raise HTTPException(502, f"Live capture failed to start: {exc}") from exc


@app.post("/api/live/stop")
def live_stop() -> dict:
    if _live_engine:
        _live_engine.stop()
    return live_status()
