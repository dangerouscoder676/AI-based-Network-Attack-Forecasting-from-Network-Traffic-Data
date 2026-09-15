"""
Shared data-loading, model-loading and inference helpers for NetSeer pages.
"""

import os
import json
import numpy as np
import pandas as pd

from functools import lru_cache
import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SYNTHETIC_NPZ = os.path.join(PROJECT_ROOT, "data", "synthetic", "synthetic_data.npz")
BENCHMARK_JSON = os.path.join(PROJECT_ROOT, "results", "benchmark_results.json")
BENCHMARK_REAL_JSON = os.path.join(PROJECT_ROOT, "results", "benchmark_real.json")

STAGE_NAMES = [
    "Benign",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Exfiltration",
]

SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
_DATA_CACHE = {}
logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────────────
def get_config():
    from src.utils.config import load_config
    return load_config(os.path.join(PROJECT_ROOT, "config.yaml"))


# ─── Model loading (cached) ──────────────────────────────────────────────────
def _model_candidates(model_type: str, model_path: str, family: str):
    """Ordered checkpoint candidates for ``model_type`` + ``family``.

    family "live" prefers checkpoints trained on :class:`LiveStateMapper` states
    (``scripts/train_live.py``); family "offline" uses the PCA-pipeline weights.
    """
    root = os.path.join(PROJECT_ROOT, "models")
    names = [
        f"live_{model_type}_model.pth",
        f"{model_type}_live_model.pth",
        "best_live_model.pth",
        f"{model_type}_model.pth",
        "best_model.pth",
    ]
    if family == "offline":
        names = [f"{model_type}_model.pth", "best_model.pth"]
    return [p for p in ([model_path] + [os.path.join(root, n) for n in names]) if p]


def available_live_models() -> list:
    """UI helper: labels for model checkpoints usable in the live path."""
    from src.utils.config import load_config

    cfg = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))
    dtype = cfg["model"].get("type", "transformer")
    out = [("Live World Model (temporal transformer)", dtype, "live")]
    if os.path.exists(os.path.join(PROJECT_ROOT, "models", "lstm_live_model.pth")):
        out.append(("LSTM Baseline (live-trained)", "lstm", "live"))
    elif os.path.exists(os.path.join(PROJECT_ROOT, "models", "lstm_model.pth")):
        out.append(("LSTM Baseline (offline weights — not live-trained)", "lstm", "live"))
    if not os.path.exists(os.path.join(PROJECT_ROOT, "models", "best_live_model.pth")):
        out.append(("Offline transformer (PCA states — not recommended for live)", dtype, "offline"))
    return out


@lru_cache(maxsize=8)
def load_model(model_type: str = "transformer", model_path: str = None, family: str = "offline"):
    """Load a WorldModel checkpoint. ``family`` picks live vs offline weights.

    Returns None when no weights exist; callers surface the error.
    """
    from src.models.world_model import WorldModel
    from src.utils.config import load_config

    cfg = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))
    config_dict = cfg.as_dict()
    config_dict["model"]["type"] = model_type
    model = WorldModel.from_config(config_dict)

    candidate = None
    for path in _model_candidates(model_type, model_path, family):
        if path and os.path.exists(path):
            candidate = path
            break
    if candidate is None:
        return None

    model.load_state_dict(torch_load(candidate))
    model.eval()
    return model


def torch_load(path: str):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


# ─── Data loading ────────────────────────────────────────────────────────────
def _build_contexts(states: np.ndarray, stages: np.ndarray, context_length: int):
    """Build [N, context_length, D] contexts with staged next-state labels."""
    from src.data.state_builder import StateBuilder

    builder = StateBuilder()
    contexts, _, seq_stages = builder.build_forecast_sequences(states, stages, context_length)
    return contexts, seq_stages


def get_data(settings: dict, context_length: int):
    """Load data once based on the sidebar settings. Returns
    (contexts, seq_stages, feature_names, states, state_stages) or None."""
    source = settings["data_source"]
    uploaded = settings.get("uploaded_file")
    data_dir = settings.get("data_dir")

    if source == "Upload File" and uploaded is not None:
        key = ("upload", uploaded.file_id, uploaded.name)
        loader = lambda: load_uploaded_data(uploaded, context_length)
    elif source == "Synthetic Demo":
        key = ("synthetic",)
        loader = lambda: load_synthetic_data(context_length)
    elif source == "Load from Directory" and data_dir:
        key = ("dir", os.path.abspath(data_dir))
        loader = lambda: load_directory_data(data_dir, context_length)
    else:
        return None

    cache_key = ("netseer_data",) + key
    if cache_key not in _DATA_CACHE:
        _DATA_CACHE[cache_key] = loader()
    return _DATA_CACHE[cache_key]


def load_directory_data(data_dir: str, context_length: int):
    """Load and concatenate all CSVs found under a directory."""
    import glob

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")
        return None
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
        return None

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            logger.warning("Skipped %s: %s", f, e)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    return _load_df(df, context_length, source=data_dir)


def load_synthetic_data(context_length: int):
    """Load the synthetic demo npz; returns (contexts, seq_stages, feature_names, timeline_stages)."""
    if not os.path.exists(SYNTHETIC_NPZ):
        raise FileNotFoundError("Synthetic data not found. Run: python scripts/generate_synthetic.py")
        return None, None, None, None

    d = np.load(SYNTHETIC_NPZ)
    if "sequences" in d and "stages" in d:
        contexts = d["sequences"]
        seq_stages = np.asarray(d["stages"])
        feature_names = list(d["feature_names"]) if "feature_names" in d else None
        timeline_stages = np.asarray(d["state_stages"]) if "state_stages" in d else seq_stages
        states = np.asarray(d["states"]) if "states" in d else None
    else:
        contexts = d["data"][:, :-1, :]
        seq_stages = np.asarray(d["labels"])
        feature_names = None
        timeline_stages = seq_stages
        states = None

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(contexts.shape[2])]
    return contexts, seq_stages.astype(int), feature_names, states, timeline_stages.astype(int)


def detect_schema(df: pd.DataFrame) -> str:
    """Return 'cic', 'states', or None depending on the dataframe schema."""
    if "Flow Duration" in df.columns or "Tot Fwd Pkts" in df.columns:
        return "cic"
    numeric = df.select_dtypes(include=[np.number]).columns
    if len(numeric) >= 60:
        return "states"
    return None


def load_uploaded_data(uploaded_file, context_length: int):
    """Parse an uploaded CSV/PCAP into (contexts, seq_stages, feature_names, states, state_stages)."""
    name = (getattr(uploaded_file, "name", None) or "").lower()

    if name.endswith(".pcap") or name.endswith(".pcapng"):
        import tempfile

        bytes_data = uploaded_file.getvalue()
        ext = ".pcapng" if name.endswith(".pcapng") else ".pcap"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(bytes_data)
            tmp_path = tmp.name
        try:
            return _load_pcap(tmp_path, context_length)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    df = pd.read_csv(uploaded_file, low_memory=False)
    return _load_df(df, context_length, source=getattr(uploaded_file, "name", "upload"))


def _load_pcap(pcap_path, context_length: int):
    """Parse PCAP into flows then run the standard flow pipeline."""
    from src.data.pcap_parser import PcapParser

    parser = PcapParser()
    flows = parser.parse(pcap_path)
    if flows.empty:
        raise ValueError("PCAP contained no parseable IP flows.")
        return None, None, None, None, None
    return _flow_pipeline(flows, context_length, "PCAP")


def _load_df(df: pd.DataFrame, context_length: int, source="csv"):
    schema = detect_schema(df)
    if schema == "cic":
        return _flow_pipeline(df, context_length, source)
    if schema == "states":
        return _state_schema_pipeline(df, context_length)
    raise ValueError("Unrecognized CSV format. Use a CIC-IDS-2018 flow file or a NetSeer state file.")
    return None, None, None, None, None


def _flow_pipeline(df: pd.DataFrame, context_length: int, source):
    """Run CIC-IDS flow CSV / PCAP-flows through feature extraction + state building."""
    from src.data.csv_loader import CICIDSLoader
    from src.data.feature_extractor import FeatureExtractor
    from src.data.normalizer import FeatureNormalizer
    from src.data.state_builder import StateBuilder, compress_states
    from src.utils.config import load_config

    cfg = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))

    loader = CICIDSLoader()
    cleaned = loader._clean_columns(df)
    cleaned = loader._clean_data(cleaned)
    cleaned = loader._parse_time(cleaned)
    cleaned = loader._map_labels(cleaned)

    if cleaned.empty or "Timestamp" not in cleaned.columns:
        raise ValueError("No temporal information available. Provide a Timestamp column at 0.0 (seconds).")
        return None, None, None, None, None

    extractor = FeatureExtractor(cfg["features"]["flow_features"])
    features = extractor.extract(cleaned)

    normalizer = FeatureNormalizer()
    try:
        features = normalizer.fit_transform(features)
    except Exception as e:
        logger.warning("Normalization skipped (%s).", e)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    builder = StateBuilder()
    states, state_stages = builder.build_states(features, window_size=cfg["data"]["window_size_seconds"])
    if len(states) == 0:
        raise ValueError("Not enough flows to build temporal windows (need more than one window).")
        return None, None, None, None, None

    states, _ = compress_states(states, cfg["features"]["num_state_features"])
    if states.shape[1] < int(cfg["features"]["num_state_features"]):
        logger.warning("Only %s PCA components were recoverable from %s windows; zero-padded.", states.shape[1], len(states))
        pad_w = int(cfg["features"]["num_state_features"]) - states.shape[1]
        pad = np.zeros((states.shape[0], pad_w), dtype=np.float32)
        states = np.concatenate([states, pad], axis=1)
    contexts, seq_stages = _build_contexts(states, state_stages, context_length)

    feature_names = [f"feature_{i}" for i in range(states.shape[1])]
    return contexts, seq_stages, feature_names, states, state_stages


def _state_schema_pipeline(df: pd.DataFrame, context_length: int):
    """Handle a 'states' CSV (rows are already normalized state vectors)."""
    from src.utils.config import load_config

    cfg = load_config(os.path.join(PROJECT_ROOT, "config.yaml"))
    target_dim = int(cfg["features"]["num_state_features"])

    numeric_df = df.select_dtypes(include=[np.number])
    stages = None
    if "stage" in df.columns:
        stages = df["stage"].fillna(0).astype(int).values
    elif "label" in df.columns:
        stages = (df["label"].fillna(0).astype(int) > 0).astype(int)
    else:
        stages = np.zeros(len(df), dtype=int)

    state_cols = [c for c in numeric_df.columns if c.lower() not in ("stage", "label")]
    states = numeric_df[state_cols].values.astype(np.float32)
    if states.shape[1] > target_dim:
        states = states[:, :target_dim]
    elif states.shape[1] < target_dim:
        pad = np.zeros((states.shape[0], target_dim - states.shape[1]), dtype=np.float32)
        states = np.concatenate([states, pad], axis=1)

    if len(states) <= context_length:
        raise ValueError(f"Need more than {context_length} rows to build a context window.")
        return None, None, None, None, None

    contexts, seq_stages = _build_contexts(states, stages, context_length)
    feature_names = [f"feature_{i}" for i in range(target_dim)]
    return contexts, seq_stages, feature_names, states, stages


# ─── Inference ───────────────────────────────────────────────────────────────
def run_rollout(model, context: np.ndarray, k_steps: int) -> dict:
    """Run K-step rollout on a single context window."""
    import torch
    from src.prediction.rollout_engine import RolloutEngine

    if model is None:
        return None
    engine = RolloutEngine(model)
    tensor = torch.tensor(context[None, :, :], dtype=torch.float32)
    return engine.rollout(tensor, k_steps=k_steps)


def run_attention(model, context: np.ndarray) -> dict:
    """Extract temporal attention weights for a context window."""
    import torch
    from src.explainability.attention_explainer import AttentionExplainer

    if model is None:
        return None
    explainer = AttentionExplainer(model)
    tensor = torch.tensor(context[None, :, :], dtype=torch.float32)
    return explainer.explain(tensor)


def run_shap(model, contexts: np.ndarray, context: np.ndarray, feature_names, method="gradient", background_n=8):
    """Compute SHAP attributions for the given context."""
    from src.explainability.shap_explainer import ShapExplainer

    if model is None:
        return None
    background = contexts[:background_n]
    explainer = ShapExplainer(model, background)
    result = explainer.explain(context[None, :, :])
    sv = np.asarray(result["shap_values"])
    top = explainer.get_top_k_features(sv, feature_names, k=10)
    return {"explainer": explainer, "result": result, "top_features": top}


def generate_alerts(rollout: dict, feature_names, feature_values, threshold_override=None):
    """Generate alert objects from a rollout result."""
    from src.prediction.alert_generator import AlertGenerator

    generator = AlertGenerator()
    alerts = generator.generate_alerts(
        rollout,
        feature_names=feature_names,
        feature_values=feature_values,
        timestamp=None,
    )
    # Apply sidebar threshold override
    if threshold_override is not None:
        alerts = [a for a in alerts if a.probability >= threshold_override]
    return alerts


# ─── Benchmarks ──────────────────────────────────────────────────────────────
def load_benchmarks() -> dict:
    if os.path.exists(BENCHMARK_JSON):
        with open(BENCHMARK_JSON, "r") as f:
            return json.load(f)
    return None


def load_benchmarks_real() -> dict | None:
    """Load real-data benchmark results (run ``scripts/benchmark_real.py`` first)."""
    if os.path.exists(BENCHMARK_REAL_JSON):
        with open(BENCHMARK_REAL_JSON, "r") as f:
            return json.load(f)
    return None


def load_history(model_type: str) -> dict:
    path = os.path.join(PROJECT_ROOT, "models", f"history_{model_type}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None
