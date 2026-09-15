"""Benchmark the shipped World Model on *real* public attack datasets.

Unlike ``evaluate.py`` (which randomly splits overlapping sliding windows and
can therefore leak training data into the test set), this script performs a
**chronological split**:

1. Load real flow CSVs (CIC-IDS-2018 / CTU-13 / any CIC-fidelity schema).
2. Extract flow features, then split the *flows* by time: earliest 70% become
   the training portion, the latest 30% the held-out test portion.
3. Fit the scaler and the PCA state compression **only** on the training tail,
   so the test windows are statistically unseen — overlapping-window leakage is
   impossible.
4. Score the World Model (optionally fine-tuned on the real training portion)
   against a logistic regression baseline trained on that same portion.

The F1 / precision / recall / AUC numbers written to
``results/benchmark_real.json`` are the honest "it works on real attack data"
claim for the app's Benchmarks page.

Usage::

    python scripts/benchmark_real.py --data-dir data/raw/cic-ids-2018
    python scripts/benchmark_real.py --data-dir data/raw/cic-ids-2018 --fine-tune --epochs 8

"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.csv_loader import CICIDSLoader
from src.data.feature_extractor import FeatureExtractor
from src.data.normalizer import FeatureNormalizer
from src.data.state_builder import StateBuilder, compress_states
from src.models.world_model import WorldModel
from src.models.logistic_baseline import LogisticBaseline
from src.utils.config import load_config, get_device


class SequenceDataset(Dataset):
    """Context windows with next-state + binary infiltration + stage labels."""

    def __init__(self, sequences, next_states, infiltration_labels, stage_labels):
        self.sequences = [np.asarray(s, dtype=np.float32) for s in sequences]
        self.next_states = [np.asarray(ns, dtype=np.float32) for ns in next_states]
        self.infiltration = [float(i) for i in infiltration_labels]
        self.stage = [int(s) for s in stage_labels]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.next_states[idx]),
            torch.FloatTensor([self.infiltration[idx]]),
            torch.LongTensor([self.stage[idx]]),
        )


def _pad_states(states: np.ndarray, target_dim: int) -> np.ndarray:
    """Right-pad 2D states to ``target_dim`` columns (mirrors session logic)."""
    if states.ndim != 2 or states.shape[1] >= target_dim:
        return states.astype(np.float32)
    pad = np.zeros((states.shape[0], target_dim - states.shape[1]), dtype=np.float32)
    return np.concatenate([states, pad], axis=1).astype(np.float32)


def _build_states_from_features(features, config):
    """Temporal windows from (already normalized) flow features -> states + labels."""
    builder = StateBuilder()
    states, labels = builder.build_states(
        features, window_size=int(config["data"]["window_size_seconds"])
    )
    return np.asarray(states, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _build_contexts(states, labels, context_length):
    builder = StateBuilder()
    seqs, nexts, seq_labels = builder.build_forecast_sequences(
        states, labels, context_length
    )
    return seqs, nexts, np.asarray(seq_labels, dtype=np.int64)


def evaluate_model(models, test_loader, device):
    """Run one or more models over the loader; returns dicts keyed by name."""
    out = {}
    for name, model in models.items():
        model.eval()
        y_true_inf, y_prob_inf, y_true_stage, y_pred_stage = [], [], [], []
        with torch.no_grad():
            for seq, _next, inf_label, stage_label in tqdm(test_loader, desc=f"{name} inference"):
                seq = seq.to(device)
                preds = model(seq)
                prob = preds["infiltration_prob"].cpu().numpy().flatten()
                stage_logits = preds["stage_logits"].cpu().numpy()
                stage_pred = np.argmax(stage_logits, axis=1).flatten()
                y_prob_inf.extend(prob.tolist())
                y_true_inf.extend(inf_label.numpy().flatten().tolist())
                y_pred_stage.extend(stage_pred.tolist())
                y_true_stage.extend(stage_label.numpy().flatten().tolist())
        out[name] = {
            "y_true_inf": np.asarray(y_true_inf),
            "y_prob_inf": np.asarray(y_prob_inf),
            "y_true_stage": np.asarray(y_true_stage),
            "y_pred_stage": np.asarray(y_pred_stage),
        }
    return out


def binary_metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "FPR": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "AUC-ROC": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.5,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }


def main():
    parser = argparse.ArgumentParser(description="Real-data (CIC-IDS-2018) model benchmark")
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing CIC-IDS-2018 (or similar) flow CSVs")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model_path", default=None,
                        help="Optional explicit checkpoint (default: live transformer weights)")
    parser.add_argument("--output", default="results/benchmark_real.json")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Fine-tune the World Model on the real training portion")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--test_ratio", type=float, default=0.30)
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    config = load_config(args.config).as_dict()
    device = get_device(config["training"]["device"])
    context_length = int(config["data"]["context_length"])
    target_dim = int(config["features"]["num_state_features"])

    # 1 ─ load + clean real flows
    print(f"Loading real flows from {args.data_dir} ...")
    loader = CICIDSLoader()
    raw = loader.load(args.data_dir)
    if raw.empty:
        print("No parseable flow rows found.")
        sys.exit(1)
    print(f"  {len(raw)} flows loaded")

    features = FeatureExtractor(config["features"]["flow_features"]).extract(raw)
    if "Timestamp" in features.columns:
        features = features.sort_values("Timestamp").reset_index(drop=True)
    n = len(features)
    test_cut = int(n * float(args.test_ratio))
    if test_cut < context_length + 1:
        print("Too few flows to form a held-out test window.")
        sys.exit(1)
    train_flows = features.iloc[: n - test_cut]
    test_flows = features.iloc[n - test_cut:]

    # 2 ─ chronological scaling: fit scaler on the *earliest* flows only
    normalizer = FeatureNormalizer()
    train_feat = normalizer.fit_transform(train_flows)
    test_feat = normalizer.transform(test_flows)

    # 3 ─ temporal windows + PCA fit on the training tail only
    train_states, train_labels = _build_states_from_features(train_feat, config)
    test_states, test_labels = _build_states_from_features(test_feat, config)
    if len(test_states) <= context_length:
        print("Held-out portion produced too few temporal windows.")
        sys.exit(1)

    train_states_c, pca = compress_states(train_states, target_dim)
    test_states_c, _ = compress_states(test_states, target_dim, pca=pca)
    train_states_c = _pad_states(train_states_c, target_dim)
    test_states_c = _pad_states(test_states_c, target_dim)

    train_seqs, train_nexts, train_ctx_labels = _build_contexts(train_states_c, train_labels, context_length)
    test_seqs, test_nexts, test_ctx_labels = _build_contexts(test_states_c, test_labels, context_length)
    if len(test_seqs) == 0:
        print("Held-out test produced no context sequences.")
        sys.exit(1)

    train_inf = (train_ctx_labels > 0).astype(np.float32)
    test_inf = (test_ctx_labels > 0).astype(np.float32)

    print(f"  train contexts: {len(train_seqs)}   test contexts: {len(test_seqs)}")
    print(f"  attack rate in test: {test_inf.mean():.1%}")

    test_ds = SequenceDataset(test_seqs, test_nexts, test_inf, test_ctx_labels)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    # 4 ─ World Model (shipped weights, optionally fine-tuned on the real tail)
    wm = WorldModel.from_config(config).to(device)
    if args.model_path:
        ckpt = args.model_path
    else:
        def _find(parts):
            for p in parts:
                if os.path.exists(p):
                    return p
            return ""
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        ckpt = _find([os.path.join(root, "live_transformer_model.pth"),
                      os.path.join(root, "best_live_model.pth"),
                      os.path.join(root, "transformer_model.pth"),
                      os.path.join(root, "best_model.pth")])
    if not ckpt:
        print("No model checkpoint found; nothing to evaluate.")
        sys.exit(1)
    wm.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    wm.eval()
    print(f"Loaded World Model weights: {ckpt}")

    if args.fine_tune:
        print(f"Fine-tuning on the real training context ({len(train_seqs)} samples, {args.epochs} epochs)...")
        train_ds = SequenceDataset(train_seqs, train_nexts, train_inf, train_ctx_labels)
        train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
        wm = _finetune(wm, train_loader, device, epochs=args.epochs,
                       lr=float(config["training"]["learning_rate"]))

    # 5 ─ logistic regression baseline fit on the same real training portion
    print("Training logistic baseline on the real training portion ...")
    log_model = LogisticBaseline()
    log_model.fit(train_seqs.reshape(len(train_seqs), -1), train_inf)
    log_stage_pred = log_model.predict(test_seqs.reshape(len(test_seqs), -1))
    log_prob = log_model.predict_proba(test_seqs.reshape(len(test_seqs), -1))[:, 1] \
        if log_model.model.classes_.tolist() == [0, 1] else np.zeros(len(test_ctx_labels))

    # 6 ─ score
    results = {}
    for _name, preds in evaluate_model({"WorldModel": wm}, test_loader, device).items():
        m = binary_metrics(preds["y_true_inf"], preds["y_prob_inf"])
        m["Stage Accuracy"] = float(accuracy_score(preds["y_true_stage"], preds["y_pred_stage"]))
        m["Stage F1 (macro)"] = float(f1_score(preds["y_true_stage"], preds["y_pred_stage"],
                                               average="macro", zero_division=0))
        m["labels_pos"] = int((preds["y_true_inf"] > 0).sum())
        m["labels_neg"] = int((preds["y_true_inf"] == 0).sum())
        fpr_pts, tpr_pts, _ = roc_curve(preds["y_true_inf"], preds["y_prob_inf"])
        m["fpr_curve"] = fpr_pts.tolist()
        m["tpr_curve"] = tpr_pts.tolist()
        results[_name] = {"name": "World Model (Transformer)", "metrics": m}

    log_metrics = binary_metrics(test_inf, log_prob)
    log_metrics["Stage Accuracy"] = float(accuracy_score(test_ctx_labels, log_stage_pred))
    log_metrics["Stage F1 (macro)"] = float(f1_score(test_ctx_labels, log_stage_pred,
                                                     average="macro", zero_division=0))
    results["LogisticBaseline"] = {"name": "Logistic Regression", "metrics": log_metrics}

    # 7 ─ report
    print("\n" + "=" * 62)
    print(" REAL-DATA BENCHMARK (chronological split, no leakage)")
    print("=" * 62)
    print(f"{'Model':<22}{'F1':>7}{'Prec':>7}{'Rec':>7}{'FPR':>7}{'AUC':>7}")
    print("-" * 62)
    for key, blob in results.items():
        m = blob["metrics"]
        print(f"{blob['name'][:22]:<22}{m['F1']:>7.3f}{m['Precision']:>7.3f}"
              f"{m['Recall']:>7.3f}{m['FPR']:>7.3f}{m['AUC-ROC']:>7.3f}")
    print("=" * 62)

    blob = {
        "source": "real-flow CSV benchmark",
        "data_dir": args.data_dir,
        "test_contexts": len(test_seqs),
        "test_attack_rate": float(test_inf.mean()),
        "fine_tuned": bool(args.fine_tune),
        "split": "chronological (scaler + PCA fit on earliest flows only)",
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(blob, f, indent=4)
    print(f"Saved -> {args.output}")


def _finetune(model, loader, device, epochs=8, lr=1e-3):
    """Short fine-tune of the full World Model on real context windows."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    for epoch in range(int(epochs)):
        total_loss = 0.0
        for seq, next_state, inf_label, stage_label in loader:
            seq = seq.to(device)
            next_state = next_state.to(device)
            inf_label = inf_label.to(device)
            stage_label = stage_label.to(device)
            preds = model(seq)
            loss = model.compute_loss(
                preds,
                {"next_state": next_state,
                 "infiltration": inf_label,
                 "stage": stage_label},
            )["total_loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        print(f"  epoch {epoch + 1}/{int(epochs)}  loss={total_loss / max(len(loader), 1):.4f}")
    model.eval()
    return model


if __name__ == "__main__":
    main()