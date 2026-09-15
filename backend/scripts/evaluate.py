import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.csv_loader import CICIDSLoader
from src.data.feature_extractor import FeatureExtractor
from src.data.normalizer import FeatureNormalizer
from src.data.state_builder import StateBuilder, compress_states
from src.models.world_model import WorldModel
from src.models.logistic_baseline import LogisticBaseline
from src.utils.config import load_config, get_device
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, accuracy_score, roc_curve
)

class TrafficDataset(Dataset):
    def __init__(self, sequences, next_states, infiltration_labels, stage_labels):
        self.sequences = sequences
        self.next_states = next_states
        self.infiltration_labels = infiltration_labels
        self.stage_labels = stage_labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(np.asarray(self.sequences[idx], dtype=np.float32)),
            torch.FloatTensor(np.asarray(self.next_states[idx], dtype=np.float32)),
            torch.FloatTensor([self.infiltration_labels[idx]]),
            torch.LongTensor([self.stage_labels[idx]])
        )


def load_data(data_source, data_dir, config):
    if data_source == 'csv':
        loader = CICIDSLoader()
        raw_data = loader.load(data_dir)
        extractor = FeatureExtractor(config['features']['flow_features'])
        features = extractor.extract(raw_data)
        normalizer = FeatureNormalizer()
        features = normalizer.fit_transform(features)
        builder = StateBuilder()
        states, stage_labels = builder.build_states(features, window_size=config['data']['window_size_seconds'])
        states, _ = compress_states(states, int(config['features']['num_state_features']))
        sequences, next_states, seq_stages = builder.build_forecast_sequences(
            states, stage_labels, context_length=int(config['data']['context_length']))
    else:
        data_path = os.path.join(data_dir, "synthetic_data.npz")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Synthetic data not found at {data_path}. Run generate_synthetic.py first.")
        loaded = np.load(data_path)
        if 'sequences' in loaded:
            sequences = loaded['sequences']
            next_states = loaded['next_states'] if 'next_states' in loaded else sequences[:, -1, :]
            seq_stages = loaded['stages']
        else:
            sequences = loaded['data'][:, :-1, :]
            next_states = loaded['data'][:, -1, :]
            seq_stages = loaded['stages']
    return sequences, next_states, seq_stages


def binary_metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "FPR": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "AUC-ROC": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.5,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }


def evaluate_world_model(model, loader, device):
    """Run the World Model over the test loader, returning binary + stage predictions."""
    y_true_inf, y_prob_inf = [], []
    y_true_stage, y_pred_stage = [], []
    feature_dim = model.input_dim

    with torch.no_grad():
        for seq, next_state, inf_label, stage_label in tqdm(loader, desc="WorldModel inference"):
            seq = seq.to(device)
            outputs = model(seq)
            prob = outputs['infiltration_prob'].cpu().numpy().flatten()
            stage_logits = outputs['stage_logits'].cpu().numpy()
            stage_pred = np.argmax(stage_logits, axis=1).flatten()

            y_prob_inf.extend(prob.tolist())
            y_true_inf.extend(inf_label.numpy().flatten().tolist())
            y_pred_stage.extend(stage_pred.tolist())
            y_true_stage.extend(stage_label.numpy().flatten().tolist())

    return np.array(y_true_inf), np.array(y_prob_inf), np.array(y_true_stage), np.array(y_pred_stage)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the AI Network Attack Forecasting models")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                        help='Path to trained WorldModel weights')
    parser.add_argument('--data_source', type=str, choices=['synthetic', 'csv'], default='synthetic',
                        help='Data source')
    parser.add_argument('--data_dir', type=str, default='data/synthetic', help='Directory containing data')
    parser.add_argument('--output_dir', type=str, default='results/', help='Output directory for evaluation results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = load_config(args.config).as_dict()
    device = get_device(config['training']['device'])

    # 1. Load data
    print(f"Loading {args.data_source} data for evaluation...")
    sequences, next_states, stages = load_data(args.data_source, args.data_dir, config)
    next_states = next_states
    stages = np.asarray(stages)

    if len(sequences) == 0:
        raise ValueError("No sequences available for evaluation.")

    # Random split with fixed seed so ground truth covers every attack phase
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(sequences))
    test_start = int(len(sequences) * (1 - config['data']['test_split']))
    test_idx = indices[test_start:]
    train_idx = indices[:test_start]

    test_seq = sequences[test_idx]
    test_next = next_states[test_idx]
    test_labels = (stages[test_idx] > 0).astype(np.float32)
    test_stages = stages[test_idx]

    test_dataset = TrafficDataset(test_seq, test_next, test_labels, test_stages)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # 2. Load trained WorldModel (transformer)
    print("Loading WorldModel...")
    world_model = WorldModel.from_config(config).to(device)
    world_model.load_state_dict(torch.load(args.model_path, map_location=device))
    world_model.eval()

    y_true_inf, y_prob_inf, y_true_stage, y_pred_stage = evaluate_world_model(world_model, test_loader, device)
    wm_inf = binary_metrics(y_true_inf, y_prob_inf)
    wm_stage = {
        "Stage Accuracy": float(accuracy_score(y_true_stage, y_pred_stage)),
        "Stage F1 (macro)": float(f1_score(y_true_stage, y_pred_stage, average='macro', zero_division=0)),
    }
    fpr_pts, tpr_pts, _ = roc_curve(y_true_inf, y_prob_inf)
    wm_inf["fpr_curve"] = fpr_pts.tolist()
    wm_inf["tpr_curve"] = tpr_pts.tolist()
    wm_inf["stage_cm"] = confusion_matrix(y_true_stage, y_pred_stage).tolist()
    wm_inf["stage_labels"] = [int(c) for c in sorted(set(stages))]
    wm_inf.update(wm_stage)
    print(f"WorldModel  -> {wm_inf}")

    # 3. Logistic regression baseline
    print("Training + evaluating LogisticBaseline...")
    train_seq = sequences[train_idx].reshape(len(train_idx), -1)
    train_stage = stages[train_idx]
    test_flat = test_seq.reshape(len(test_seq), -1)

    log_model = LogisticBaseline()
    log_model.fit(train_seq, train_stage)

    stage_proba = log_model.predict_proba(test_flat)
    log_stage_pred = log_model.predict(test_flat)
    # predict_proba only emits columns for classes seen in the training split,
    # so align against the model's own classes_ instead of np.unique(stages)
    # (which can include classes absent from training -> silent misalignment).
    classes = getattr(log_model.model, 'classes_', None)
    if classes is None:
        classes = np.unique(stages)
    if stage_proba.shape[1] != len(classes):
        raise ValueError(
            "Logistic predict_proba columns do not match the training classes; "
            "re-train the baseline on a split that covers every stage."
        )
    log_prob = np.zeros(len(test_flat))
    for i, c in enumerate(classes):
        if c > 0:
            log_prob += stage_proba[:, i]
    y_log_inf = (test_stages > 0).astype(int)

    log_inf = binary_metrics(y_log_inf, log_prob)
    fpr_pts, tpr_pts, _ = roc_curve(y_log_inf, log_prob)
    log_inf["fpr_curve"] = fpr_pts.tolist()
    log_inf["tpr_curve"] = tpr_pts.tolist()
    # The confusion matrix must compare multiclass ground truth against
    # multiclass predictions (y_log_inf is the *binary* attack flag).
    log_inf["stage_cm"] = confusion_matrix(test_stages, log_stage_pred).tolist()
    log_inf["stage_labels"] = [int(c) for c in sorted(set(stages))]
    log_inf["Stage Accuracy"] = float(accuracy_score(test_stages, log_stage_pred))
    log_inf["Stage F1 (macro)"] = float(f1_score(test_stages, log_stage_pred, average='macro', zero_division=0))
    print(f"Logistic     -> {log_inf}")

    # 4. LSTM baseline (WorldModel with LSTM dynamics)
    lstm_metrics = None
    lstm_path = os.path.join(os.path.dirname(args.model_path), 'lstm_model.pth')
    lstm_config = dict(config)
    lstm_config['model'] = dict(config['model'])
    lstm_config['model']['type'] = 'lstm'
    if os.path.exists(lstm_path):
        print("Evaluating LSTM baseline (lstm_model.pth)...")
        lstm_model = WorldModel.from_config(lstm_config).to(device)
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
        lstm_model.eval()
        y_l_inf, y_l_prob, y_l_stage, y_l_stage_pred = evaluate_world_model(lstm_model, test_loader, device)
        lstm_metrics = binary_metrics(y_l_inf, y_l_prob)
        fpr_pts, tpr_pts, _ = roc_curve(y_l_inf, y_l_prob)
        lstm_metrics["fpr_curve"] = fpr_pts.tolist()
        lstm_metrics["tpr_curve"] = tpr_pts.tolist()
        lstm_metrics["stage_cm"] = confusion_matrix(y_l_stage, y_l_stage_pred).tolist()
        lstm_metrics["stage_labels"] = [int(c) for c in sorted(set(stages))]
        lstm_metrics["Stage Accuracy"] = float(accuracy_score(y_l_stage, y_l_stage_pred))
        lstm_metrics["Stage F1 (macro)"] = float(f1_score(y_l_stage, y_l_stage_pred, average='macro', zero_division=0))
        print(f"LSTM         -> {lstm_metrics}")
    else:
        print(f"Skipping LSTM baseline (no {lstm_path} found).")

    # 5. Comparison report
    print("\n" + "=" * 60)
    print(" BENCHMARK REPORT ")
    print("=" * 60)
    print(f"{'Model':<20} | {'F1':<6} | {'Prec':<6} | {'Rec':<6} | {'FPR':<6} | {'AUC':<6}")
    print("-" * 60)

    def print_row(name, m):
        if m is None:
            return
        print(f"{name:<20} | {m.get('F1', 0):.4f} | {m.get('Precision', 0):.4f} | "
              f"{m.get('Recall', 0):.4f} | {m.get('FPR', 0):.4f} | {m.get('AUC-ROC', 0):.4f}")

    print_row("Logistic Baseline", log_inf)
    if lstm_metrics:
        print_row("LSTM Baseline", lstm_metrics)
    print_row("WorldModel (Ours)", wm_inf)
    print("=" * 60)

    results = {
        "WorldModel": {"name": "World Model (Transformer)", "metrics": wm_inf},
        "LogisticBaseline": {"name": "Logistic Regression", "metrics": log_inf},
        "LSTMBaseline": {"name": "LSTM Baseline", "metrics": lstm_metrics} if lstm_metrics else None,
    }

    res_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {res_path}")


if __name__ == '__main__':
    main()