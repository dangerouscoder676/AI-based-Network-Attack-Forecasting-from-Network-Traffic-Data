import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.csv_loader import CICIDSLoader
from src.data.feature_extractor import FeatureExtractor
from src.data.normalizer import FeatureNormalizer
from src.data.state_builder import StateBuilder, compress_states
from src.models.world_model import WorldModel
from src.utils.config import load_config, get_device, set_seed

class TrafficDataset(Dataset):
    """Dataset for the World Model.

    Each sample is a context window of length `context_length` paired with the
    next state it should predict, plus infiltration + stage labels for that
    next state.
    """
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
            torch.FloatTensor(np.asarray([self.infiltration_labels[idx]], dtype=np.float32)).view(1),
            torch.LongTensor(np.asarray([self.stage_labels[idx]], dtype=np.int64)).view(1)
        )


def load_synthetic(data_dir: str, context_length: int):
    """Load the synthetic npz produced by generate_synthetic.py."""
    data_path = os.path.join(data_dir, "synthetic_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Synthetic data not found at {data_path}. Run generate_synthetic.py first.")
    loaded = np.load(data_path)

    if 'sequences' in loaded:
        sequences = loaded['sequences']
        next_states = loaded['next_states'] if 'next_states' in loaded else sequences[:, -1, :]
        stages = loaded['stages']
    elif 'data' in loaded:
        sequences = loaded['data'][:, :-1, :]
        next_states = loaded['data'][:, -1, :]
        stages = loaded['stages']
    else:
        raise ValueError(f"Unrecognized npz schema in {data_path}")

    return sequences, next_states, stages


def load_csv(data_dir: str, config) -> tuple:
    """Load real CIC-IDS-2018 CSV files through the full flow pipeline."""
    loader = CICIDSLoader()
    raw_data = loader.load(data_dir)
    if raw_data.empty:
        raise ValueError(f"No usable CSV data found in {data_dir}")

    extractor = FeatureExtractor(config['features']['flow_features'])
    features = extractor.extract(raw_data)

    normalizer = FeatureNormalizer()
    features = normalizer.fit_transform(features)
    normalizer.save(os.path.join(config['data']['processed_dir'], 'normalizer.pkl'))

    builder = StateBuilder()
    states, stage_labels = builder.build_states(features, window_size=config['data']['window_size_seconds'])

    target_dim = int(config['features']['num_state_features'])
    states, _ = compress_states(states, target_dim)

    sequences, next_states, seq_stages = builder.build_forecast_sequences(
        states, stage_labels, context_length=int(config['data']['context_length'])
    )
    return sequences, next_states, seq_stages


def main():
    parser = argparse.ArgumentParser(description="Train the AI Network Attack Forecasting model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data_source', type=str, choices=['synthetic', 'csv'], default='synthetic', help='Data source')
    parser.add_argument('--data_dir', type=str, default='data/synthetic', help='Directory containing data')
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--model_type', type=str, choices=['transformer', 'lstm'], help='Override model type')
    parser.add_argument('--output_dir', type=str, default='models/', help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # 1. Load config
    config_dict = load_config(args.config).as_dict()

    # Overrides
    if args.epochs:
        config_dict['training']['epochs'] = args.epochs
    if args.batch_size:
        config_dict['training']['batch_size'] = args.batch_size
    if args.model_type:
        config_dict['model']['type'] = args.model_type
    if args.data_source == 'csv':
        os.makedirs(config_dict['data']['processed_dir'], exist_ok=True)

    # 2. Set random seeds
    set_seed(config_dict['data']['random_seed'])
    device = get_device(config_dict['training']['device'])
    print(f"Using device: {device}")

    # 3. Load data
    print(f"Loading {args.data_source} data from {args.data_dir}...")
    if args.data_source == 'csv':
        sequences, next_states, stages = load_csv(args.data_dir, config_dict)
    else:
        sequences, next_states, stages = load_synthetic(args.data_dir, context_length=config_dict['data']['context_length'])

    infiltration_labels = (np.asarray(stages) > 0).astype(np.float32)

    # 4. Create train/val/test splits (chronological-safe random split)
    num_samples = len(sequences)
    if num_samples == 0:
        raise ValueError("No training sequences available — check the data source.")
    indices = np.random.permutation(num_samples)
    train_end = int(num_samples * config_dict['data']['train_split'])
    val_end = train_end + int(num_samples * config_dict['data']['val_split'])

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]

    # 5. Create DataLoaders
    train_dataset = TrafficDataset(sequences[train_idx], next_states[train_idx],
                                   infiltration_labels[train_idx], stages[train_idx])
    val_dataset = TrafficDataset(sequences[val_idx], next_states[val_idx],
                                 infiltration_labels[val_idx], stages[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=config_dict['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config_dict['training']['batch_size'], shuffle=False)

    # 6. Build WorldModel
    print(f"Building WorldModel (type: {config_dict['model']['type']})...")
    model = WorldModel.from_config(config_dict).to(device)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # 7. Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config_dict['training']['learning_rate'],
        weight_decay=config_dict['training']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config_dict['training']['epochs']
    )

    # 8. Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    print("Starting training...")
    for epoch in range(config_dict['training']['epochs']):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config_dict['training']['epochs']} [Train]")
        for seq, next_state, inf_label, stage_label in pbar:
            seq = seq.to(device)
            next_state = next_state.to(device)
            inf_label = inf_label.to(device)
            stage_label = stage_label.squeeze(1).to(device) if stage_label.ndim > 1 else stage_label.to(device)

            optimizer.zero_grad()

            outputs = model(seq)
            targets = {
                'next_state': next_state,
                'infiltration': inf_label,
                'stage': stage_label
            }

            losses = model.compute_loss(outputs, targets)
            total_loss = losses['total_loss']

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config_dict['training']['gradient_clip_norm'])
            optimizer.step()

            train_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})

        train_loss /= max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, next_state, inf_label, stage_label in val_loader:
                seq = seq.to(device)
                next_state = next_state.to(device)
                inf_label = inf_label.to(device)
                stage_label = stage_label.squeeze(1).to(device) if stage_label.ndim > 1 else stage_label.to(device)

                outputs = model(seq)
                targets = {
                    'next_state': next_state,
                    'infiltration': inf_label,
                    'stage': stage_label
                }
                losses = model.compute_loss(outputs, targets)
                val_loss += losses['total_loss'].item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print("  Saved best model")

    # 9. Save final model and history
    model_suffix = config_dict['model']['type']
    torch.save(model.state_dict(), os.path.join(args.output_dir, f'{model_suffix}_model.pth'))
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    with open(os.path.join(args.output_dir, f'history_{model_suffix}.json'), 'w') as f:
        json.dump(history, f)

    print("\nTraining Summary:")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("Training complete.")


if __name__ == '__main__':
    main()