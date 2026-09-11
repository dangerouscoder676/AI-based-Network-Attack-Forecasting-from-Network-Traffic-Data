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
from src.data.state_builder import StateBuilder
from src.models.world_model import WorldModel
from src.utils.config import load_config, get_device, set_seed

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
            torch.FloatTensor(np.array(self.sequences[idx], dtype=np.float32)),
            torch.FloatTensor(np.array(self.next_states[idx], dtype=np.float32)),
            torch.FloatTensor(np.array([self.infiltration_labels[idx]], dtype=np.float32)).view(1),
            torch.LongTensor(np.array([self.stage_labels[idx]], dtype=np.int64)).view(1)
        )

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
    config = load_config(args.config)
    
    # Overrides
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.model_type:
        config.model.type = args.model_type

    # 2. Set random seeds
    set_seed(config.data.random_seed)
    device = get_device(config.training.device)
    print(f"Using device: {device}")

    # 3. Load data
    print(f"Loading {args.data_source} data from {args.data_dir}...")
    if args.data_source == 'csv':
        loader = CICIDSLoader(args.data_dir)
        raw_data = loader.load()
        extractor = FeatureExtractor()
        features = extractor.extract(raw_data)
        normalizer = FeatureNormalizer()
        features = normalizer.fit_transform(features)
        builder = StateBuilder(context_length=config.data.context_length)
        sequences, next_states, labels, stages = builder.build(features)
    else:
        # Load synthetic
        data_path = os.path.join(args.data_dir, "synthetic_data.npz")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Synthetic data not found at {data_path}. Run generate_synthetic.py first.")
        loaded = np.load(data_path)
        
        if 'sequences' in loaded:
            sequences = loaded['sequences']
            labels = loaded['labels']
            stages = loaded['stages']
            next_states = sequences[:, -1, :]
        else:
            data = loaded['data']
            sequences = data[:, :-1, :]
            next_states = data[:, -1, :]
            labels = loaded['labels']
            stages = loaded['stages']

    # 4. Create train/val/test splits
    num_samples = len(sequences)
    indices = np.random.permutation(num_samples)
    train_end = int(num_samples * config.data.train_split)
    val_end = train_end + int(num_samples * config.data.val_split)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # 5. Create DataLoaders
    train_dataset = TrafficDataset(sequences[train_idx], next_states[train_idx], labels[train_idx], stages[train_idx])
    val_dataset = TrafficDataset(sequences[val_idx], next_states[val_idx], labels[val_idx], stages[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    # 6. Build WorldModel
    print(f"Building WorldModel (type: {config.model.type})...")
    config_dict = config._dict if hasattr(config, '_dict') else config
    model = WorldModel.from_config(config_dict).to(device)
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # 7. Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.training.epochs
    )

    # Losses
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()

    # 8. Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    print("Starting training...")
    for epoch in range(config.training.epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.epochs} [Train]")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            optimizer.step()
            
            train_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})
            
        train_loss /= len(train_loader)
        
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
                
        val_loss /= len(val_loader)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print("  Saved best model")

    # 9. Save final model and history
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
        
    print("\nTraining Summary:")
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print("Training complete.")

if __name__ == '__main__':
    main()
