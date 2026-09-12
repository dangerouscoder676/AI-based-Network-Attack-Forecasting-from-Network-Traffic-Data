import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic_generator import SyntheticGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic network traffic data")
    parser.add_argument('--num_windows', type=int, default=1000, help="Number of windows to generate")
    parser.add_argument('--context_length', type=int, default=16, help="Context length")
    parser.add_argument('--output_dir', type=str, default='data/synthetic', help="Output directory")
    parser.add_argument('--num_features', type=int, default=76, help="Number of features per state")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_windows} windows of synthetic data...")
    generator = SyntheticGenerator()
    
    states, stages, feature_names = generator.generate(num_windows=args.num_windows, num_features=args.num_features)
    
    # Binary infiltration label (0 for benign stage 0, 1 for any attack stage > 0)
    labels = (stages > 0).astype(np.int64)
    
    # Construct sequences with explicit next-state targets
    num_seq = len(states) - args.context_length
    sequences = np.zeros((num_seq, args.context_length, args.num_features), dtype=np.float32)
    next_states = np.zeros((num_seq, args.num_features), dtype=np.float32)
    seq_labels = np.zeros((num_seq, 1), dtype=np.float32)
    seq_stages = np.zeros((num_seq,), dtype=np.int64)
    
    for i in range(num_seq):
        sequences[i] = states[i:i+args.context_length]
        next_states[i] = states[i+args.context_length]
        seq_labels[i] = labels[i+args.context_length]
        seq_stages[i] = stages[i+args.context_length]
        
    npz_path = os.path.join(args.output_dir, "synthetic_data.npz")
    np.savez_compressed(
        npz_path,
        sequences=sequences,
        next_states=next_states,
        labels=seq_labels,
        stages=seq_stages,
        states=states,
        state_stages=stages,
        feature_names=feature_names
    )
    print(f"Saved synthetic data to {npz_path}")
    
    # Save a small CSV sample for dashboard demo
    csv_path = os.path.join(args.output_dir, "dashboard_demo.csv")
    demo_df = pd.DataFrame(states[:100], columns=feature_names)
    demo_df['label'] = labels[:100]
    demo_df['stage'] = stages[:100]
    demo_df.to_csv(csv_path, index=False)
    print(f"Saved dashboard demo sample to {csv_path}")

    # Summary statistics
    print("\nSummary Statistics:")
    print(f"Total sequences: {len(sequences)}")
    print(f"Infiltration rate: {np.mean(seq_labels):.2%}")
    print(f"Stage distribution: {np.bincount(seq_stages)}")

if __name__ == '__main__':
    main()
