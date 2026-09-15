"""Train a World Model on live-style flow-statistic states.

Generates flow-level traffic windows (benign + kill-chain phases),
maps them through LiveStateMapper to 76-dim states, normalises, and
trains a WorldModel using the same hyper-parameters as the main
``train.py`` pipeline.  The resulting checkpoint is saved under
``models/`` as ``live_transformer_model.pth`` (and best-of-run as
``best_live_model.pth``).

Usage::

    python scripts/train_live.py                  # defaults
    python scripts/train_live.py --epochs 80 --windows 2000
"""

import os
import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scapy.all import IP, TCP, UDP  # noqa: F401 – UDP available if needed

from src.data.pcap_parser import FlowAggregator
from src.data.live_capture import LiveStateMapper, STATE_DIM
from src.models.world_model import WorldModel
from src.utils.config import load_config, get_device, set_seed

# ---------------------------------------------------------------------------
# Synthetic flow-level generator
# ---------------------------------------------------------------------------

_RNG = None


def _rng(seed):
    global _RNG
    _RNG = np.random.RandomState(seed)
    return _RNG


def _rand(n):
    return _RNG.randint(0, n)


def _uniform(lo, hi):
    return _RNG.uniform(lo, hi)


def _rand_sample(arr):
    return arr[_rand(len(arr))]


INTERNAL_IPS = [f"10.0.0.{i}" for i in range(1, 201)]
EXTERNAL_IPS = [
    "93.184.216.34", "151.101.1.69", "185.199.108.153",
    "140.82.121.4", "172.217.14.110", "20.190.159.0",
]
ATTACKER_IPS = ["198.51.100.7", "203.0.113.42", "192.0.2.99"]


def _generate_benign(agg, base, rng):
    """Mix of web browsing, DNS queries, and light bulk transfer."""
    n_flows = int(rng.randint(12, 35))
    for i in range(n_flows):
        src = _rand_sample(INTERNAL_IPS)
        r = rng.rand()
        if r < 0.30:
            dport = 53
            proto = 17  # UDP
            payload = bytes(rng.randint(0, 256, size=int(rng.randint(40, 120))))
            pkt = IP(src=src, dst=_rand_sample(EXTERNAL_IPS), proto=proto) / \
                  UDP(sport=int(rng.randint(1024, 60000)), dport=53) / payload
        elif r < 0.85:
            dport = _rand_sample([443, 80, 8080, 8443])
            sport = int(rng.randint(49152, 65535))
            payload = bytes(rng.randint(0, 256, size=int(rng.randint(64, 512))))
            pkt = IP(src=src, dst=_rand_sample(EXTERNAL_IPS), proto=6) / \
                  TCP(sport=sport, dport=dport, flags="AP") / payload
        else:
            dport = 443
            sport = int(rng.randint(49152, 65535))
            n_pkts = int(rng.randint(4, 12))
            for j in range(n_pkts):
                p = IP(src=src, dst=_rand_sample(EXTERNAL_IPS), proto=6) / \
                    TCP(sport=sport, dport=dport, flags="AP") / \
                    bytes(rng.randint(0, 256, size=int(rng.randint(128, 1460))))
                agg.update(p, timestamp=base + (i * 0.01) + (j * 0.001))
            continue
        agg.update(pkt, timestamp=base + i * _uniform(0.001, 0.05))


def _generate_recon(agg, base, rng):
    """SYN scan across port range with occasional RST-ACK responses."""
    n_ports = int(rng.randint(60, 160))
    attacker = _rand_sample(ATTACKER_IPS)
    target = _rand_sample(INTERNAL_IPS)
    for port in range(1, n_ports + 1):
        sport = int(rng.randint(1024, 65535))
        agg.update(
            IP(src=attacker, dst=target, proto=6) / TCP(sport=sport, dport=port, flags="S"),
            timestamp=base + port * 0.003,
        )
        if rng.rand() < 0.55:
            agg.update(
                IP(src=target, dst=attacker, proto=6) / TCP(sport=port, dport=sport, flags="RA"),
                timestamp=base + port * 0.003 + 0.001,
            )


def _generate_bruteforce(agg, base, rng):
    """Repeated SSH/FTP login attempts from many source ports."""
    n_tries = int(rng.randint(60, 150))
    attacker = _rand_sample(ATTACKER_IPS)
    target = _rand_sample(INTERNAL_IPS)
    target_port = _rand_sample([22, 21, 3389])
    for i in range(n_tries):
        sport = int(rng.randint(1024, 65535))
        agg.update(
            IP(src=attacker, dst=target, proto=6) / TCP(sport=sport, dport=target_port, flags="S"),
            timestamp=base + i * 0.025,
        )
        if rng.rand() < 0.40:
            agg.update(
                IP(src=target, dst=attacker, proto=6) / TCP(sport=target_port, dport=sport, flags="SA"),
                timestamp=base + i * 0.025 + 0.005,
            )


def _generate_dos(agg, base, rng):
    """High-volume burst to a single web server port."""
    n_pkts = int(rng.randint(300, 700))
    target = _rand_sample(INTERNAL_IPS[:50])
    dport = _rand_sample([80, 443, 8080])
    for i in range(n_pkts):
        src = f"10.0.0.{_rand(200) + 1}"
        sport = int(rng.randint(1024, 65535))
        payload = bytes(rng.randint(0, 256, size=_rand_sample([64, 128, 256])))
        agg.update(
            IP(src=src, dst=target, proto=6) / TCP(sport=sport, dport=dport, flags="AP") / payload,
            timestamp=base + i * 0.0006,
        )


_GENERATORS = {
    0: _generate_benign,
    1: _generate_recon,
    2: _generate_bruteforce,
    3: _generate_dos,
}

# Phase distribution: ~70% benign, 10% recon, 10% bruteforce, 10% DOS
_CDF = [0.70, 0.80, 0.90, 1.00]


def _choose_phase(rng):
    r = rng.rand()
    for i, threshold in enumerate(_CDF):
        if r < threshold:
            return i
    return 0


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _run_sequence(n_windows, rng):
    """Schedule phase runs. Returns a list of per-window phases.

    Benign traffic appears in long quiet runs; attacks appear as contiguous
    episodes of 3-8 windows (matching real kill-chain persistence) so the World
    Model has temporal context to learn from.
    """
    phases = []
    i = 0
    while i < n_windows:
        r = rng.rand()
        if r < 0.72:                                # benign run (5-20 windows)
            length = int(rng.randint(5, 20))
            phases.extend([0] * min(length, n_windows - i))
        else:                                       # attack episode (3-8 windows)
            length = int(rng.randint(3, 8))
            attack_phase = int(rng.randint(1, 4))
            phases.extend([attack_phase] * min(length, n_windows - i))
        i += length
    return phases[:n_windows]


def generate_windows(n_windows: int, window_size_s: float, seed: int):
    """Generate n_windows of flow-level traffic, return (raw_states, phase_labels)."""
    rng = _rng(seed)
    mapper = LiveStateMapper()
    t0 = 1_700_000_000.0
    raws = []
    labels = []
    for w, phase in enumerate(_run_sequence(n_windows, rng)):
        base = t0 + w * window_size_s
        agg = FlowAggregator()
        _GENERATORS[phase](agg, base, rng)
        f = agg.snapshot()
        if f.empty:
            continue
        raws.append(mapper.build_raw_state(f, window_size_s=window_size_s))
        labels.append(phase)
    return np.stack(raws), np.asarray(labels, dtype=np.int64)


def prepare_dataset(raws, labels, context_length):
    """Z-score normalise, build context sequences, return dict of arrays."""
    mu = raws.mean(axis=0)
    sd = raws.std(axis=0) + 1e-6
    states = np.clip((raws - mu) / sd, -6.0, 6.0).astype(np.float32)
    stats = {"mean": mu, "std": sd}

    n = len(states)
    seqs = np.zeros((n - context_length, context_length, STATE_DIM), dtype=np.float32)
    nexts = np.zeros((n - context_length, STATE_DIM), dtype=np.float32)
    for i in range(n - context_length):
        seqs[i] = states[i : i + context_length]
        nexts[i] = states[i + context_length]

    # Label the *last window in the context*: the model reads the present,
    # like the live monitor's rolling context window.
    inf_labels = (labels[context_length - 1 : -1] > 0).astype(np.float32)
    stage_labels = labels[context_length - 1 : -1].astype(np.int64)

    return {
        "sequences": seqs,
        "next_states": nexts,
        "infiltration_labels": inf_labels,
        "stage_labels": stage_labels,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Training Dataset
# ---------------------------------------------------------------------------

class LiveDataset(Dataset):
    def __init__(self, sequences, next_states, infiltration_labels, stage_labels):
        self.sequences = sequences
        self.next_states = next_states
        self.infiltration_labels = infiltration_labels
        self.stage_labels = stage_labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.next_states[idx]),
            torch.FloatTensor(np.asarray([self.infiltration_labels[idx]], dtype=np.float32)).view(1),
            torch.LongTensor(np.asarray([self.stage_labels[idx]], dtype=np.int64)).view(1),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train World Model on live-style flow states")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--windows", type=int, default=1400, help="Total traffic windows to generate")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--w_transition", type=float, default=None, help="Override transition loss weight")
    parser.add_argument("--w_infiltration", type=float, default=None, help="Override infiltration loss weight")
    parser.add_argument("--w_stage", type=float, default=None, help="Override stage loss weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data generation")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--save_dataset", type=str, default=None, help="Path to save generated .npz (for reuse)")
    parser.add_argument("--load_dataset", type=str, default=None, help="Path to load previously saved .npz")
    args = parser.parse_args()

    cfg = load_config(args.config).as_dict()
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.w_transition or args.w_infiltration or args.w_stage:
        cfg["training"]["loss_weights"] = {
            "transition": args.w_transition or cfg["training"]["loss_weights"]["transition"],
            "infiltration": args.w_infiltration or cfg["training"]["loss_weights"]["infiltration"],
            "stage": args.w_stage or cfg["training"]["loss_weights"]["stage"],
        }

    set_seed(cfg["data"]["random_seed"])
    device = get_device(cfg["training"]["device"])
    print(f"Device: {device}")

    # 1. Generate or load data
    if args.load_dataset and os.path.exists(args.load_dataset):
        print(f"Loading dataset from {args.load_dataset}")
        loaded = np.load(args.load_dataset, allow_pickle=True)
        raws, labels = loaded["raws"], loaded["labels"]
    else:
        print(f"Generating {args.windows} flow-level windows (seed={args.seed})...")
        t0 = time.time()
        raws, labels = generate_windows(args.windows, cfg["data"]["window_size_seconds"], args.seed)
        print(f"  Generated {len(labels)} windows in {time.time()-t0:.1f}s")
        print(f"  Distribution: benign={int((labels==0).sum())}, recon={int((labels==1).sum())}, "
              f"bruteforce={int((labels==2).sum())}, dos={int((labels==3).sum())}")

    if args.save_dataset:
        os.makedirs(os.path.dirname(args.save_dataset) or ".", exist_ok=True)
        np.savez_compressed(args.save_dataset, raws=raws, labels=labels)
        print(f"  Saved dataset to {args.save_dataset}")

    # 2. Prepare sequences
    ctx_len = int(cfg["data"]["context_length"])
    data = prepare_dataset(raws, labels, ctx_len)
    n_seqs = len(data["sequences"])
    print(f"  Sequences: {n_seqs} (context_length={ctx_len})")

    # 3. Train/val split
    indices = np.random.permutation(n_seqs)
    n_train = int(n_seqs * cfg["data"]["train_split"])
    n_val = int(n_seqs * cfg["data"]["val_split"])
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]

    train_ds = LiveDataset(
        data["sequences"][train_idx],
        data["next_states"][train_idx],
        data["infiltration_labels"][train_idx],
        data["stage_labels"][train_idx],
    )
    val_ds = LiveDataset(
        data["sequences"][val_idx],
        data["next_states"][val_idx],
        data["infiltration_labels"][val_idx],
        data["stage_labels"][val_idx],
    )
    train_dl = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # 4. Build model
    print(f"Building WorldModel (type: {cfg['model']['type']})...")
    model = WorldModel.from_config(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )

    # 5. Train
    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")
    history = {"train_loss": [], "val_loss": []}
    patience = cfg["training"].get("early_stopping_patience", 10)
    patience_ctr = 0

    print("Training...")
    t0 = time.time()
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        train_loss = 0
        for seq, nxt, inf, stage in train_dl:
            seq, nxt = seq.to(device), nxt.to(device)
            inf, stage = inf.to(device), stage.to(device)
            stage = stage.squeeze(1) if stage.ndim > 1 else stage

            optimizer.zero_grad()
            out = model(seq)
            losses = model.compute_loss(out, {"next_state": nxt, "infiltration": inf, "stage": stage})
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_norm"])
            optimizer.step()
            train_loss += losses["total_loss"].item()
        train_loss /= max(len(train_dl), 1)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, nxt, inf, stage in val_dl:
                seq, nxt = seq.to(device), nxt.to(device)
                inf, stage = inf.to(device), stage.to(device)
                stage = stage.squeeze(1) if stage.ndim > 1 else stage
                out = model(seq)
                losses = model.compute_loss(out, {"next_state": nxt, "infiltration": inf, "stage": stage})
                val_loss += losses["total_loss"].item()
        val_loss /= max(len(val_dl), 1)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_live_model.pth"))
        else:
            patience_ctr += 1

        marker = "  [best]" if improved else ""
        print(f"  Epoch {epoch+1:3d}  train={train_loss:.4f}  val={val_loss:.4f}{marker}")

        if patience_ctr >= patience:
            print(f"  Early stopping after {patience} epochs without improvement")
            break

    elapsed = time.time() - t0
    # Save final
    torch.save(model.state_dict(), os.path.join(args.output_dir, "live_transformer_model.pth"))
    with open(os.path.join(args.output_dir, "history_live.json"), "w") as f:
        json.dump(history, f)

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Best val loss: {best_val:.4f}")
    print(f"  Saved: {args.output_dir}/live_transformer_model.pth")
    print(f"  Saved: {args.output_dir}/best_live_model.pth")

    # 6. Evaluate
    print("\nEvaluation on validation set:")
    best_model = WorldModel.from_config(cfg).to(device)
    best_model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_live_model.pth"), map_location=device))
    best_model.eval()

    correct = 0
    total = 0
    correct_stage = 0
    inf_probs_by_phase = {0: [], 1: [], 2: [], 3: []}
    val_labels = data["stage_labels"][val_idx]

    with torch.no_grad():
        for seq, nxt, inf, stage in val_dl:
            seq = seq.to(device)
            out = best_model(seq)
            probs = out["infiltration_prob"].squeeze(-1).cpu().numpy()
            pred_stages = out["stage_logits"].argmax(dim=-1).cpu().numpy()
            total += len(probs)
            correct += int(((probs > 0.5).astype(float) == inf.squeeze(-1).numpy()).sum())
            correct_stage += int((pred_stages == stage.squeeze(1).cpu().numpy()).sum())
            for i in range(len(probs)):
                inf_probs_by_phase[int(val_labels[total - len(probs) + i])].append(float(probs[i]))

    print(f"  Infiltration accuracy: {correct/total:.3f}")
    print(f"  Stage accuracy: {correct_stage/total:.3f}")
    names = {0: "benign", 1: "recon", 2: "bruteforce", 3: "dos"}
    for phase in sorted(inf_probs_by_phase):
        if inf_probs_by_phase[phase]:
            probs = inf_probs_by_phase[phase]
            print(f"  {names[phase]:10s}  n={len(probs):4d}  "
                  f"mean_prob={sum(probs)/len(probs):.3f}  "
                  f"max={max(probs):.3f}")


if __name__ == "__main__":
    main()
