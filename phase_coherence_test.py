import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import (
    PhaseStructuredDataset,
    MedleyDBSamplePairs,
)


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    dataset: str = "synthetic"  # synthetic | medleydb_sample
    medleydb_root: str = "data/MedleyDB_sample"

    seq_len: int = 256
    num_classes: int = 4

    samples_per_class_train: int = 1500
    samples_per_class_val: int = 400

    # For MedleyDB Sample we use repeated random segment sampling
    train_samples_per_epoch: int = 1024
    val_samples_per_epoch: int = 256

    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3

    hidden_dim: int = 128
    embed_dim: int = 64

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_runs: int = 3
    num_workers: int = 0

    # Synthetic data difficulty
    base_f1: float = 6.0
    base_f2: float = 12.0
    freq_jitter: float = 0.5
    amp_low: float = 0.7
    amp_high: float = 1.3
    phase_jitter: float = 0.35
    noise_std: float = 0.2

    # MedleyDB
    sample_rate: int = 44100
    segment_seconds: float = 2.0
    medley_max_tracks: int = 0  # 0 = all

    # Alignment losses
    lambda_amp: float = 0.10
    lambda_phase: float = 0.10

    # Coherence gating
    gate_beta: float = 2.0
    gate_gamma: float = 0.20
    gate_warmup_epochs: int = 5

    # Logging
    log_every_n_batches: int = 10

    # Saving
    checkpoint_dir: str = "checkpoints"
    save_checkpoints: bool = True


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Logging helpers
# ============================================================

def log_batch_progress(
    stage: str,
    epoch: int,
    batch_idx: int,
    total_batches: int,
    loss: float,
    start_time: float,
    extra: str = "",
) -> None:
    elapsed = time.time() - start_time
    it_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0.0

    msg = (
        f"[{stage}] "
        f"epoch {epoch + 1:02d} "
        f"batch {batch_idx + 1}/{total_batches} "
        f"loss {loss:.4f} "
        f"{it_per_sec:.2f} it/s"
    )

    if extra:
        msg += f" | {extra}"

    print(msg, flush=True)


# ============================================================
# Checkpoint helpers
# ============================================================

def ensure_checkpoint_dir(cfg: Config) -> Path:
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def save_run_config(cfg: Config) -> None:
    checkpoint_dir = ensure_checkpoint_dir(cfg)
    config_path = checkpoint_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def maybe_save_best(
    best_acc: float,
    current_acc: float,
    model: nn.Module,
    model_name: str,
    cfg: Config,
) -> Tuple[float, Optional[Dict[str, torch.Tensor]]]:
    if current_acc > best_acc:
        state = copy.deepcopy(model.state_dict())

        if cfg.save_checkpoints:
            checkpoint_dir = ensure_checkpoint_dir(cfg)
            ckpt_path = checkpoint_dir / f"{model_name}_best.pt"
            torch.save(state, ckpt_path)
            print(f"[checkpoint] saved best {model_name} -> {ckpt_path}", flush=True)

        return current_acc, state

    return best_acc, None


def load_best_state(
    model: nn.Module,
    best_state: Optional[Dict[str, torch.Tensor]],
) -> nn.Module:
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================
# Models
# ============================================================

class ConvEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return self.pool(h).squeeze(-1)


class BaselineModel(nn.Module):
    def __init__(self, hidden_dim: int, embed_dim: int, num_classes: int):
        super().__init__()

        self.encoder = ConvEncoder(hidden_dim)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = F.normalize(self.proj(h), dim=-1)
        logits = self.cls(z)
        return z, logits


class PhaseModel(nn.Module):
    def __init__(self, hidden_dim: int, embed_dim: int, num_classes: int):
        super().__init__()

        self.encoder = ConvEncoder(hidden_dim)
        self.real = nn.Linear(hidden_dim, embed_dim)
        self.imag = nn.Linear(hidden_dim, embed_dim)
        self.cls = nn.Linear(embed_dim * 2, num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)

        r = self.real(h)
        i = self.imag(h)

        amp = torch.sqrt(r.pow(2) + i.pow(2) + 1e-8)
        phase = torch.atan2(i, r)

        z = torch.cat([r, i], dim=-1)
        z = F.normalize(z, dim=-1)

        logits = self.cls(z)

        return r, i, amp, phase, z, logits


# ============================================================
# Coherence / losses
# ============================================================

def amplitude_alignment_loss(amp_a: torch.Tensor, amp_b: torch.Tensor) -> torch.Tensor:
    return ((amp_a - amp_b) ** 2).mean(dim=-1)


def phase_alignment_loss(phase_a: torch.Tensor, phase_b: torch.Tensor) -> torch.Tensor:
    return (1.0 - torch.cos(phase_a - phase_b)).mean(dim=-1)


def normalized_phase_coherence(
    amp_a: torch.Tensor,
    phase_a: torch.Tensor,
    amp_b: torch.Tensor,
    phase_b: torch.Tensor,
) -> torch.Tensor:
    amp_a_n = amp_a / (amp_a.mean(dim=-1, keepdim=True) + 1e-6)
    amp_b_n = amp_b / (amp_b.mean(dim=-1, keepdim=True) + 1e-6)
    return (amp_a_n * amp_b_n * torch.cos(phase_a - phase_b)).mean(dim=-1)


def coherence_gate(
    coh: torch.Tensor,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    coh_centered = (coh - coh.mean().detach()) / (coh.std().detach() + 1e-6)
    alpha = 1.0 + gamma * torch.tanh(beta * coh_centered)
    return alpha


# ============================================================
# Dataset helpers
# ============================================================

def build_datasets(cfg: Config):
    if cfg.dataset == "synthetic":
        train_ds = PhaseStructuredDataset(train=True, cfg=cfg)
        val_ds = PhaseStructuredDataset(train=False, cfg=cfg)
        num_classes = cfg.num_classes
        is_paired = False
        return train_ds, val_ds, num_classes, is_paired

    if cfg.dataset == "medleydb_sample":
        train_ds = MedleyDBSamplePairs(
            root=cfg.medleydb_root,
            sample_rate=cfg.sample_rate,
            segment_seconds=cfg.segment_seconds,
            max_tracks=cfg.medley_max_tracks,
            samples_per_epoch=cfg.train_samples_per_epoch,
            seed=cfg.seed,
        )
        val_ds = MedleyDBSamplePairs(
            root=cfg.medleydb_root,
            sample_rate=cfg.sample_rate,
            segment_seconds=cfg.segment_seconds,
            max_tracks=cfg.medley_max_tracks,
            samples_per_epoch=cfg.val_samples_per_epoch,
            seed=cfg.seed + 10_000,
            track_names=train_ds.track_names,
        )
        num_classes = train_ds.num_classes
        is_paired = True
        return train_ds, val_ds, num_classes, is_paired

    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def make_loaders(cfg: Config):
    train_ds, val_ds, num_classes, is_paired = build_datasets(cfg)

    print(
        f"Built datasets | train_samples={len(train_ds)} | "
        f"val_samples={len(val_ds)} | num_classes={num_classes} | paired={is_paired}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    print(
        f"Built loaders | train_batches={len(train_loader)} | "
        f"val_batches={len(val_loader)} | batch_size={cfg.batch_size}",
        flush=True,
    )

    return train_loader, val_loader, num_classes, is_paired


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_baseline(
    model: BaselineModel,
    loader: DataLoader,
    cfg: Config,
    is_paired: bool,
) -> float:
    model.eval()

    total = 0
    correct = 0

    for batch in loader:
        if is_paired:
            x, _, y = batch
        else:
            x, y = batch

        x = x.to(cfg.device)
        y = y.to(cfg.device)

        _, logits = model(x)
        preds = logits.argmax(dim=-1)

        total += y.size(0)
        correct += (preds == y).sum().item()

    return correct / total


@torch.no_grad()
def evaluate_phase(
    model: PhaseModel,
    loader: DataLoader,
    cfg: Config,
    is_paired: bool,
) -> Tuple[float, float]:
    model.eval()

    total = 0
    correct = 0
    coh_vals: List[float] = []

    for batch in loader:
        if is_paired:
            x, x_ref, y = batch
        else:
            x, y = batch
            x_ref = x

        x = x.to(cfg.device)
        x_ref = x_ref.to(cfg.device)
        y = y.to(cfg.device)

        _, _, amp, phase, _, logits = model(x)
        _, _, amp_ref, phase_ref, _, _ = model(x_ref)

        preds = logits.argmax(dim=-1)

        total += y.size(0)
        correct += (preds == y).sum().item()

        coh = normalized_phase_coherence(amp, phase, amp_ref, phase_ref)
        coh_vals.append(coh.mean().item())

    return correct / total, sum(coh_vals) / len(coh_vals)


# ============================================================
# Training variants
# ============================================================

def train_baseline(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    is_paired: bool,
) -> BaselineModel:
    print("\n=== Baseline ===", flush=True)

    model = BaselineModel(cfg.hidden_dim, cfg.embed_dim, num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        loss_sum = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if is_paired:
                x, _, y = batch
            else:
                x, y = batch

            x = x.to(cfg.device)
            y = y.to(cfg.device)

            _, logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()

            if batch_idx % cfg.log_every_n_batches == 0:
                log_batch_progress(
                    "Baseline",
                    epoch,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    epoch_start,
                )

        val_acc = evaluate_baseline(model, val_loader, cfg, is_paired)
        best_acc, maybe_state = maybe_save_best(best_acc, val_acc, model, "baseline", cfg)
        if maybe_state is not None:
            best_state = maybe_state

        print(
            f"[Baseline] epoch {epoch + 1:02d} done | "
            f"loss {loss_sum / len(train_loader):.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"best {best_acc:.4f} | "
            f"epoch_time {time.time() - epoch_start:.1f}s",
            flush=True,
        )

    return load_best_state(model, best_state)


def train_complex_only(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    is_paired: bool,
) -> PhaseModel:
    print("\n=== Complex Latent Only ===", flush=True)

    model = PhaseModel(cfg.hidden_dim, cfg.embed_dim, num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        loss_sum = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if is_paired:
                x, _, y = batch
            else:
                x, y = batch

            x = x.to(cfg.device)
            y = y.to(cfg.device)

            _, _, _, _, _, logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()

            if batch_idx % cfg.log_every_n_batches == 0:
                log_batch_progress(
                    "Complex",
                    epoch,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    epoch_start,
                )

        val_acc, _ = evaluate_phase(model, val_loader, cfg, is_paired)
        best_acc, maybe_state = maybe_save_best(best_acc, val_acc, model, "complex", cfg)
        if maybe_state is not None:
            best_state = maybe_state

        print(
            f"[Complex] epoch {epoch + 1:02d} done | "
            f"loss {loss_sum / len(train_loader):.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"best {best_acc:.4f} | "
            f"epoch_time {time.time() - epoch_start:.1f}s",
            flush=True,
        )

    return load_best_state(model, best_state)


def train_alignment_only(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    is_paired: bool,
) -> PhaseModel:
    print("\n=== Alignment Loss Only ===", flush=True)

    model = PhaseModel(cfg.hidden_dim, cfg.embed_dim, num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        loss_sum = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if is_paired:
                x, x_ref, y = batch
            else:
                x, y = batch
                x_ref = x

            x = x.to(cfg.device)
            x_ref = x_ref.to(cfg.device)
            y = y.to(cfg.device)

            _, _, amp, phase, _, logits = model(x)
            _, _, amp_ref, phase_ref, _, _ = model(x_ref)

            ce = F.cross_entropy(logits, y)
            amp_l = amplitude_alignment_loss(amp, amp_ref.detach()).mean()
            phase_l = phase_alignment_loss(phase, phase_ref.detach()).mean()

            loss = ce + cfg.lambda_amp * amp_l + cfg.lambda_phase * phase_l

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()

            if batch_idx % cfg.log_every_n_batches == 0:
                log_batch_progress(
                    "Align",
                    epoch,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    epoch_start,
                    extra=f"amp={amp_l.item():.4f} phase={phase_l.item():.4f}",
                )

        val_acc, _ = evaluate_phase(model, val_loader, cfg, is_paired)
        best_acc, maybe_state = maybe_save_best(best_acc, val_acc, model, "align", cfg)
        if maybe_state is not None:
            best_state = maybe_state

        print(
            f"[Align] epoch {epoch + 1:02d} done | "
            f"loss {loss_sum / len(train_loader):.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"best {best_acc:.4f} | "
            f"epoch_time {time.time() - epoch_start:.1f}s",
            flush=True,
        )

    return load_best_state(model, best_state)


def train_full(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    is_paired: bool,
) -> PhaseModel:
    print("\n=== Full Method (Alignment + Gentle Coherence Gating) ===", flush=True)

    model = PhaseModel(cfg.hidden_dim, cfg.embed_dim, num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        loss_sum = 0.0
        gate_sum = 0.0
        coh_sum = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if is_paired:
                x, x_ref, y = batch
            else:
                x, y = batch
                x_ref = x

            x = x.to(cfg.device)
            x_ref = x_ref.to(cfg.device)
            y = y.to(cfg.device)

            _, _, amp, phase, _, logits = model(x)
            _, _, amp_ref, phase_ref, _, _ = model(x_ref)

            per_sample_ce = F.cross_entropy(logits, y, reduction="none")
            per_sample_amp = amplitude_alignment_loss(amp, amp_ref.detach())
            per_sample_phase = phase_alignment_loss(phase, phase_ref.detach())

            coh = normalized_phase_coherence(
                amp, phase,
                amp_ref.detach(), phase_ref.detach()
            )

            if epoch < cfg.gate_warmup_epochs:
                alpha = torch.ones_like(coh)
            else:
                alpha = coherence_gate(coh, beta=cfg.gate_beta, gamma=cfg.gate_gamma)

            per_sample_loss = (
                alpha * per_sample_ce
                + cfg.lambda_amp * per_sample_amp
                + cfg.lambda_phase * per_sample_phase
            )

            loss = per_sample_loss.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()
            gate_sum += alpha.mean().item()
            coh_sum += coh.mean().item()

            if batch_idx % cfg.log_every_n_batches == 0:
                log_batch_progress(
                    "Full",
                    epoch,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    epoch_start,
                    extra=(
                        f"gate={alpha.mean().item():.4f} "
                        f"coh={coh.mean().item():.4f} "
                        f"amp={per_sample_amp.mean().item():.4f} "
                        f"phase={per_sample_phase.mean().item():.4f}"
                    ),
                )

        val_acc, val_coh = evaluate_phase(model, val_loader, cfg, is_paired)
        best_acc, maybe_state = maybe_save_best(best_acc, val_acc, model, "full", cfg)
        if maybe_state is not None:
            best_state = maybe_state

        print(
            f"[Full] epoch {epoch + 1:02d} done | "
            f"loss {loss_sum / len(train_loader):.4f} | "
            f"train_gate {gate_sum / len(train_loader):.4f} | "
            f"train_coh {coh_sum / len(train_loader):.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"val_coh {val_coh:.4f} | "
            f"best {best_acc:.4f} | "
            f"epoch_time {time.time() - epoch_start:.1f}s",
            flush=True,
        )

    return load_best_state(model, best_state)


# ============================================================
# Experiment runner
# ============================================================

def run_once(cfg: Config) -> Tuple[float, float, float, float]:
    set_seed(cfg.seed)

    train_loader, val_loader, num_classes, is_paired = make_loaders(cfg)

    baseline = train_baseline(cfg, train_loader, val_loader, num_classes, is_paired)
    complex_only = train_complex_only(cfg, train_loader, val_loader, num_classes, is_paired)
    alignment_only = train_alignment_only(cfg, train_loader, val_loader, num_classes, is_paired)
    full = train_full(cfg, train_loader, val_loader, num_classes, is_paired)

    baseline_acc = evaluate_baseline(baseline, val_loader, cfg, is_paired)
    complex_acc, _ = evaluate_phase(complex_only, val_loader, cfg, is_paired)
    align_acc, _ = evaluate_phase(alignment_only, val_loader, cfg, is_paired)
    full_acc, full_coh = evaluate_phase(full, val_loader, cfg, is_paired)

    print("\nRESULTS", flush=True)
    print(f"Baseline: {baseline_acc:.6f}", flush=True)
    print(f"Complex:  {complex_acc:.6f}", flush=True)
    print(f"Align:    {align_acc:.6f}", flush=True)
    print(f"Full:     {full_acc:.6f}", flush=True)
    print(f"Full coh: {full_coh:.6f}", flush=True)

    return baseline_acc, complex_acc, align_acc, full_acc


# ============================================================
# CLI / main
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase coherence experiments")

    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "medleydb_sample"])
    parser.add_argument("--medleydb_root", type=str, default="data/MedleyDB_sample")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--segment_seconds", type=float, default=2.0)
    parser.add_argument("--medley_max_tracks", type=int, default=0)
    parser.add_argument("--train_samples_per_epoch", type=int, default=1024)
    parser.add_argument("--val_samples_per_epoch", type=int, default=256)
    parser.add_argument("--log_every_n_batches", type=int, default=10)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_checkpoints", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_cfg = replace(
        Config(),
        dataset=args.dataset,
        medleydb_root=args.medleydb_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_runs=args.num_runs,
        seed=args.seed,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        medley_max_tracks=args.medley_max_tracks,
        train_samples_per_epoch=args.train_samples_per_epoch,
        val_samples_per_epoch=args.val_samples_per_epoch,
        log_every_n_batches=args.log_every_n_batches,
        checkpoint_dir=args.checkpoint_dir,
        save_checkpoints=args.save_checkpoints,
    )

    if base_cfg.save_checkpoints:
        ensure_checkpoint_dir(base_cfg)
        save_run_config(base_cfg)

    results = []

    for run_idx in range(base_cfg.num_runs):
        run_cfg = replace(base_cfg, seed=base_cfg.seed + run_idx)

        print("\n" + "=" * 60, flush=True)
        print(f"RUN {run_cfg.seed} | device={run_cfg.device} | dataset={run_cfg.dataset}", flush=True)
        print("=" * 60, flush=True)

        results.append(run_once(run_cfg))

    r = torch.tensor(results, dtype=torch.float32)

    names = ["Baseline", "Complex", "Align", "Full"]

    print("\nSUMMARY", flush=True)
    for i, name in enumerate(names):
        vals = r[:, i]
        print(f"{name:<10} {vals.mean().item():.6f} ± {vals.std(unbiased=True).item():.6f}", flush=True)

    print(f"\nDelta(Complex-Baseline): {(r[:,1].mean() - r[:,0].mean()).item():+.6f}", flush=True)
    print(f"Delta(Align-Baseline):   {(r[:,2].mean() - r[:,0].mean()).item():+.6f}", flush=True)
    print(f"Delta(Full-Baseline):    {(r[:,3].mean() - r[:,0].mean()).item():+.6f}", flush=True)
    print(f"Delta(Full-Align):       {(r[:,3].mean() - r[:,2].mean()).item():+.6f}", flush=True)


if __name__ == "__main__":
    main()