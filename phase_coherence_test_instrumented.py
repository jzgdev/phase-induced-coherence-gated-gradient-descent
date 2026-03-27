import argparse
import copy
import json
import math
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
    MedleyDBSamplePairs,
    PhaseStructuredDataset,
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
    metrics_dir: str = "metrics"
    grad_stats_every_n_batches: int = 1
    track_gradient_cosine: bool = True

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


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: Dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


@dataclass
class VariantLoggerBundle:
    train_step: JsonlLogger
    train_epoch: JsonlLogger
    val_epoch: JsonlLogger
    summary: JsonlLogger


# ============================================================
# Metrics helpers
# ============================================================

def ensure_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_run_dir(cfg: Config) -> Path:
    dataset_tag = cfg.dataset
    return ensure_dir(str(Path(cfg.metrics_dir) / dataset_tag / f"seed_{cfg.seed}"))


def create_variant_loggers(run_dir: Path, variant: str) -> VariantLoggerBundle:
    variant_dir = run_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    return VariantLoggerBundle(
        train_step=JsonlLogger(variant_dir / "train_steps.jsonl"),
        train_epoch=JsonlLogger(variant_dir / "train_epochs.jsonl"),
        val_epoch=JsonlLogger(variant_dir / "val_epochs.jsonl"),
        summary=JsonlLogger(variant_dir / "summary.jsonl"),
    )


def get_grad_vector(model: nn.Module) -> Optional[torch.Tensor]:
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())

    if not grads:
        return None

    return torch.cat(grads)


def compute_grad_stats(
    model: nn.Module,
    prev_grad_vector: Optional[torch.Tensor],
    track_gradient_cosine: bool,
) -> Tuple[float, Optional[float], Optional[torch.Tensor]]:
    grad_vec = get_grad_vector(model)
    if grad_vec is None:
        return 0.0, None, None

    grad_norm = grad_vec.norm().item()
    grad_cosine = None

    if track_gradient_cosine and prev_grad_vector is not None:
        denom = grad_vec.norm() * prev_grad_vector.norm()
        if denom.item() > 0:
            grad_cosine = torch.dot(grad_vec, prev_grad_vector).item() / (denom.item() + 1e-8)

    return grad_norm, grad_cosine, grad_vec.clone()


def mean_or_none(values: List[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def std_or_none(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean_val = sum(values) / len(values)
    var = sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)
    return float(math.sqrt(var))


def compute_steps_to_threshold(metrics: List[Dict], metric_key: str, threshold: float, mode: str) -> Optional[int]:
    for row in metrics:
        value = row.get(metric_key)
        if value is None:
            continue
        if mode == "max" and value >= threshold:
            return int(row["global_step"])
        if mode == "min" and value <= threshold:
            return int(row["global_step"])
    return None


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
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
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
) -> Dict[str, Optional[float]]:
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0
    batch_count = 0

    for batch in loader:
        if is_paired:
            x, _, y = batch
        else:
            x, y = batch

        x = x.to(cfg.device)
        y = y.to(cfg.device)

        _, logits = model(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)

        total += y.size(0)
        correct += (preds == y).sum().item()
        loss_sum += loss.item()
        batch_count += 1

    return {
        "val_accuracy": correct / total,
        "val_loss": loss_sum / max(batch_count, 1),
        "val_coherence": None,
    }


@torch.no_grad()
def evaluate_phase(
    model: PhaseModel,
    loader: DataLoader,
    cfg: Config,
    is_paired: bool,
    include_alignment_terms: bool = False,
) -> Dict[str, Optional[float]]:
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0
    batch_count = 0
    coh_vals: List[float] = []
    amp_losses: List[float] = []
    phase_losses: List[float] = []

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

        ce = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)

        total += y.size(0)
        correct += (preds == y).sum().item()

        coh = normalized_phase_coherence(amp, phase, amp_ref, phase_ref)
        coh_vals.append(coh.mean().item())

        if include_alignment_terms:
            amp_l = amplitude_alignment_loss(amp, amp_ref).mean()
            phase_l = phase_alignment_loss(phase, phase_ref).mean()
            amp_losses.append(amp_l.item())
            phase_losses.append(phase_l.item())
            loss = ce + cfg.lambda_amp * amp_l + cfg.lambda_phase * phase_l
        else:
            loss = ce

        loss_sum += loss.item()
        batch_count += 1

    return {
        "val_accuracy": correct / total,
        "val_loss": loss_sum / max(batch_count, 1),
        "val_coherence": sum(coh_vals) / len(coh_vals) if coh_vals else None,
        "val_amp_loss": mean_or_none(amp_losses),
        "val_phase_loss": mean_or_none(phase_losses),
    }


# ============================================================
# Generic training helpers
# ============================================================

def log_train_step(
    logger: JsonlLogger,
    cfg: Config,
    variant: str,
    epoch: int,
    batch_idx: int,
    global_step: int,
    step_time_sec: float,
    wall_time_sec: float,
    loss: float,
    grad_norm: Optional[float],
    grad_cosine: Optional[float],
    extra_metrics: Optional[Dict[str, Optional[float]]] = None,
) -> None:
    payload = {
        "variant": variant,
        "seed": cfg.seed,
        "dataset": cfg.dataset,
        "epoch": epoch + 1,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "step_time_sec": step_time_sec,
        "wall_time_sec": wall_time_sec,
        "train_loss": float(loss),
        "grad_norm": None if grad_norm is None else float(grad_norm),
        "grad_cosine": None if grad_cosine is None else float(grad_cosine),
    }
    if extra_metrics:
        payload.update(extra_metrics)
    logger.write(payload)


def log_epoch_metrics(
    logger: JsonlLogger,
    cfg: Config,
    variant: str,
    epoch: int,
    global_step: int,
    metrics: Dict[str, Optional[float]],
    wall_time_sec: float,
) -> None:
    payload = {
        "variant": variant,
        "seed": cfg.seed,
        "dataset": cfg.dataset,
        "epoch": epoch + 1,
        "global_step": global_step,
        "wall_time_sec": wall_time_sec,
    }
    payload.update(metrics)
    logger.write(payload)


def summarize_training_epoch(
    losses: List[float],
    grad_norms: List[float],
    grad_cosines: List[float],
    extra_metric_lists: Dict[str, List[float]],
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "train_loss_mean": mean_or_none(losses),
        "train_loss_std": std_or_none(losses),
        "grad_norm_mean": mean_or_none(grad_norms),
        "grad_norm_std": std_or_none(grad_norms),
        "grad_cosine_mean": mean_or_none(grad_cosines),
        "grad_cosine_std": std_or_none(grad_cosines),
    }
    for key, values in extra_metric_lists.items():
        metrics[f"{key}_mean"] = mean_or_none(values)
        metrics[f"{key}_std"] = std_or_none(values)
    return metrics


# ============================================================
# Training variants
# ============================================================

def train_baseline(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    is_paired: bool,
    loggers: VariantLoggerBundle,
) -> Tuple[BaselineModel, Dict[str, float]]:
    variant = "baseline"
    print("\n=== Baseline ===", flush=True)

    model = BaselineModel(cfg.hidden_dim, cfg.embed_dim, num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None
    best_epoch = -1
    global_step = 0
    run_start = time.time()
    val_history: List[Dict] = []

    prev_grad_vector: Optional[torch.Tensor] = None

    for epoch in range(cfg.epochs):
        model.train()
        epoch_start = time.time()
        loss_values: List[float] = []
        grad_norm_values: List[float] = []
        grad_cosine_values: List[float] = []

        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()

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
            grad_norm, grad_cosine, prev_grad_vector = compute_grad_stats(
                model,
                prev_grad_vector,
                cfg.track_gradient_cosine,
            )
            opt.step()

            global_step += 1
            loss_values.append(loss.item())
            grad_norm_values.append(grad_norm)
            if grad_cosine is not None:
                grad_cosine_values.append(grad_cosine)

            step_time_sec = time.time() - step_start
            wall_time_sec = time.time() - run_start

            if (batch_idx % cfg.grad_stats_every_n_batches) == 0:
                log_train_step(
                    loggers.train_step,
                    cfg,
                    variant,
                    epoch,
                    batch_idx,
                    global_step,
                    step_time_sec,
                    wall_time_sec,
                    loss.item(),
                    grad_norm,
                    grad_cosine,
                )

            if batch_idx % cfg.log_every_n_batches == 0:
                extra = f"grad={grad_norm:.4f}"
                if grad_cosine is not None:
                    extra += f" cos={grad_cosine:.4f}"
                log_batch_progress(
                    "Baseline",
                    epoch,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    epoch_start,
                    extra=extra,
                )

        train_epoch_metrics = summarize_training_epoch(
            loss_values,
            grad_norm_values,
            grad_cosine_values,
            extra_metric_lists={},
        )
        log_epoch_metrics(
            loggers.train_epoch,
            cfg,
            variant,
            epoch,
            global_step,
            train_epoch_metrics,
            wall_time_sec=time.time() - run_start,
        )

        val_metrics = evaluate_baseline(model, val_loader, cfg, is_paired)
        val_history.append({"epoch": epoch + 1, "global_step": global_step, **val_metrics})
        log_epoch_metrics(
            loggers.val_epoch,
            cfg,
            variant,
            epoch,
            global_step,
            val_metrics,
            wall_time_sec=time.time() - run_start,
        )

        val_acc = float(val_metrics["val_accuracy"])
        best_acc, maybe_state = maybe_save_best(best_acc, val_acc, model, variant, cfg)
        if maybe_state is not None:
            best_state = maybe_state
            best_epoch = epoch + 1

        print(
            f"[Baseline] epoch {epoch + 1:02d} done | "
            f"train_loss {train_epoch_metrics['train_loss_mean']:.4f} | "
            f"val_loss {val_metrics['val_loss']:.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"best {best_acc:.4f} | "
            f"epoch_time {time.time() - epoch_start:.1f}s",
            flush=True,
        )

    model = load_best_state(model, best_state)
    final_metrics = evaluate_baseline(model, val_loader, cfg, is_paired)
    target_acc = 0.90 * best_acc if best_acc > 0 else None
    steps_to_target_acc = None
    if target_acc is not None:
        steps_to_target_acc = compute_steps_to_threshold(val_history, "val_accuracy", target_acc, mode="max")

    summary = {
        "best_val_accuracy": best_acc,
        "best_epoch": best_epoch,
        "final_val_accuracy": float(final_metrics["val_accuracy"]),
        "final_val_loss": float(final_metrics["val_loss"]),
        "target_val_accuracy_90pct_best": target_acc,
        "steps_to_90pct_best_acc": steps_to_target_acc,
        "train_to_val_gap": None,
        "total_wall_time_sec": time.time() - run_start,
    }
    loggers.summary.write({"variant": variant, "seed": cfg.seed, "dataset": cfg.dataset, **summary})
    return model, summary


def train_complex_only(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    is_paired: bool,
    loggers: VariantLoggerBundle,
) -> Tuple[PhaseModel, Dict[str, float]]:
    variant = "complex"
    print("\n=== Complex Latent Only ===", flush=True)

    model = PhaseModel(cfg.hidden_dim, cfg.embed_dim, num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None
    best_epoch = -1
    global_step = 0
    run_start = time.time()
    val_history: List[Dict] = []

    prev_grad_vector: Optional[torch.Tensor] = None

    for epoch in range(cfg.epochs):
        model.train()
        epoch_start = time.time()
        loss_values: List[float] = []
        grad_norm_values: List[float] = []
        grad_cosine_values: List[float] = []
        coh_values: List[float] = []

        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()

            if is_paired:
                x, _, y = batch
                x_ref = x
            else:
                x, y = batch
                x_ref = x

            x = x.to(cfg.device)
            x_ref = x_ref.to(cfg.device)
            y = y.to(cfg.device)

            _, _, amp, phase, _, logits = model(x)
            _, _, amp_ref, phase_ref, _, _ = model(x_ref)
            coh = normalized_phase_coherence(amp, phase, amp_ref.detach(), phase_ref.detach())
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            grad_norm, grad_cosine, prev_grad_vector = compute_grad_stats(
                model,
                prev_grad_vector,
                cfg.track_gradient_cosine,
            )
            opt.step()

            global_step += 1
            loss_values.append(loss.item())
            grad_norm_values.append(grad_norm)
            coh_values.append(coh.mean().item())
            if grad_cosine is not None:
                grad_cosine_values.append(grad_cosine)

            step_time_sec = time.time() - step_start
            wall_time_sec = time.time() - run_start

            if (batch_idx % cfg.grad_stats_every_n_batches) == 0:
                log_train_step(
                    loggers.train_step,
                    cfg,
                    variant,
                    epoch,
                    batch_idx,
                    global_step,
                    step_time_sec,
                    wall_time_sec,
                    loss.item(),
                    grad_norm,
                    grad_cosine,
                    extra_metrics={"coherence_mean": coh.mean().item()},
                )

            if batch_idx % cfg.log_every_n_batches == 0:
                extra = f"grad={grad_norm:.4f} coh={coh.mean().item():.4f}"
                if grad_cosine is not None:
                    extra += f" cos={grad_cosine:.4f}"
                log_batch_progress(
                    "Complex",
                    epoch,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    epoch_start,
                    extra=extra,
                )

        train_epoch_metrics = summarize_training_epoch(
            loss_values,
            grad_norm_values,
            grad_cosine_values,
            extra_metric_lists={"coherence": coh_values},
        )
        log_epoch_metrics(
            loggers.train_epoch,
            cfg,
            variant,
            epoch,
            global_step,
            train_epoch_metrics,
            wall_time_sec=time.time() - run_start,
        )

        val_metrics = evaluate_phase(model, val_loader, cfg, is_paired, include_alignment_terms=False)
        val_history.append({"epoch": epoch + 1, "global_step": global_step, **val_metrics})
        log_epoch_metrics(
            loggers.val_epoch,
            cfg,
            variant,
            epoch,
            global_step,
            val_metrics,
            wall_time_sec=time.time() - run_start,
        )

        val_acc = float(val_metrics["val_accuracy"])
        best_acc, maybe_state = maybe_save_best(best_acc, val_acc, model, variant, cfg)
        if maybe_state is not None:
            best_state = maybe_state
            best_epoch = epoch + 1

        print(
            f"[Complex] epoch {epoch + 1:02d} done | "
            f"train_loss {train_epoch_metrics['train_loss_mean']:.4f} | "
            f"val_loss {val_metrics['val_loss']:.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"val_coh {val_metrics['val_coherence']:.4f} | "
            f"best {best_acc:.4f} | "
            f"epoch_time {time.time() - epoch_start:.1f}s",
            flush=True,
        )

    model = load_best_state(model, best_state)
    final_metrics = evaluate_phase(model, val_loader, cfg, is_paired, include_alignment_terms=False)
    target_acc = 0.90 * best_acc if best_acc > 0 else None
    steps_to_target_acc = None
    if target_acc is not None:
        steps_to_target_acc = compute_steps_to_threshold(val_history, "val_accuracy", target_acc, mode="max")

    summary = {
        "best_val_accuracy": best_acc,
        "best_epoch": best_epoch,
        "final_val_accuracy": float(final_metrics["val_accuracy"]),
        "final_val_loss": float(final_metrics["val_loss"]),
        "final_val_coherence": float(final_metrics["val_coherence"]),
        "target_val_accuracy_90pct_best": target_acc,
        "steps_to_90pct_best_acc": steps_to_target_acc,
        "train_to_val_gap": None,
        "total_wall_time_sec": time.time() - run_start,
    }
    loggers.summary.write({"variant": variant, "seed": cfg.seed, "dataset": cfg.dataset, **summary})
    return model, summary


def train_alignment_only(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    is_paired: bool,
    loggers: VariantLoggerBundle,
) -> Tuple[PhaseModel, Dict[str, float]]:
    variant = "align"
    print("\n=== Alignment Loss Only ===", flush=True)

    model = PhaseModel(cfg.hidden_dim, cfg.embed_dim, num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None
    best_epoch = -1
    global_step = 0
    run_start = time.time()
    val_history: List[Dict] = []

    prev_grad_vector: Optional[torch.Tensor] = None

    for epoch in range(cfg.epochs):
        model.train()
        epoch_start = time.time()
        loss_values: List[float] = []
        grad_norm_values: List[float] = []
        grad_cosine_values: List[float] = []
        coh_values: List[float] = []
        amp_values: List[float] = []
        phase_values: List[float] = []

        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()

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
            coh = normalized_phase_coherence(amp, phase, amp_ref.detach(), phase_ref.detach())
            loss = ce + cfg.lambda_amp * amp_l + cfg.lambda_phase * phase_l

            opt.zero_grad()
            loss.backward()
            grad_norm, grad_cosine, prev_grad_vector = compute_grad_stats(
                model,
                prev_grad_vector,
                cfg.track_gradient_cosine,
            )
            opt.step()

            global_step += 1
            loss_values.append(loss.item())
            grad_norm_values.append(grad_norm)
            amp_values.append(amp_l.item())
            phase_values.append(phase_l.item())
            coh_values.append(coh.mean().item())
            if grad_cosine is not None:
                grad_cosine_values.append(grad_cosine)

            step_time_sec = time.time() - step_start
            wall_time_sec = time.time() - run_start

            if (batch_idx % cfg.grad_stats_every_n_batches) == 0:
                log_train_step(
                    loggers.train_step,
                    cfg,
                    variant,
                    epoch,
                    batch_idx,
                    global_step,
                    step_time_sec,
                    wall_time_sec,
                    loss.item(),
                    grad_norm,
                    grad_cosine,
                    extra_metrics={
                        "coherence_mean": coh.mean().item(),
                        "amp_loss": amp_l.item(),
                        "phase_loss": phase_l.item(),
                    },
                )

            if batch_idx % cfg.log_every_n_batches == 0:
                extra = (
                    f"grad={grad_norm:.4f} coh={coh.mean().item():.4f} "
                    f"amp={amp_l.item():.4f} phase={phase_l.item():.4f}"
                )
                if grad_cosine is not None:
                    extra += f" cos={grad_cosine:.4f}"
                log_batch_progress(
                    "Align",
                    epoch,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    epoch_start,
                    extra=extra,
                )

        train_epoch_metrics = summarize_training_epoch(
            loss_values,
            grad_norm_values,
            grad_cosine_values,
            extra_metric_lists={
                "coherence": coh_values,
                "amp_loss": amp_values,
                "phase_loss": phase_values,
            },
        )
        log_epoch_metrics(
            loggers.train_epoch,
            cfg,
            variant,
            epoch,
            global_step,
            train_epoch_metrics,
            wall_time_sec=time.time() - run_start,
        )

        val_metrics = evaluate_phase(model, val_loader, cfg, is_paired, include_alignment_terms=True)
        val_history.append({"epoch": epoch + 1, "global_step": global_step, **val_metrics})
        log_epoch_metrics(
            loggers.val_epoch,
            cfg,
            variant,
            epoch,
            global_step,
            val_metrics,
            wall_time_sec=time.time() - run_start,
        )

        val_acc = float(val_metrics["val_accuracy"])
        best_acc, maybe_state = maybe_save_best(best_acc, val_acc, model, variant, cfg)
        if maybe_state is not None:
            best_state = maybe_state
            best_epoch = epoch + 1

        print(
            f"[Align] epoch {epoch + 1:02d} done | "
            f"train_loss {train_epoch_metrics['train_loss_mean']:.4f} | "
            f"val_loss {val_metrics['val_loss']:.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"val_coh {val_metrics['val_coherence']:.4f} | "
            f"best {best_acc:.4f} | "
            f"epoch_time {time.time() - epoch_start:.1f}s",
            flush=True,
        )

    model = load_best_state(model, best_state)
    final_metrics = evaluate_phase(model, val_loader, cfg, is_paired, include_alignment_terms=True)
    target_acc = 0.90 * best_acc if best_acc > 0 else None
    steps_to_target_acc = None
    if target_acc is not None:
        steps_to_target_acc = compute_steps_to_threshold(val_history, "val_accuracy", target_acc, mode="max")

    summary = {
        "best_val_accuracy": best_acc,
        "best_epoch": best_epoch,
        "final_val_accuracy": float(final_metrics["val_accuracy"]),
        "final_val_loss": float(final_metrics["val_loss"]),
        "final_val_coherence": float(final_metrics["val_coherence"]),
        "final_val_amp_loss": float(final_metrics["val_amp_loss"]),
        "final_val_phase_loss": float(final_metrics["val_phase_loss"]),
        "target_val_accuracy_90pct_best": target_acc,
        "steps_to_90pct_best_acc": steps_to_target_acc,
        "train_to_val_gap": None,
        "total_wall_time_sec": time.time() - run_start,
    }
    loggers.summary.write({"variant": variant, "seed": cfg.seed, "dataset": cfg.dataset, **summary})
    return model, summary


def train_full(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    is_paired: bool,
    loggers: VariantLoggerBundle,
) -> Tuple[PhaseModel, Dict[str, float]]:
    variant = "full"
    print("\n=== Full Method (Alignment + Gentle Coherence Gating) ===", flush=True)

    model = PhaseModel(cfg.hidden_dim, cfg.embed_dim, num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None
    best_epoch = -1
    global_step = 0
    run_start = time.time()
    val_history: List[Dict] = []

    prev_grad_vector: Optional[torch.Tensor] = None

    for epoch in range(cfg.epochs):
        model.train()
        epoch_start = time.time()
        loss_values: List[float] = []
        grad_norm_values: List[float] = []
        grad_cosine_values: List[float] = []
        gate_values: List[float] = []
        coh_values: List[float] = []
        amp_values: List[float] = []
        phase_values: List[float] = []

        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()

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
            coh = normalized_phase_coherence(amp, phase, amp_ref.detach(), phase_ref.detach())

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
            grad_norm, grad_cosine, prev_grad_vector = compute_grad_stats(
                model,
                prev_grad_vector,
                cfg.track_gradient_cosine,
            )
            opt.step()

            global_step += 1
            loss_values.append(loss.item())
            grad_norm_values.append(grad_norm)
            gate_values.append(alpha.mean().item())
            coh_values.append(coh.mean().item())
            amp_values.append(per_sample_amp.mean().item())
            phase_values.append(per_sample_phase.mean().item())
            if grad_cosine is not None:
                grad_cosine_values.append(grad_cosine)

            step_time_sec = time.time() - step_start
            wall_time_sec = time.time() - run_start

            if (batch_idx % cfg.grad_stats_every_n_batches) == 0:
                log_train_step(
                    loggers.train_step,
                    cfg,
                    variant,
                    epoch,
                    batch_idx,
                    global_step,
                    step_time_sec,
                    wall_time_sec,
                    loss.item(),
                    grad_norm,
                    grad_cosine,
                    extra_metrics={
                        "gate_mean": alpha.mean().item(),
                        "coherence_mean": coh.mean().item(),
                        "amp_loss": per_sample_amp.mean().item(),
                        "phase_loss": per_sample_phase.mean().item(),
                    },
                )

            if batch_idx % cfg.log_every_n_batches == 0:
                extra = (
                    f"grad={grad_norm:.4f} gate={alpha.mean().item():.4f} "
                    f"coh={coh.mean().item():.4f} "
                    f"amp={per_sample_amp.mean().item():.4f} "
                    f"phase={per_sample_phase.mean().item():.4f}"
                )
                if grad_cosine is not None:
                    extra += f" cos={grad_cosine:.4f}"
                log_batch_progress(
                    "Full",
                    epoch,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    epoch_start,
                    extra=extra,
                )

        train_epoch_metrics = summarize_training_epoch(
            loss_values,
            grad_norm_values,
            grad_cosine_values,
            extra_metric_lists={
                "gate": gate_values,
                "coherence": coh_values,
                "amp_loss": amp_values,
                "phase_loss": phase_values,
            },
        )
        log_epoch_metrics(
            loggers.train_epoch,
            cfg,
            variant,
            epoch,
            global_step,
            train_epoch_metrics,
            wall_time_sec=time.time() - run_start,
        )

        val_metrics = evaluate_phase(model, val_loader, cfg, is_paired, include_alignment_terms=True)
        val_history.append({"epoch": epoch + 1, "global_step": global_step, **val_metrics})
        log_epoch_metrics(
            loggers.val_epoch,
            cfg,
            variant,
            epoch,
            global_step,
            val_metrics,
            wall_time_sec=time.time() - run_start,
        )

        val_acc = float(val_metrics["val_accuracy"])
        best_acc, maybe_state = maybe_save_best(best_acc, val_acc, model, variant, cfg)
        if maybe_state is not None:
            best_state = maybe_state
            best_epoch = epoch + 1

        print(
            f"[Full] epoch {epoch + 1:02d} done | "
            f"train_loss {train_epoch_metrics['train_loss_mean']:.4f} | "
            f"train_gate {train_epoch_metrics['gate_mean']:.4f} | "
            f"train_coh {train_epoch_metrics['coherence_mean']:.4f} | "
            f"val_loss {val_metrics['val_loss']:.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"val_coh {val_metrics['val_coherence']:.4f} | "
            f"best {best_acc:.4f} | "
            f"epoch_time {time.time() - epoch_start:.1f}s",
            flush=True,
        )

    model = load_best_state(model, best_state)
    final_metrics = evaluate_phase(model, val_loader, cfg, is_paired, include_alignment_terms=True)
    target_acc = 0.90 * best_acc if best_acc > 0 else None
    steps_to_target_acc = None
    if target_acc is not None:
        steps_to_target_acc = compute_steps_to_threshold(val_history, "val_accuracy", target_acc, mode="max")

    summary = {
        "best_val_accuracy": best_acc,
        "best_epoch": best_epoch,
        "final_val_accuracy": float(final_metrics["val_accuracy"]),
        "final_val_loss": float(final_metrics["val_loss"]),
        "final_val_coherence": float(final_metrics["val_coherence"]),
        "final_val_amp_loss": float(final_metrics['val_amp_loss']),
        "final_val_phase_loss": float(final_metrics['val_phase_loss']),
        "target_val_accuracy_90pct_best": target_acc,
        "steps_to_90pct_best_acc": steps_to_target_acc,
        "train_to_val_gap": None,
        "total_wall_time_sec": time.time() - run_start,
    }
    loggers.summary.write({"variant": variant, "seed": cfg.seed, "dataset": cfg.dataset, **summary})
    return model, summary


# ============================================================
# Experiment runner
# ============================================================

def run_once(cfg: Config) -> Dict[str, Dict[str, float]]:
    set_seed(cfg.seed)

    train_loader, val_loader, num_classes, is_paired = make_loaders(cfg)
    run_dir = build_run_dir(cfg)

    baseline, baseline_summary = train_baseline(
        cfg,
        train_loader,
        val_loader,
        num_classes,
        is_paired,
        create_variant_loggers(run_dir, "baseline"),
    )
    complex_only, complex_summary = train_complex_only(
        cfg,
        train_loader,
        val_loader,
        num_classes,
        is_paired,
        create_variant_loggers(run_dir, "complex"),
    )
    alignment_only, align_summary = train_alignment_only(
        cfg,
        train_loader,
        val_loader,
        num_classes,
        is_paired,
        create_variant_loggers(run_dir, "align"),
    )
    full, full_summary = train_full(
        cfg,
        train_loader,
        val_loader,
        num_classes,
        is_paired,
        create_variant_loggers(run_dir, "full"),
    )

    baseline_eval = evaluate_baseline(baseline, val_loader, cfg, is_paired)
    complex_eval = evaluate_phase(complex_only, val_loader, cfg, is_paired, include_alignment_terms=False)
    align_eval = evaluate_phase(alignment_only, val_loader, cfg, is_paired, include_alignment_terms=True)
    full_eval = evaluate_phase(full, val_loader, cfg, is_paired, include_alignment_terms=True)

    print("\nRESULTS", flush=True)
    print(f"Baseline: {baseline_eval['val_accuracy']:.6f}", flush=True)
    print(f"Complex:  {complex_eval['val_accuracy']:.6f}", flush=True)
    print(f"Align:    {align_eval['val_accuracy']:.6f}", flush=True)
    print(f"Full:     {full_eval['val_accuracy']:.6f}", flush=True)
    print(f"Full coh: {full_eval['val_coherence']:.6f}", flush=True)

    run_summary = {
        "baseline": baseline_summary,
        "complex": complex_summary,
        "align": align_summary,
        "full": full_summary,
    }

    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    return run_summary


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
    parser.add_argument("--grad_stats_every_n_batches", type=int, default=1)
    parser.add_argument("--metrics_dir", type=str, default="metrics")
    parser.add_argument("--disable_gradient_cosine", action="store_true")

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
        grad_stats_every_n_batches=args.grad_stats_every_n_batches,
        metrics_dir=args.metrics_dir,
        track_gradient_cosine=not args.disable_gradient_cosine,
        checkpoint_dir=args.checkpoint_dir,
        save_checkpoints=args.save_checkpoints,
    )

    if base_cfg.save_checkpoints:
        ensure_checkpoint_dir(base_cfg)
        save_run_config(base_cfg)

    all_run_summaries: List[Dict[str, Dict[str, float]]] = []

    for run_idx in range(base_cfg.num_runs):
        run_cfg = replace(base_cfg, seed=base_cfg.seed + run_idx)

        print("\n" + "=" * 60, flush=True)
        print(f"RUN {run_cfg.seed} | device={run_cfg.device} | dataset={run_cfg.dataset}", flush=True)
        print("=" * 60, flush=True)

        all_run_summaries.append(run_once(run_cfg))

    names = ["baseline", "complex", "align", "full"]

    print("\nSUMMARY", flush=True)
    aggregate_summary: Dict[str, Dict[str, Optional[float]]] = {}

    for name in names:
        vals = torch.tensor(
            [run[name]["final_val_accuracy"] for run in all_run_summaries],
            dtype=torch.float32,
        )
        mean_acc = vals.mean().item()
        std_acc = vals.std(unbiased=True).item() if len(vals) > 1 else 0.0
        aggregate_summary[name] = {
            "final_val_accuracy_mean": mean_acc,
            "final_val_accuracy_std": std_acc,
            "best_val_accuracy_mean": float(sum(run[name]["best_val_accuracy"] for run in all_run_summaries) / len(all_run_summaries)),
            "wall_time_mean_sec": float(sum(run[name]["total_wall_time_sec"] for run in all_run_summaries) / len(all_run_summaries)),
        }
        print(f"{name.capitalize():<10} {mean_acc:.6f} ± {std_acc:.6f}", flush=True)

    baseline_mean = aggregate_summary["baseline"]["final_val_accuracy_mean"]
    print(
        f"\nDelta(Complex-Baseline): {aggregate_summary['complex']['final_val_accuracy_mean'] - baseline_mean:+.6f}",
        flush=True,
    )
    print(
        f"Delta(Align-Baseline):   {aggregate_summary['align']['final_val_accuracy_mean'] - baseline_mean:+.6f}",
        flush=True,
    )
    print(
        f"Delta(Full-Baseline):    {aggregate_summary['full']['final_val_accuracy_mean'] - baseline_mean:+.6f}",
        flush=True,
    )
    print(
        f"Delta(Full-Align):       {aggregate_summary['full']['final_val_accuracy_mean'] - aggregate_summary['align']['final_val_accuracy_mean']:+.6f}",
        flush=True,
    )

    metrics_root = ensure_dir(base_cfg.metrics_dir)
    summary_path = metrics_root / f"summary_{base_cfg.dataset}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(base_cfg),
                "aggregate_summary": aggregate_summary,
                "run_summaries": all_run_summaries,
            },
            f,
            indent=2,
        )
    print(f"\nSaved aggregate summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
