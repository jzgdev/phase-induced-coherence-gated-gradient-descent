import argparse
import copy
import importlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import FMASmallPairs, MedleyDBSamplePairs, PhaseStructuredDataset


ALL_VARIANTS: Tuple[str, ...] = ("baseline", "complex", "align", "gate_only", "full")
DEFAULT_ANALYSIS_VARIANTS: Tuple[str, ...] = ("baseline", "full")


# ============================================================
# Config
# ============================================================


@dataclass
class Config:
    dataset: str = "synthetic"  # synthetic | medleydb_sample
    medleydb_root: str = "data/MedleyDB_sample"
    fma_root: str = "data/fma"
    fma_metadata_root: str = ""

    seq_len: int = 256
    num_classes: int = 4

    samples_per_class_train: int = 1500
    samples_per_class_val: int = 400
    synthetic_reference_mode: str = "paired_view"  # self | paired_view

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

    base_f1: float = 6.0
    base_f2: float = 12.0
    freq_jitter: float = 0.5
    amp_low: float = 0.7
    amp_high: float = 1.3
    phase_jitter: float = 0.35
    noise_std: float = 0.2

    sample_rate: int = 48000
    segment_seconds: float = 2.0
    medley_max_tracks: int = 0
    fma_max_tracks: int = 0
    eval_protocol: str = "same_track_fixed"

    lambda_amp: float = 0.10
    lambda_phase: float = 0.10

    gate_beta: float = 2.0
    gate_gamma: float = 0.20
    gate_warmup_epochs: int = 5

    log_every_n_batches: int = 10
    step_log_every_n_batches: int = 1
    results_dir: str = "results"
    variants: Tuple[str, ...] = ALL_VARIANTS
    analysis_variants: Tuple[str, ...] = DEFAULT_ANALYSIS_VARIANTS
    loss_target: float = 1.0
    grad_variance_steps: int = 200
    rolling_window: int = 25
    generate_plots: bool = False
    track_gradient_cosine: bool = True

    checkpoint_dir: str = "checkpoints"
    save_checkpoints: bool = False


@dataclass
class VariantSpec:
    name: str
    title: str
    uses_alignment: bool = False
    uses_gating: bool = False


@dataclass
class VariantLoggerBundle:
    train_step: "JsonlLogger"
    train_epoch: "JsonlLogger"
    val_epoch: "JsonlLogger"
    summary: "JsonlLogger"


@dataclass
class VariantRunArtifacts:
    summary: Dict[str, object]
    step_history: List[Dict[str, object]]
    train_epoch_history: List[Dict[str, object]]
    val_history: List[Dict[str, object]]


VARIANT_SPECS: Dict[str, VariantSpec] = {
    "baseline": VariantSpec(name="baseline", title="Baseline"),
    "complex": VariantSpec(name="complex", title="Complex Latent Only"),
    "align": VariantSpec(name="align", title="Alignment Loss Only", uses_alignment=True),
    "gate_only": VariantSpec(name="gate_only", title="Gate Only", uses_gating=True),
    "full": VariantSpec(name="full", title="Full Method", uses_alignment=True, uses_gating=True),
}


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


def deep_copy_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return copy.deepcopy(state)


def capture_initial_states(
    cfg: Config,
    num_classes: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    set_seed(cfg.seed + 101)
    baseline_state = deep_copy_state_dict(
        BaselineModel(cfg.hidden_dim, cfg.embed_dim, num_classes).state_dict()
    )

    set_seed(cfg.seed + 202)
    phase_state = deep_copy_state_dict(
        PhaseModel(cfg.hidden_dim, cfg.embed_dim, num_classes).state_dict()
    )

    return baseline_state, phase_state


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
        if self.path.exists():
            self.path.unlink()

    def write(self, payload: Dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


def ensure_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def build_run_dir(cfg: Config) -> Path:
    return ensure_dir(
        str(Path(cfg.results_dir) / cfg.dataset / cfg.eval_protocol / f"seed_{cfg.seed}")
    )


def create_variant_loggers(run_dir: Path, variant: str) -> VariantLoggerBundle:
    variant_dir = run_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    return VariantLoggerBundle(
        train_step=JsonlLogger(variant_dir / "train_steps.jsonl"),
        train_epoch=JsonlLogger(variant_dir / "train_epochs.jsonl"),
        val_epoch=JsonlLogger(variant_dir / "val_epochs.jsonl"),
        summary=JsonlLogger(variant_dir / "summary.jsonl"),
    )


def save_run_config(cfg: Config, run_dir: Path) -> None:
    write_json(run_dir / "config.json", asdict(cfg))


def log_train_step(
    logger: JsonlLogger,
    cfg: Config,
    variant: str,
    epoch: int,
    batch_idx: int,
    global_step: int,
    step_time_sec: float,
    wall_time_sec: float,
    train_loss: float,
    train_ce_loss: float,
    grad_norm: Optional[float],
    grad_cosine: Optional[float],
    extra_metrics: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "variant": variant,
        "seed": cfg.seed,
        "dataset": cfg.dataset,
        "epoch": epoch + 1,
        "batch_idx": batch_idx,
        "global_step": global_step,
        "step_time_sec": step_time_sec,
        "wall_time_sec": wall_time_sec,
        "train_loss": float(train_loss),
        "train_ce_loss": float(train_ce_loss),
        "grad_norm": None if grad_norm is None else float(grad_norm),
        "grad_cosine": None if grad_cosine is None else float(grad_cosine),
    }
    if extra_metrics:
        payload.update(extra_metrics)
    logger.write(payload)
    return payload


def log_epoch_metrics(
    logger: JsonlLogger,
    cfg: Config,
    variant: str,
    epoch: int,
    global_step: int,
    wall_time_sec: float,
    metrics: Dict[str, Optional[float]],
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "variant": variant,
        "seed": cfg.seed,
        "dataset": cfg.dataset,
        "epoch": epoch + 1,
        "global_step": global_step,
        "wall_time_sec": wall_time_sec,
    }
    payload.update(metrics)
    logger.write(payload)
    return payload


# ============================================================
# Metrics helpers
# ============================================================


def get_grad_vector(model: nn.Module) -> Optional[torch.Tensor]:
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().flatten())

    if not grads:
        return None

    return torch.cat(grads)


def compute_grad_stats(
    model: nn.Module,
    prev_grad_vector: Optional[torch.Tensor],
    track_gradient_cosine: bool,
) -> Tuple[Optional[float], Optional[float], Optional[torch.Tensor]]:
    grad_vec = get_grad_vector(model)
    if grad_vec is None:
        return None, None, None

    grad_norm = float(grad_vec.norm().item())
    grad_cosine = None

    if track_gradient_cosine and prev_grad_vector is not None:
        denom = grad_vec.norm() * prev_grad_vector.norm()
        if float(denom.item()) > 0:
            grad_cosine = float(torch.dot(grad_vec, prev_grad_vector).item() / (denom.item() + 1e-8))

    return grad_norm, grad_cosine, grad_vec.clone()


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def std_or_none(values: Sequence[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean_val = sum(values) / len(values)
    var = sum((value - mean_val) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(var))


def variance_or_none(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    mean_val = sum(values) / len(values)
    return float(sum((value - mean_val) ** 2 for value in values) / len(values))


def compute_gradient_variance(values: Sequence[float], max_steps: int) -> Optional[float]:
    if max_steps <= 0:
        return None
    subset = list(values[:max_steps])
    return variance_or_none(subset)


def moving_average(values: Sequence[float], window: int) -> List[float]:
    if not values:
        return []

    window = max(int(window), 1)
    smoothed: List[float] = []
    running_sum = 0.0

    for idx, value in enumerate(values):
        running_sum += value
        if idx >= window:
            running_sum -= values[idx - window]
        denom = min(idx + 1, window)
        smoothed.append(running_sum / denom)

    return smoothed


def rolling_variance(values: Sequence[float], window: int) -> List[Optional[float]]:
    if not values:
        return []

    window = max(int(window), 1)
    result: List[Optional[float]] = []

    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start:idx + 1]
        result.append(variance_or_none(chunk))

    return result


def compute_steps_to_threshold(
    metrics: Sequence[Dict[str, object]],
    metric_key: str,
    threshold: float,
    mode: str,
) -> Optional[int]:
    for row in metrics:
        value = row.get(metric_key)
        if value is None:
            continue
        numeric = float(value)
        if mode == "max" and numeric >= threshold:
            return int(row["global_step"])
        if mode == "min" and numeric <= threshold:
            return int(row["global_step"])
    return None


def compute_summary_stats(values: Sequence[float]) -> Dict[str, object]:
    values_list = list(values)
    mean = sum(values_list) / len(values_list)

    if len(values_list) >= 2:
        std = float(torch.tensor(values_list, dtype=torch.float32).std(unbiased=True).item())
    else:
        std = None

    return {"mean": mean, "std": std, "n": len(values_list)}


def compute_optional_summary_stats(values: Sequence[Optional[float]]) -> Dict[str, object]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return {"mean": None, "std": None, "n": 0}
    return compute_summary_stats(filtered)


def format_summary_line(name: str, values: Sequence[float]) -> str:
    stats = compute_summary_stats(values)
    if stats["std"] is None:
        return f"{name:<10} {stats['mean']:.6f} (n=1)"
    return f"{name:<10} {stats['mean']:.6f} ± {stats['std']:.6f}"


def summarize_training_epoch(
    train_loss_values: Sequence[float],
    train_ce_values: Sequence[float],
    grad_norm_values: Sequence[float],
    grad_cosine_values: Sequence[float],
    extra_metric_lists: Dict[str, Sequence[float]],
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "train_loss_mean": mean_or_none(train_loss_values),
        "train_loss_std": std_or_none(train_loss_values),
        "train_ce_mean": mean_or_none(train_ce_values),
        "train_ce_std": std_or_none(train_ce_values),
        "grad_norm_mean": mean_or_none(grad_norm_values),
        "grad_norm_std": std_or_none(grad_norm_values),
        "grad_cosine_mean": mean_or_none(grad_cosine_values),
        "grad_cosine_std": std_or_none(grad_cosine_values),
    }
    for key, values in extra_metric_lists.items():
        metrics[f"{key}_mean"] = mean_or_none(values)
        metrics[f"{key}_std"] = std_or_none(values)
    return metrics


# ============================================================
# Checkpoint helpers
# ============================================================


def ensure_checkpoint_dir(cfg: Config) -> Path:
    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def maybe_save_best(
    best_acc: float,
    current_acc: float,
    model: nn.Module,
    model_name: str,
    cfg: Config,
) -> Tuple[float, Optional[Dict[str, torch.Tensor]]]:
    if current_acc > best_acc:
        state = deep_copy_state_dict(model.state_dict())

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
    return 1.0 + gamma * torch.tanh(beta * coh_centered)


# ============================================================
# Dataset helpers
# ============================================================


def parse_variants(value: str) -> Tuple[str, ...]:
    variants = tuple(part.strip() for part in value.split(",") if part.strip())
    if not variants:
        raise argparse.ArgumentTypeError("At least one variant must be specified.")
    return variants


def validate_config(cfg: Config) -> None:
    if cfg.eval_protocol != "same_track_fixed":
        raise ValueError(
            f"Unsupported eval_protocol: {cfg.eval_protocol}. "
            "This repo currently implements same_track_fixed only."
        )

    unknown_variants = [variant for variant in cfg.variants if variant not in VARIANT_SPECS]
    if unknown_variants:
        raise ValueError(f"Unknown variants requested: {unknown_variants}")

    unknown_analysis = [variant for variant in cfg.analysis_variants if variant not in VARIANT_SPECS]
    if unknown_analysis:
        raise ValueError(f"Unknown analysis_variants requested: {unknown_analysis}")

    missing = [variant for variant in cfg.analysis_variants if variant not in cfg.variants]
    if missing:
        raise ValueError(f"analysis_variants must be a subset of variants: {missing}")


def build_datasets(cfg: Config):
    if cfg.dataset == "synthetic":
        train_ds = PhaseStructuredDataset(train=True, cfg=cfg)
        val_ds = PhaseStructuredDataset(train=False, cfg=cfg)
        return train_ds, val_ds, cfg.num_classes

    if cfg.dataset == "medleydb_sample":
        train_ds = MedleyDBSamplePairs(
            root=cfg.medleydb_root,
            sample_rate=cfg.sample_rate,
            segment_seconds=cfg.segment_seconds,
            max_tracks=cfg.medley_max_tracks,
            samples_per_epoch=cfg.train_samples_per_epoch,
            seed=cfg.seed,
            split="train",
        )
        val_ds = MedleyDBSamplePairs(
            root=cfg.medleydb_root,
            sample_rate=cfg.sample_rate,
            segment_seconds=cfg.segment_seconds,
            max_tracks=cfg.medley_max_tracks,
            samples_per_epoch=cfg.val_samples_per_epoch,
            seed=cfg.seed + 10_000,
            track_names=train_ds.track_names,
            split="val",
        )
        return train_ds, val_ds, train_ds.num_classes

    if cfg.dataset == "fma_small":
        metadata_root = cfg.fma_metadata_root or None
        train_ds = FMASmallPairs(
            root=cfg.fma_root,
            metadata_root=metadata_root,
            sample_rate=cfg.sample_rate,
            segment_seconds=cfg.segment_seconds,
            max_tracks=cfg.fma_max_tracks,
            samples_per_epoch=cfg.train_samples_per_epoch,
            seed=cfg.seed,
            split="train",
        )
        val_ds = FMASmallPairs(
            root=cfg.fma_root,
            metadata_root=metadata_root,
            sample_rate=cfg.sample_rate,
            segment_seconds=cfg.segment_seconds,
            max_tracks=cfg.fma_max_tracks,
            samples_per_epoch=cfg.val_samples_per_epoch,
            seed=cfg.seed + 10_000,
            split="val",
            label_names=train_ds.label_names,
        )
        return train_ds, val_ds, train_ds.num_classes

    raise ValueError(f"Unknown dataset: {cfg.dataset}")


def make_loaders(cfg: Config):
    train_ds, val_ds, num_classes = build_datasets(cfg)

    print(
        f"Built datasets | train_samples={len(train_ds)} | "
        f"val_samples={len(val_ds)} | num_classes={num_classes} | paired=True",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=False,
    )

    print(
        f"Built loaders | train_batches={len(train_loader)} | "
        f"val_batches={len(val_loader)} | batch_size={cfg.batch_size}",
        flush=True,
    )

    return train_loader, val_loader, num_classes


def maybe_set_epoch(loader: DataLoader, epoch: int) -> None:
    dataset = loader.dataset
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) == 3:
        x, x_ref, y = batch
        return x, x_ref, y

    if len(batch) == 2:
        x, y = batch
        return x, x, y

    raise ValueError(f"Unsupported batch format with {len(batch)} entries")


def get_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ============================================================
# Evaluation
# ============================================================


@torch.no_grad()
def evaluate_baseline(
    model: BaselineModel,
    loader: DataLoader,
    cfg: Config,
) -> Dict[str, Optional[float]]:
    model.eval()
    device = get_model_device(model)

    total = 0
    correct = 0
    ce_sum = 0.0
    batch_count = 0

    for batch in loader:
        x, _, y = unpack_batch(batch)
        x = x.to(device)
        y = y.to(device)

        _, logits = model(x)
        ce = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)

        total += y.size(0)
        correct += (preds == y).sum().item()
        ce_sum += ce.item()
        batch_count += 1

    return {
        "val_accuracy": correct / total,
        "val_loss": ce_sum / max(batch_count, 1),
        "val_total_loss": ce_sum / max(batch_count, 1),
        "val_coherence": None,
        "val_amp_loss": None,
        "val_phase_loss": None,
    }


@torch.no_grad()
def evaluate_phase(
    model: PhaseModel,
    loader: DataLoader,
    cfg: Config,
    include_alignment_terms: bool = False,
) -> Dict[str, Optional[float]]:
    model.eval()
    device = get_model_device(model)

    total = 0
    correct = 0
    ce_sum = 0.0
    total_loss_sum = 0.0
    batch_count = 0
    coh_vals: List[float] = []
    amp_losses: List[float] = []
    phase_losses: List[float] = []

    for batch in loader:
        x, x_ref, y = unpack_batch(batch)
        x = x.to(device)
        x_ref = x_ref.to(device)
        y = y.to(device)

        _, _, amp, phase, _, logits = model(x)
        _, _, amp_ref, phase_ref, _, _ = model(x_ref)

        ce = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)

        total += y.size(0)
        correct += (preds == y).sum().item()
        ce_sum += ce.item()

        coh = normalized_phase_coherence(amp, phase, amp_ref, phase_ref)
        coh_vals.append(coh.mean().item())

        if include_alignment_terms:
            amp_l = amplitude_alignment_loss(amp, amp_ref).mean()
            phase_l = phase_alignment_loss(phase, phase_ref).mean()
            amp_losses.append(amp_l.item())
            phase_losses.append(phase_l.item())
            total_loss = ce + cfg.lambda_amp * amp_l + cfg.lambda_phase * phase_l
        else:
            total_loss = ce

        total_loss_sum += total_loss.item()
        batch_count += 1

    return {
        "val_accuracy": correct / total,
        "val_loss": ce_sum / max(batch_count, 1),
        "val_total_loss": total_loss_sum / max(batch_count, 1),
        "val_coherence": sum(coh_vals) / len(coh_vals) if coh_vals else None,
        "val_amp_loss": mean_or_none(amp_losses),
        "val_phase_loss": mean_or_none(phase_losses),
    }


# ============================================================
# Training variants
# ============================================================


def finalize_variant_summary(
    cfg: Config,
    spec: VariantSpec,
    best_acc: float,
    best_epoch: int,
    step_history: List[Dict[str, object]],
    train_epoch_history: List[Dict[str, object]],
    val_history: List[Dict[str, object]],
    total_wall_time_sec: float,
    last_val_metrics: Dict[str, Optional[float]],
) -> Dict[str, object]:
    last_train_epoch = train_epoch_history[-1]
    steps_to_loss_target = compute_steps_to_threshold(
        val_history,
        "val_loss",
        cfg.loss_target,
        mode="min",
    )
    hit_loss_target = steps_to_loss_target is not None

    grad_norm_values = [
        float(row["grad_norm"])
        for row in step_history
        if row.get("grad_norm") is not None
    ]
    grad_norm_variance = compute_gradient_variance(grad_norm_values, cfg.grad_variance_steps)

    summary: Dict[str, object] = {
        "variant": spec.name,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "final_val_accuracy": float(last_val_metrics["val_accuracy"]),
        "final_val_loss": float(last_val_metrics["val_loss"]),
        "final_val_total_loss": float(last_val_metrics["val_total_loss"]),
        "final_val_coherence": last_val_metrics["val_coherence"],
        "final_train_loss": last_train_epoch["train_loss_mean"],
        "final_train_ce_loss": last_train_epoch["train_ce_mean"],
        "final_generalization_gap": last_train_epoch["generalization_gap"],
        "loss_target": cfg.loss_target,
        "steps_to_loss_target": steps_to_loss_target,
        "hit_loss_target": hit_loss_target,
        "grad_norm_variance": grad_norm_variance,
        "grad_variance_steps": cfg.grad_variance_steps,
        "total_wall_time_sec": total_wall_time_sec,
    }

    if spec.uses_alignment:
        summary["final_val_amp_loss"] = last_val_metrics["val_amp_loss"]
        summary["final_val_phase_loss"] = last_val_metrics["val_phase_loss"]

    if spec.uses_gating:
        summary["final_train_gate_mean"] = last_train_epoch.get("gate_mean")

    return summary


def train_baseline(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    initial_state: Dict[str, torch.Tensor],
    loggers: VariantLoggerBundle,
) -> Tuple[BaselineModel, VariantRunArtifacts]:
    spec = VARIANT_SPECS["baseline"]
    print(f"\n=== {spec.title} ===", flush=True)

    model = BaselineModel(cfg.hidden_dim, cfg.embed_dim, num_classes).to(cfg.device)
    model.load_state_dict(initial_state)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None
    best_epoch = -1
    global_step = 0
    run_start = time.time()

    step_history: List[Dict[str, object]] = []
    train_epoch_history: List[Dict[str, object]] = []
    val_history: List[Dict[str, object]] = []

    prev_grad_vector: Optional[torch.Tensor] = None
    last_val_metrics: Dict[str, Optional[float]] = {
        "val_accuracy": 0.0,
        "val_loss": 0.0,
        "val_total_loss": 0.0,
        "val_coherence": None,
        "val_amp_loss": None,
        "val_phase_loss": None,
    }

    for epoch in range(cfg.epochs):
        maybe_set_epoch(train_loader, epoch)
        model.train()
        epoch_start = time.time()

        train_loss_values: List[float] = []
        train_ce_values: List[float] = []
        grad_norm_values: List[float] = []
        grad_cosine_values: List[float] = []

        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()
            x, _, y = unpack_batch(batch)
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            _, logits = model(x)
            ce = F.cross_entropy(logits, y)
            loss = ce

            opt.zero_grad()
            loss.backward()
            grad_norm, grad_cosine, prev_grad_vector = compute_grad_stats(
                model,
                prev_grad_vector,
                cfg.track_gradient_cosine,
            )
            opt.step()

            global_step += 1
            train_loss_values.append(float(loss.item()))
            train_ce_values.append(float(ce.item()))
            if grad_norm is not None:
                grad_norm_values.append(grad_norm)
            if grad_cosine is not None:
                grad_cosine_values.append(grad_cosine)

            if batch_idx % cfg.step_log_every_n_batches == 0:
                step_row = log_train_step(
                    loggers.train_step,
                    cfg,
                    spec.name,
                    epoch,
                    batch_idx,
                    global_step,
                    time.time() - step_start,
                    time.time() - run_start,
                    float(loss.item()),
                    float(ce.item()),
                    grad_norm,
                    grad_cosine,
                )
                step_history.append(step_row)

            if batch_idx % cfg.log_every_n_batches == 0:
                extra = f"grad={0.0 if grad_norm is None else grad_norm:.4f}"
                if grad_cosine is not None:
                    extra += f" cos={grad_cosine:.4f}"
                log_batch_progress(
                    spec.title,
                    epoch,
                    batch_idx,
                    len(train_loader),
                    float(loss.item()),
                    epoch_start,
                    extra=extra,
                )

        train_epoch_metrics = summarize_training_epoch(
            train_loss_values,
            train_ce_values,
            grad_norm_values,
            grad_cosine_values,
            extra_metric_lists={},
        )
        train_epoch_metrics["generalization_gap"] = None
        train_epoch_row = log_epoch_metrics(
            loggers.train_epoch,
            cfg,
            spec.name,
            epoch,
            global_step,
            time.time() - run_start,
            train_epoch_metrics,
        )
        train_epoch_history.append(train_epoch_row)

        last_val_metrics = evaluate_baseline(model, val_loader, cfg)
        val_metrics = dict(last_val_metrics)
        val_metrics["generalization_gap"] = (
            float(val_metrics["val_loss"]) - float(train_epoch_metrics["train_ce_mean"])
        )
        train_epoch_history[-1]["generalization_gap"] = val_metrics["generalization_gap"]
        val_row = log_epoch_metrics(
            loggers.val_epoch,
            cfg,
            spec.name,
            epoch,
            global_step,
            time.time() - run_start,
            val_metrics,
        )
        val_history.append(val_row)

        val_acc = float(last_val_metrics["val_accuracy"])
        best_acc, maybe_state = maybe_save_best(best_acc, val_acc, model, spec.name, cfg)
        if maybe_state is not None:
            best_state = maybe_state
            best_epoch = epoch + 1

        print(
            f"[{spec.title}] epoch {epoch + 1:02d} done | "
            f"train_loss {train_epoch_metrics['train_loss_mean']:.4f} | "
            f"val_loss {last_val_metrics['val_loss']:.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"best {best_acc:.4f} | "
            f"epoch_time {time.time() - epoch_start:.1f}s",
            flush=True,
        )

    model = load_best_state(model, best_state)
    summary = finalize_variant_summary(
        cfg,
        spec,
        best_acc,
        best_epoch,
        step_history,
        train_epoch_history,
        val_history,
        time.time() - run_start,
        last_val_metrics,
    )
    loggers.summary.write({"variant": spec.name, "seed": cfg.seed, "dataset": cfg.dataset, **summary})

    return model, VariantRunArtifacts(
        summary=summary,
        step_history=step_history,
        train_epoch_history=train_epoch_history,
        val_history=val_history,
    )


def train_phase_variant(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    initial_state: Dict[str, torch.Tensor],
    variant: str,
    loggers: VariantLoggerBundle,
) -> Tuple[PhaseModel, VariantRunArtifacts]:
    spec = VARIANT_SPECS[variant]
    print(f"\n=== {spec.title} ===", flush=True)

    model = PhaseModel(cfg.hidden_dim, cfg.embed_dim, num_classes).to(cfg.device)
    model.load_state_dict(initial_state)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None
    best_epoch = -1
    global_step = 0
    run_start = time.time()

    step_history: List[Dict[str, object]] = []
    train_epoch_history: List[Dict[str, object]] = []
    val_history: List[Dict[str, object]] = []

    prev_grad_vector: Optional[torch.Tensor] = None
    last_val_metrics: Dict[str, Optional[float]] = {
        "val_accuracy": 0.0,
        "val_loss": 0.0,
        "val_total_loss": 0.0,
        "val_coherence": None,
        "val_amp_loss": None,
        "val_phase_loss": None,
    }

    for epoch in range(cfg.epochs):
        maybe_set_epoch(train_loader, epoch)
        model.train()
        epoch_start = time.time()

        train_loss_values: List[float] = []
        train_ce_values: List[float] = []
        grad_norm_values: List[float] = []
        grad_cosine_values: List[float] = []
        gate_values: List[float] = []
        coherence_values: List[float] = []
        amp_values: List[float] = []
        phase_values: List[float] = []

        for batch_idx, batch in enumerate(train_loader):
            step_start = time.time()
            x, x_ref, y = unpack_batch(batch)
            x = x.to(cfg.device)
            x_ref = x_ref.to(cfg.device)
            y = y.to(cfg.device)

            _, _, amp, phase, _, logits = model(x)
            _, _, amp_ref, phase_ref, _, _ = model(x_ref)

            per_sample_ce = F.cross_entropy(logits, y, reduction="none")
            per_sample_amp = amplitude_alignment_loss(amp, amp_ref.detach())
            per_sample_phase = phase_alignment_loss(phase, phase_ref.detach())
            coherence = normalized_phase_coherence(amp, phase, amp_ref.detach(), phase_ref.detach())

            if spec.uses_gating and epoch >= cfg.gate_warmup_epochs:
                alpha = coherence_gate(coherence, beta=cfg.gate_beta, gamma=cfg.gate_gamma)
            else:
                alpha = torch.ones_like(coherence)

            per_sample_loss = alpha * per_sample_ce
            if spec.uses_alignment:
                per_sample_loss = (
                    per_sample_loss
                    + cfg.lambda_amp * per_sample_amp
                    + cfg.lambda_phase * per_sample_phase
                )

            loss = per_sample_loss.mean()
            ce = per_sample_ce.mean()

            opt.zero_grad()
            loss.backward()
            grad_norm, grad_cosine, prev_grad_vector = compute_grad_stats(
                model,
                prev_grad_vector,
                cfg.track_gradient_cosine,
            )
            opt.step()

            global_step += 1
            train_loss_values.append(float(loss.item()))
            train_ce_values.append(float(ce.item()))
            if grad_norm is not None:
                grad_norm_values.append(grad_norm)
            if grad_cosine is not None:
                grad_cosine_values.append(grad_cosine)
            gate_values.append(float(alpha.mean().item()))
            coherence_values.append(float(coherence.mean().item()))
            amp_values.append(float(per_sample_amp.mean().item()))
            phase_values.append(float(per_sample_phase.mean().item()))

            if batch_idx % cfg.step_log_every_n_batches == 0:
                step_row = log_train_step(
                    loggers.train_step,
                    cfg,
                    spec.name,
                    epoch,
                    batch_idx,
                    global_step,
                    time.time() - step_start,
                    time.time() - run_start,
                    float(loss.item()),
                    float(ce.item()),
                    grad_norm,
                    grad_cosine,
                    extra_metrics={
                        "coherence_mean": float(coherence.mean().item()),
                        "gate_mean": float(alpha.mean().item()),
                        "amp_loss": float(per_sample_amp.mean().item()),
                        "phase_loss": float(per_sample_phase.mean().item()),
                    },
                )
                step_history.append(step_row)

            if batch_idx % cfg.log_every_n_batches == 0:
                extra = (
                    f"gate={alpha.mean().item():.4f} "
                    f"coh={coherence.mean().item():.4f} "
                    f"amp={per_sample_amp.mean().item():.4f} "
                    f"phase={per_sample_phase.mean().item():.4f}"
                )
                if grad_norm is not None:
                    extra = f"grad={grad_norm:.4f} " + extra
                if grad_cosine is not None:
                    extra += f" cos={grad_cosine:.4f}"
                log_batch_progress(
                    spec.title,
                    epoch,
                    batch_idx,
                    len(train_loader),
                    float(loss.item()),
                    epoch_start,
                    extra=extra,
                )

        train_epoch_metrics = summarize_training_epoch(
            train_loss_values,
            train_ce_values,
            grad_norm_values,
            grad_cosine_values,
            extra_metric_lists={
                "gate": gate_values,
                "coherence": coherence_values,
                "amp_loss": amp_values,
                "phase_loss": phase_values,
            },
        )
        train_epoch_metrics["generalization_gap"] = None
        train_epoch_row = log_epoch_metrics(
            loggers.train_epoch,
            cfg,
            spec.name,
            epoch,
            global_step,
            time.time() - run_start,
            train_epoch_metrics,
        )
        train_epoch_history.append(train_epoch_row)

        last_val_metrics = evaluate_phase(
            model,
            val_loader,
            cfg,
            include_alignment_terms=spec.uses_alignment,
        )
        val_metrics = dict(last_val_metrics)
        val_metrics["generalization_gap"] = (
            float(val_metrics["val_loss"]) - float(train_epoch_metrics["train_ce_mean"])
        )
        train_epoch_history[-1]["generalization_gap"] = val_metrics["generalization_gap"]
        val_row = log_epoch_metrics(
            loggers.val_epoch,
            cfg,
            spec.name,
            epoch,
            global_step,
            time.time() - run_start,
            val_metrics,
        )
        val_history.append(val_row)

        val_acc = float(last_val_metrics["val_accuracy"])
        best_acc, maybe_state = maybe_save_best(best_acc, val_acc, model, spec.name, cfg)
        if maybe_state is not None:
            best_state = maybe_state
            best_epoch = epoch + 1

        print(
            f"[{spec.title}] epoch {epoch + 1:02d} done | "
            f"train_loss {train_epoch_metrics['train_loss_mean']:.4f} | "
            f"val_loss {last_val_metrics['val_loss']:.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"val_coh {0.0 if last_val_metrics['val_coherence'] is None else last_val_metrics['val_coherence']:.4f} | "
            f"best {best_acc:.4f} | "
            f"epoch_time {time.time() - epoch_start:.1f}s",
            flush=True,
        )

    model = load_best_state(model, best_state)
    summary = finalize_variant_summary(
        cfg,
        spec,
        best_acc,
        best_epoch,
        step_history,
        train_epoch_history,
        val_history,
        time.time() - run_start,
        last_val_metrics,
    )
    loggers.summary.write({"variant": spec.name, "seed": cfg.seed, "dataset": cfg.dataset, **summary})

    return model, VariantRunArtifacts(
        summary=summary,
        step_history=step_history,
        train_epoch_history=train_epoch_history,
        val_history=val_history,
    )


# ============================================================
# Plotting helpers
# ============================================================


def save_current_figure(path: Path) -> None:
    plt = get_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def get_pyplot():
    matplotlib_spec = importlib.util.find_spec("matplotlib")
    if matplotlib_spec is None:
        raise RuntimeError(
            "Plot generation requires matplotlib in the runtime environment. "
            "Install it on RunPod or rerun without --generate_plots."
        )

    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    return importlib.import_module("matplotlib.pyplot")


def plot_val_loss_vs_steps(
    run_dir: Path,
    cfg: Config,
    artifacts: Dict[str, VariantRunArtifacts],
) -> None:
    plt = get_pyplot()
    plt.figure(figsize=(7, 4))

    for variant in cfg.analysis_variants:
        history = artifacts[variant].val_history
        x = [int(row["global_step"]) for row in history]
        y = [float(row["val_loss"]) for row in history]
        plt.plot(x, y, marker="o", label=variant)

    plt.axhline(cfg.loss_target, color="gray", linestyle="--", linewidth=1.0, label="loss_target")
    plt.xlabel("Global step")
    plt.ylabel("Validation CE loss")
    plt.title("Validation Loss vs Steps")
    plt.legend()
    save_current_figure(run_dir / "val_loss_vs_steps.png")


def plot_steps_to_loss_target(
    run_dir: Path,
    cfg: Config,
    artifacts: Dict[str, VariantRunArtifacts],
) -> None:
    plt = get_pyplot()
    plt.figure(figsize=(6, 4))

    variants = list(cfg.analysis_variants)
    values = []
    labels = []
    for variant in variants:
        summary = artifacts[variant].summary
        labels.append(variant)
        step_value = summary["steps_to_loss_target"]
        values.append(float(step_value) if step_value is not None else math.nan)

    colors = ["#377eb8" if not math.isnan(value) else "#cccccc" for value in values]
    plt.bar(labels, values, color=colors)
    plt.ylabel("Steps to val-loss target")
    plt.title(f"Steps to Loss Target ({cfg.loss_target:.2f})")
    save_current_figure(run_dir / "steps_to_loss_target.png")


def plot_grad_norm_variance(
    run_dir: Path,
    cfg: Config,
    artifacts: Dict[str, VariantRunArtifacts],
) -> None:
    plt = get_pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(7, 7))

    bar_labels = []
    bar_values = []
    for variant in cfg.analysis_variants:
        bar_labels.append(variant)
        bar_values.append(float(artifacts[variant].summary["grad_norm_variance"] or 0.0))

    axes[0].bar(bar_labels, bar_values, color=["#377eb8", "#e41a1c"][: len(bar_labels)])
    axes[0].set_ylabel(f"Var(||g||), first {cfg.grad_variance_steps} steps")
    axes[0].set_title("Gradient Norm Variance")

    for variant in cfg.analysis_variants:
        step_history = artifacts[variant].step_history
        x = [int(row["global_step"]) for row in step_history if row.get("grad_norm") is not None]
        grad_norms = [float(row["grad_norm"]) for row in step_history if row.get("grad_norm") is not None]
        rolling = rolling_variance(grad_norms, cfg.rolling_window)
        axes[1].plot(x, [float(v) if v is not None else math.nan for v in rolling], label=variant)

    axes[1].set_xlabel("Global step")
    axes[1].set_ylabel(f"Rolling var(||g||), window={cfg.rolling_window}")
    axes[1].legend()
    save_current_figure(run_dir / "grad_norm_variance.png")


def plot_train_vs_val_loss(
    run_dir: Path,
    cfg: Config,
    artifacts: Dict[str, VariantRunArtifacts],
) -> None:
    plt = get_pyplot()
    plt.figure(figsize=(7, 4))

    for variant in cfg.analysis_variants:
        train_epochs = artifacts[variant].train_epoch_history
        val_epochs = artifacts[variant].val_history
        x = [int(row["global_step"]) for row in train_epochs]
        train_y = [float(row["train_ce_mean"]) for row in train_epochs]
        val_y = [float(row["val_loss"]) for row in val_epochs]
        plt.plot(x, train_y, linestyle="-", label=f"{variant} train")
        plt.plot(x, val_y, linestyle="--", label=f"{variant} val")

    plt.xlabel("Global step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    save_current_figure(run_dir / "train_vs_val_loss.png")


def plot_loss_vs_steps(
    run_dir: Path,
    cfg: Config,
    artifacts: Dict[str, VariantRunArtifacts],
) -> None:
    plt = get_pyplot()
    plt.figure(figsize=(8, 4.5))

    for variant in cfg.analysis_variants:
        step_history = artifacts[variant].step_history
        x = [int(row["global_step"]) for row in step_history]
        losses = [float(row["train_ce_loss"]) for row in step_history]
        smoothed = moving_average(losses, cfg.rolling_window)

        plt.plot(x, losses, alpha=0.2, label=f"{variant} raw")
        plt.plot(x, smoothed, linewidth=2, label=f"{variant} smooth")

    plt.xlabel("Global step")
    plt.ylabel("Train CE loss")
    plt.title("Loss vs Steps")
    plt.legend()
    save_current_figure(run_dir / "loss_vs_steps.png")


def generate_run_plots(
    run_dir: Path,
    cfg: Config,
    artifacts: Dict[str, VariantRunArtifacts],
) -> None:
    plot_val_loss_vs_steps(run_dir, cfg, artifacts)
    plot_steps_to_loss_target(run_dir, cfg, artifacts)
    plot_grad_norm_variance(run_dir, cfg, artifacts)
    plot_train_vs_val_loss(run_dir, cfg, artifacts)
    plot_loss_vs_steps(run_dir, cfg, artifacts)


# ============================================================
# Reporting helpers
# ============================================================


def build_run_summary(
    cfg: Config,
    variant_artifacts: Dict[str, VariantRunArtifacts],
    run_dir: Path,
) -> None:
    variants_payload = {
        variant: artifact.summary
        for variant, artifact in variant_artifacts.items()
    }

    reviewer_metrics: Dict[str, object] = {}
    if "baseline" in variants_payload and "full" in variants_payload:
        reviewer_metrics["full_minus_baseline"] = {
            "best_val_acc_delta": (
                float(variants_payload["full"]["best_val_acc"])
                - float(variants_payload["baseline"]["best_val_acc"])
            ),
            "final_val_loss_delta": (
                float(variants_payload["full"]["final_val_loss"])
                - float(variants_payload["baseline"]["final_val_loss"])
            ),
            "generalization_gap_delta": (
                float(variants_payload["full"]["final_generalization_gap"])
                - float(variants_payload["baseline"]["final_generalization_gap"])
            ),
            "grad_norm_variance_delta": (
                float(variants_payload["full"]["grad_norm_variance"] or 0.0)
                - float(variants_payload["baseline"]["grad_norm_variance"] or 0.0)
            ),
        }

    if "complex" in variants_payload and "full" in variants_payload:
        reviewer_metrics["full_minus_complex"] = {
            "best_val_acc_delta": (
                float(variants_payload["full"]["best_val_acc"])
                - float(variants_payload["complex"]["best_val_acc"])
            ),
            "final_val_loss_delta": (
                float(variants_payload["full"]["final_val_loss"])
                - float(variants_payload["complex"]["final_val_loss"])
            ),
            "generalization_gap_delta": (
                float(variants_payload["full"]["final_generalization_gap"])
                - float(variants_payload["complex"]["final_generalization_gap"])
            ),
        }

    write_json(
        run_dir / "run_summary.json",
        {
            "seed": cfg.seed,
            "dataset": cfg.dataset,
            "eval_protocol": cfg.eval_protocol,
            "variants": variants_payload,
            "reviewer_metrics": reviewer_metrics,
        },
    )


def build_experiment_summary(
    cfg: Config,
    all_results: List[Dict[str, VariantRunArtifacts]],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "dataset": cfg.dataset,
        "eval_protocol": cfg.eval_protocol,
        "num_runs": len(all_results),
        "variants": {},
        "paired_deltas": {},
    }

    metric_names = [
        "best_val_acc",
        "final_val_loss",
        "final_generalization_gap",
        "grad_norm_variance",
        "steps_to_loss_target",
    ]

    for variant in cfg.variants:
        variant_summary: Dict[str, object] = {}
        for metric_name in metric_names:
            values = [run[variant].summary.get(metric_name) for run in all_results if variant in run]
            variant_summary[metric_name] = compute_optional_summary_stats(values)
        summary["variants"][variant] = variant_summary

    paired_specs = [
        ("full", "baseline"),
        ("full", "complex"),
    ]
    paired_metric_names = [
        "best_val_acc",
        "final_val_loss",
        "final_generalization_gap",
        "grad_norm_variance",
        "steps_to_loss_target",
    ]

    for lhs, rhs in paired_specs:
        if lhs not in cfg.variants or rhs not in cfg.variants:
            continue

        delta_payload: Dict[str, object] = {}
        for metric_name in paired_metric_names:
            deltas: List[float] = []
            for run in all_results:
                lhs_value = run[lhs].summary.get(metric_name)
                rhs_value = run[rhs].summary.get(metric_name)
                if lhs_value is None or rhs_value is None:
                    continue
                deltas.append(float(lhs_value) - float(rhs_value))
            delta_payload[metric_name] = compute_optional_summary_stats(deltas)
        summary["paired_deltas"][f"{lhs}_minus_{rhs}"] = delta_payload

    return summary


# ============================================================
# Experiment runner
# ============================================================


def run_once(cfg: Config) -> Dict[str, VariantRunArtifacts]:
    set_seed(cfg.seed)
    train_loader, val_loader, num_classes = make_loaders(cfg)
    baseline_state, phase_state = capture_initial_states(cfg, num_classes)

    run_dir = build_run_dir(cfg)
    save_run_config(cfg, run_dir)

    artifacts: Dict[str, VariantRunArtifacts] = {}

    for variant in cfg.variants:
        loggers = create_variant_loggers(run_dir, variant)

        if variant == "baseline":
            _, artifact = train_baseline(
                cfg=cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=num_classes,
                initial_state=baseline_state,
                loggers=loggers,
            )
        else:
            _, artifact = train_phase_variant(
                cfg=cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=num_classes,
                initial_state=phase_state,
                variant=variant,
                loggers=loggers,
            )

        artifacts[variant] = artifact

    build_run_summary(cfg, artifacts, run_dir)

    if cfg.generate_plots:
        generate_run_plots(run_dir, cfg, artifacts)

    print("\nRESULTS", flush=True)
    for variant in cfg.variants:
        metrics = artifacts[variant].summary
        line = (
            f"{variant:<10} "
            f"best_acc={float(metrics['best_val_acc']):.6f} "
            f"val_loss={float(metrics['final_val_loss']):.6f} "
            f"gap={float(metrics['final_generalization_gap']):+.6f}"
        )
        if metrics["final_val_coherence"] is not None:
            line += f" | coh {float(metrics['final_val_coherence']):.6f}"
        print(line, flush=True)

    return artifacts


# ============================================================
# CLI / main
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase coherence experiments")

    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "medleydb_sample", "fma_small"])
    parser.add_argument("--medleydb_root", type=str, default="data/MedleyDB_sample")
    parser.add_argument("--fma_root", type=str, default="data/fma")
    parser.add_argument("--fma_metadata_root", type=str, default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--variants", type=parse_variants, default=ALL_VARIANTS)
    parser.add_argument("--analysis_variants", type=parse_variants, default=DEFAULT_ANALYSIS_VARIANTS)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--synthetic_reference_mode", type=str, default="paired_view", choices=["self", "paired_view"])
    parser.add_argument("--eval_protocol", type=str, default="same_track_fixed")
    parser.add_argument("--loss_target", type=float, default=1.0)
    parser.add_argument("--step_log_every_n_batches", type=int, default=1)
    parser.add_argument("--grad_variance_steps", type=int, default=200)
    parser.add_argument("--rolling_window", type=int, default=25)
    parser.add_argument("--generate_plots", action="store_true")
    parser.add_argument("--disable_gradient_cosine", action="store_true")

    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--segment_seconds", type=float, default=2.0)
    parser.add_argument("--medley_max_tracks", type=int, default=0)
    parser.add_argument("--fma_max_tracks", type=int, default=0)
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
        fma_root=args.fma_root,
        fma_metadata_root=args.fma_metadata_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_runs=args.num_runs,
        seed=args.seed,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        medley_max_tracks=args.medley_max_tracks,
        fma_max_tracks=args.fma_max_tracks,
        train_samples_per_epoch=args.train_samples_per_epoch,
        val_samples_per_epoch=args.val_samples_per_epoch,
        log_every_n_batches=args.log_every_n_batches,
        checkpoint_dir=args.checkpoint_dir,
        save_checkpoints=args.save_checkpoints,
        variants=args.variants,
        analysis_variants=args.analysis_variants,
        results_dir=args.results_dir,
        synthetic_reference_mode=args.synthetic_reference_mode,
        eval_protocol=args.eval_protocol,
        loss_target=args.loss_target,
        step_log_every_n_batches=args.step_log_every_n_batches,
        grad_variance_steps=args.grad_variance_steps,
        rolling_window=args.rolling_window,
        generate_plots=args.generate_plots,
        track_gradient_cosine=not args.disable_gradient_cosine,
    )

    validate_config(base_cfg)

    all_results: List[Dict[str, VariantRunArtifacts]] = []

    if base_cfg.save_checkpoints:
        ensure_checkpoint_dir(base_cfg)

    for run_idx in range(base_cfg.num_runs):
        run_cfg = replace(base_cfg, seed=base_cfg.seed + run_idx)

        print("\n" + "=" * 60, flush=True)
        print(f"RUN {run_cfg.seed} | device={run_cfg.device} | dataset={run_cfg.dataset}", flush=True)
        print("=" * 60, flush=True)

        all_results.append(run_once(run_cfg))

    summary = build_experiment_summary(base_cfg, all_results)
    summary_dir = ensure_dir(str(Path(base_cfg.results_dir) / base_cfg.dataset / base_cfg.eval_protocol))
    write_json(summary_dir / "summary.json", summary)

    print("\nSUMMARY", flush=True)
    for variant in base_cfg.variants:
        values = [float(run[variant].summary["best_val_acc"]) for run in all_results]
        print(format_summary_line(variant, values), flush=True)

    print("\nREVIEWER METRICS", flush=True)
    for variant in base_cfg.analysis_variants:
        metrics = summary["variants"][variant]
        loss_stats = metrics["final_val_loss"]
        gap_stats = metrics["final_generalization_gap"]
        grad_stats = metrics["grad_norm_variance"]
        print(
            f"{variant:<10} "
            f"val_loss={loss_stats['mean']:.6f if loss_stats['mean'] is not None else 'n/a'}",
            flush=True,
        )
        print(
            f"{variant:<10} "
            f"gap={gap_stats['mean']:.6f if gap_stats['mean'] is not None else 'n/a'} "
            f"grad_var={grad_stats['mean']:.6f if grad_stats['mean'] is not None else 'n/a'}",
            flush=True,
        )

    if "full_minus_baseline" in summary["paired_deltas"]:
        print("\nFULL VS BASELINE", flush=True)
        for metric_name, stats in summary["paired_deltas"]["full_minus_baseline"].items():
            if stats["mean"] is None:
                print(f"{metric_name:<24} n/a", flush=True)
            elif stats["std"] is None:
                print(f"{metric_name:<24} {stats['mean']:+.6f} (n=1)", flush=True)
            else:
                print(f"{metric_name:<24} {stats['mean']:+.6f} ± {stats['std']:.6f}", flush=True)

    if "full_minus_complex" in summary["paired_deltas"]:
        print("\nFULL VS COMPLEX", flush=True)
        for metric_name, stats in summary["paired_deltas"]["full_minus_complex"].items():
            if stats["mean"] is None:
                print(f"{metric_name:<24} n/a", flush=True)
            elif stats["std"] is None:
                print(f"{metric_name:<24} {stats['mean']:+.6f} (n=1)", flush=True)
            else:
                print(f"{metric_name:<24} {stats['mean']:+.6f} ± {stats['std']:.6f}", flush=True)


if __name__ == "__main__":
    main()
