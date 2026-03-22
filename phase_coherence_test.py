import math
import random
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    seq_len: int = 256
    num_classes: int = 4
    samples_per_class_train: int = 1500
    samples_per_class_val: int = 400
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    hidden_dim: int = 128
    embed_dim: int = 64
    beta: float = 4.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_runs: int = 3

    # Synthetic task difficulty
    base_f1: float = 6.0
    base_f2: float = 12.0
    freq_jitter: float = 0.5
    amp_low: float = 0.7
    amp_high: float = 1.3
    phase_jitter: float = 0.35
    noise_std: float = 0.2

    # Augmentation settings
    aug_noise_std: float = 0.05
    aug_gain_low: float = 0.9
    aug_gain_high: float = 1.1
    aug_shift_min: int = -6
    aug_shift_max: int = 6

    # Experimental loss weights
    lambda_amp: float = 0.1
    lambda_phase: float = 0.1


cfg = Config()


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
# Synthetic Dataset
# ============================================================

class PhaseStructuredDataset(Dataset):
    """
    Harder synthetic task:
    - all classes use the SAME base frequencies
    - classes differ mainly by relative phase offset
    - added frequency jitter, amplitude jitter, phase jitter, and noise

    This forces the model to care more about phase structure.
    """

    def __init__(self, train: bool, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.samples: List[torch.Tensor] = []
        self.labels: List[int] = []

        total = cfg.samples_per_class_train if train else cfg.samples_per_class_val
        t = torch.linspace(0.0, 1.0, cfg.seq_len)

        # Same frequencies for all classes. Only phase offset differs.
        phase_offsets = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]

        for label, class_phase_offset in enumerate(phase_offsets):
            for _ in range(total):
                base_phase = random.uniform(0.0, 2.0 * math.pi)

                a1 = random.uniform(cfg.amp_low, cfg.amp_high)
                a2 = random.uniform(cfg.amp_low, cfg.amp_high)

                f1 = cfg.base_f1 + random.uniform(-cfg.freq_jitter, cfg.freq_jitter)
                f2 = cfg.base_f2 + random.uniform(-cfg.freq_jitter, cfg.freq_jitter)

                local_phase_jitter = random.uniform(-cfg.phase_jitter, cfg.phase_jitter)

                x = (
                    a1 * torch.sin(2.0 * math.pi * f1 * t + base_phase)
                    + a2 * torch.sin(
                        2.0 * math.pi * f2 * t + base_phase + class_phase_offset + local_phase_jitter
                    )
                )

                x = x + cfg.noise_std * torch.randn_like(x)
                x = (x - x.mean()) / (x.std() + 1e-6)

                self.samples.append(x.unsqueeze(0))  # [1, T]
                self.labels.append(label)

        self.samples = torch.stack(self.samples)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.labels[idx]


# ============================================================
# Augmentations
# ============================================================

def augment_signal(x: torch.Tensor, cfg: Config) -> torch.Tensor:
    """
    Mild augmentations for the reference view.
    x: [B, 1, T]
    """
    x = x + cfg.aug_noise_std * torch.randn_like(x)

    gain = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(
        cfg.aug_gain_low, cfg.aug_gain_high
    )
    x = x * gain

    shifts = torch.randint(
        low=cfg.aug_shift_min,
        high=cfg.aug_shift_max + 1,
        size=(x.size(0),),
        device=x.device,
    )

    x = torch.stack(
        [torch.roll(x[i], shifts=int(shifts[i].item()), dims=-1) for i in range(x.size(0))],
        dim=0,
    )

    return x


# ============================================================
# Encoder Backbone
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
        h = self.pool(h).squeeze(-1)
        return h


# ============================================================
# Baseline Model
# ============================================================

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


# ============================================================
# Experimental Model
# ============================================================

class PhaseCoherenceModel(nn.Module):
    """
    Projection head outputs real + imaginary latent parts.
    We derive amplitude and phase from them.
    """

    def __init__(self, hidden_dim: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.encoder = ConvEncoder(hidden_dim)
        self.proj_real = nn.Linear(hidden_dim, embed_dim)
        self.proj_imag = nn.Linear(hidden_dim, embed_dim)
        self.cls = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)

        z_real = self.proj_real(h)
        z_imag = self.proj_imag(h)

        amplitude = torch.sqrt(z_real.pow(2) + z_imag.pow(2) + 1e-8)
        phase = torch.atan2(z_imag, z_real)

        z_cat = torch.cat([z_real, z_imag], dim=-1)
        z_norm = F.normalize(z_cat, dim=-1)
        logits = self.cls(z_norm)

        return {
            "z_real": z_real,
            "z_imag": z_imag,
            "amplitude": amplitude,
            "phase": phase,
            "embedding": z_norm,
            "logits": logits,
        }


# ============================================================
# Coherence / Auxiliary Losses
# ============================================================

def coherence_score(
    amp_a: torch.Tensor,
    phase_a: torch.Tensor,
    amp_b: torch.Tensor,
    phase_b: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample coherence:
    C(x) = (1/n) * sum_i |h_i||r_i| cos(phi_i - phi_r_i)
    """
    return (amp_a * amp_b * torch.cos(phase_a - phase_b)).mean(dim=-1)


def phase_alignment_loss(
    phase_a: torch.Tensor,
    phase_b: torch.Tensor,
) -> torch.Tensor:
    return (1.0 - torch.cos(phase_a - phase_b)).mean(dim=-1)


def amplitude_alignment_loss(
    amp_a: torch.Tensor,
    amp_b: torch.Tensor,
) -> torch.Tensor:
    return ((amp_a - amp_b) ** 2).mean(dim=-1)


# ============================================================
# Evaluation Helpers
# ============================================================

@torch.no_grad()
def evaluate_baseline(
    model: BaselineModel,
    loader: DataLoader,
    cfg: Config,
) -> float:
    model.eval()
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)

        _, logits = model(x)
        preds = logits.argmax(dim=-1)

        total += y.size(0)
        correct += (preds == y).sum().item()

    return correct / total


@torch.no_grad()
def evaluate_experimental(
    model: PhaseCoherenceModel,
    loader: DataLoader,
    cfg: Config,
) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    coh_vals: List[float] = []

    for x, y in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)

        x_ref = augment_signal(x, cfg)

        out = model(x)
        ref = model(x_ref)

        preds = out["logits"].argmax(dim=-1)
        total += y.size(0)
        correct += (preds == y).sum().item()

        coh = coherence_score(
            out["amplitude"], out["phase"],
            ref["amplitude"], ref["phase"]
        )
        coh_vals.append(coh.mean().item())

    mean_coh = sum(coh_vals) / max(len(coh_vals), 1)
    return correct / total, mean_coh


# ============================================================
# Training
# ============================================================

def train_baseline(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> BaselineModel:
    print("\n=== Training baseline ===")
    model = BaselineModel(cfg.hidden_dim, cfg.embed_dim, cfg.num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            _, logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()

        val_acc = evaluate_baseline(model, val_loader, cfg)
        print(
            f"[Baseline] Epoch {epoch + 1:02d}/{cfg.epochs} | "
            f"train_loss={running_loss / len(train_loader):.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    return model


def train_experimental(
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> PhaseCoherenceModel:
    print("\n=== Training experimental model ===")
    model = PhaseCoherenceModel(cfg.hidden_dim, cfg.embed_dim, cfg.num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_coh = 0.0

        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            x_ref = augment_signal(x, cfg)

            out = model(x)
            ref = model(x_ref)

            # Per-sample CE, so gating is actually per-sample.
            per_sample_ce = F.cross_entropy(out["logits"], y, reduction="none")

            coh = coherence_score(
                out["amplitude"], out["phase"],
                ref["amplitude"].detach(), ref["phase"].detach()
            )

            alpha = torch.sigmoid(cfg.beta * coh)

            per_sample_amp = amplitude_alignment_loss(
                out["amplitude"], ref["amplitude"].detach()
            )
            per_sample_phase = phase_alignment_loss(
                out["phase"], ref["phase"].detach()
            )

            per_sample_loss = (
                alpha * per_sample_ce
                + cfg.lambda_amp * per_sample_amp
                + cfg.lambda_phase * per_sample_phase
            )

            loss = per_sample_loss.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            running_coh += coh.mean().item()

        val_acc, val_coh = evaluate_experimental(model, val_loader, cfg)
        print(
            f"[Experimental] Epoch {epoch + 1:02d}/{cfg.epochs} | "
            f"train_loss={running_loss / len(train_loader):.4f} | "
            f"train_coh={running_coh / len(train_loader):.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_coh={val_coh:.4f}"
        )

    return model


# ============================================================
# One Full Run
# ============================================================

def run_once(run_cfg: Config) -> Dict[str, float]:
    set_seed(run_cfg.seed)

    print(f"\n{'=' * 72}")
    print(f"RUN seed={run_cfg.seed} | device={run_cfg.device}")
    print(f"{'=' * 72}")

    train_ds = PhaseStructuredDataset(train=True, cfg=run_cfg)
    val_ds = PhaseStructuredDataset(train=False, cfg=run_cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=run_cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=run_cfg.batch_size,
        shuffle=False,
    )

    baseline = train_baseline(run_cfg, train_loader, val_loader)
    experimental = train_experimental(run_cfg, train_loader, val_loader)

    baseline_acc = evaluate_baseline(baseline, val_loader, run_cfg)
    experimental_acc, experimental_coh = evaluate_experimental(experimental, val_loader, run_cfg)

    print("\n=== Final Results (single run) ===")
    print(f"Baseline val acc:      {baseline_acc:.4f}")
    print(f"Experimental val acc:  {experimental_acc:.4f}")
    print(f"Experimental val coh:  {experimental_coh:.4f}")

    return {
        "baseline_acc": baseline_acc,
        "experimental_acc": experimental_acc,
        "experimental_coh": experimental_coh,
    }


# ============================================================
# Multi-Run Summary
# ============================================================

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main() -> None:
    all_results: List[Dict[str, float]] = []

    for run_idx in range(cfg.num_runs):
        run_cfg = replace(cfg, seed=cfg.seed + run_idx)
        results = run_once(run_cfg)
        all_results.append(results)

    baseline_accs = [r["baseline_acc"] for r in all_results]
    experimental_accs = [r["experimental_acc"] for r in all_results]
    experimental_cohs = [r["experimental_coh"] for r in all_results]

    print(f"\n{'=' * 72}")
    print("MULTI-RUN SUMMARY")
    print(f"{'=' * 72}")
    print(
        f"Baseline acc:      {mean(baseline_accs):.4f} ± {std(baseline_accs):.4f}"
    )
    print(
        f"Experimental acc:  {mean(experimental_accs):.4f} ± {std(experimental_accs):.4f}"
    )
    print(
        f"Experimental coh:  {mean(experimental_cohs):.4f} ± {std(experimental_cohs):.4f}"
    )
    print(
        f"Delta acc:         {mean(experimental_accs) - mean(baseline_accs):+.4f}"
    )


if __name__ == "__main__":
    main()