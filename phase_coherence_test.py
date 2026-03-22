import copy
import math
import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_runs: int = 3

    # Synthetic data difficulty
    base_f1: float = 6.0
    base_f2: float = 12.0
    freq_jitter: float = 0.5
    amp_low: float = 0.7
    amp_high: float = 1.3
    phase_jitter: float = 0.35
    noise_std: float = 0.2

    # Augmentations
    aug_noise_std: float = 0.05
    aug_gain_low: float = 0.9
    aug_gain_high: float = 1.1
    aug_shift_min: int = -6
    aug_shift_max: int = 6

    # Alignment losses
    lambda_amp: float = 0.10
    lambda_phase: float = 0.10

    # Coherence gating
    gate_beta: float = 2.0
    gate_gamma: float = 0.20
    gate_warmup_epochs: int = 5

    # Optional dataloader workers
    num_workers: int = 0


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
    Hard synthetic task:
    - same base frequencies for all classes
    - class identity differs mainly by relative phase offset
    - frequency jitter, amplitude jitter, phase jitter, and noise
    """

    def __init__(self, train: bool, cfg: Config):
        super().__init__()

        total = cfg.samples_per_class_train if train else cfg.samples_per_class_val
        t = torch.linspace(0.0, 1.0, cfg.seq_len)

        phase_offsets = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]

        samples: List[torch.Tensor] = []
        labels: List[int] = []

        for label, offset in enumerate(phase_offsets):
            for _ in range(total):
                base_phase = random.uniform(0.0, 2.0 * math.pi)

                a1 = random.uniform(cfg.amp_low, cfg.amp_high)
                a2 = random.uniform(cfg.amp_low, cfg.amp_high)

                f1 = cfg.base_f1 + random.uniform(-cfg.freq_jitter, cfg.freq_jitter)
                f2 = cfg.base_f2 + random.uniform(-cfg.freq_jitter, cfg.freq_jitter)

                jitter = random.uniform(-cfg.phase_jitter, cfg.phase_jitter)

                x = (
                    a1 * torch.sin(2.0 * math.pi * f1 * t + base_phase)
                    + a2 * torch.sin(2.0 * math.pi * f2 * t + base_phase + offset + jitter)
                )

                x = x + cfg.noise_std * torch.randn_like(x)
                x = (x - x.mean()) / (x.std() + 1e-6)

                samples.append(x.unsqueeze(0))
                labels.append(label)

        self.samples = torch.stack(samples)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx], self.labels[idx]


# ============================================================
# Augmentations
# ============================================================

def augment_signal(x: torch.Tensor, cfg: Config) -> torch.Tensor:
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
    """
    Complex-style latent model:
    - real / imag projections
    - amplitude / phase derived from them
    - classifier on concatenated latent
    """

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
# Coherence / Auxiliary losses
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
    """
    Phase-aware coherence with amplitude normalization.
    Returns one scalar per sample.
    """
    amp_a_n = amp_a / (amp_a.mean(dim=-1, keepdim=True) + 1e-6)
    amp_b_n = amp_b / (amp_b.mean(dim=-1, keepdim=True) + 1e-6)

    coh = (amp_a_n * amp_b_n * torch.cos(phase_a - phase_b)).mean(dim=-1)
    return coh


def coherence_gate(
    coh: torch.Tensor,
    beta: float,
    gamma: float,
) -> torch.Tensor:
    """
    Gate centered at 1.0 instead of ~0.5.
    This avoids cutting gradients in half all the time.
    """
    coh_centered = (coh - coh.mean().detach()) / (coh.std().detach() + 1e-6)
    alpha = 1.0 + gamma * torch.tanh(beta * coh_centered)
    return alpha


# ============================================================
# Evaluation
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
def evaluate_phase(
    model: PhaseModel,
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

        _, _, amp, phase, _, logits = model(x)
        _, _, amp_ref, phase_ref, _, _ = model(x_ref)

        preds = logits.argmax(dim=-1)

        total += y.size(0)
        correct += (preds == y).sum().item()

        coh = normalized_phase_coherence(amp, phase, amp_ref, phase_ref)
        coh_vals.append(coh.mean().item())

    return correct / total, sum(coh_vals) / len(coh_vals)


# ============================================================
# Training helpers
# ============================================================

def maybe_save_best(
    best_acc: float,
    current_acc: float,
    model: nn.Module,
) -> Tuple[float, Optional[Dict[str, torch.Tensor]]]:
    if current_acc > best_acc:
        return current_acc, copy.deepcopy(model.state_dict())
    return best_acc, None


def load_best_state(model: nn.Module, best_state: Optional[Dict[str, torch.Tensor]]) -> nn.Module:
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================
# Training variants
# ============================================================

def train_baseline(cfg: Config, train_loader: DataLoader, val_loader: DataLoader) -> BaselineModel:
    print("\n=== Baseline ===")

    model = BaselineModel(cfg.hidden_dim, cfg.embed_dim, cfg.num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            _, logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()

        val_acc = evaluate_baseline(model, val_loader, cfg)
        best_acc, maybe_state = maybe_save_best(best_acc, val_acc, model)
        if maybe_state is not None:
            best_state = maybe_state

        print(
            f"[Baseline] epoch {epoch + 1:02d} | "
            f"loss {loss_sum / len(train_loader):.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"best {best_acc:.4f}"
        )

    return load_best_state(model, best_state)


def train_complex_only(cfg: Config, train_loader: DataLoader, val_loader: DataLoader) -> PhaseModel:
    print("\n=== Complex Latent Only ===")

    model = PhaseModel(cfg.hidden_dim, cfg.embed_dim, cfg.num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            _, _, _, _, _, logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()

        val_acc, _ = evaluate_phase(model, val_loader, cfg)
        best_acc, maybe_state = maybe_save_best(best_acc, val_acc, model)
        if maybe_state is not None:
            best_state = maybe_state

        print(
            f"[Complex] epoch {epoch + 1:02d} | "
            f"loss {loss_sum / len(train_loader):.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"best {best_acc:.4f}"
        )

    return load_best_state(model, best_state)


def train_alignment_only(cfg: Config, train_loader: DataLoader, val_loader: DataLoader) -> PhaseModel:
    print("\n=== Alignment Loss Only ===")

    model = PhaseModel(cfg.hidden_dim, cfg.embed_dim, cfg.num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            x_ref = augment_signal(x, cfg)

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

        val_acc, _ = evaluate_phase(model, val_loader, cfg)
        best_acc, maybe_state = maybe_save_best(best_acc, val_acc, model)
        if maybe_state is not None:
            best_state = maybe_state

        print(
            f"[Align] epoch {epoch + 1:02d} | "
            f"loss {loss_sum / len(train_loader):.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"best {best_acc:.4f}"
        )

    return load_best_state(model, best_state)


def train_full(cfg: Config, train_loader: DataLoader, val_loader: DataLoader) -> PhaseModel:
    print("\n=== Full Method (Alignment + Gentle Coherence Gating) ===")

    model = PhaseModel(cfg.hidden_dim, cfg.embed_dim, cfg.num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = -1.0
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        loss_sum = 0.0
        gate_sum = 0.0
        coh_sum = 0.0

        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            x_ref = augment_signal(x, cfg)

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

        val_acc, val_coh = evaluate_phase(model, val_loader, cfg)
        best_acc, maybe_state = maybe_save_best(best_acc, val_acc, model)
        if maybe_state is not None:
            best_state = maybe_state

        print(
            f"[Full] epoch {epoch + 1:02d} | "
            f"loss {loss_sum / len(train_loader):.4f} | "
            f"train_gate {gate_sum / len(train_loader):.4f} | "
            f"train_coh {coh_sum / len(train_loader):.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"val_coh {val_coh:.4f} | "
            f"best {best_acc:.4f}"
        )

    return load_best_state(model, best_state)


# ============================================================
# Experiment runner
# ============================================================

def run_once(cfg: Config) -> Tuple[float, float, float, float]:
    set_seed(cfg.seed)

    train_ds = PhaseStructuredDataset(train=True, cfg=cfg)
    val_ds = PhaseStructuredDataset(train=False, cfg=cfg)

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

    baseline = train_baseline(cfg, train_loader, val_loader)
    complex_only = train_complex_only(cfg, train_loader, val_loader)
    alignment_only = train_alignment_only(cfg, train_loader, val_loader)
    full = train_full(cfg, train_loader, val_loader)

    baseline_acc = evaluate_baseline(baseline, val_loader, cfg)
    complex_acc, _ = evaluate_phase(complex_only, val_loader, cfg)
    align_acc, _ = evaluate_phase(alignment_only, val_loader, cfg)
    full_acc, full_coh = evaluate_phase(full, val_loader, cfg)

    print("\nRESULTS")
    print(f"Baseline: {baseline_acc:.6f}")
    print(f"Complex:  {complex_acc:.6f}")
    print(f"Align:    {align_acc:.6f}")
    print(f"Full:     {full_acc:.6f}")
    print(f"Full coh: {full_coh:.6f}")

    return baseline_acc, complex_acc, align_acc, full_acc


# ============================================================
# Main
# ============================================================

def main() -> None:
    results = []

    for run_idx in range(cfg.num_runs):
        run_cfg = replace(cfg, seed=cfg.seed + run_idx)

        print("\n" + "=" * 60)
        print(f"RUN {run_cfg.seed} | device={run_cfg.device}")
        print("=" * 60)

        results.append(run_once(run_cfg))

    r = torch.tensor(results, dtype=torch.float32)

    names = ["Baseline", "Complex", "Align", "Full"]

    print("\nSUMMARY")
    for i, name in enumerate(names):
        vals = r[:, i]
        print(f"{name:<10} {vals.mean().item():.6f} ± {vals.std(unbiased=True).item():.6f}")

    print(f"\nDelta(Complex-Baseline): {(r[:,1].mean() - r[:,0].mean()).item():+.6f}")
    print(f"Delta(Align-Baseline):   {(r[:,2].mean() - r[:,0].mean()).item():+.6f}")
    print(f"Delta(Full-Baseline):    {(r[:,3].mean() - r[:,0].mean()).item():+.6f}")
    print(f"Delta(Full-Align):       {(r[:,3].mean() - r[:,2].mean()).item():+.6f}")


if __name__ == "__main__":
    main()