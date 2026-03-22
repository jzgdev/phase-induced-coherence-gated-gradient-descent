import math
import random
from dataclasses import dataclass

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
    samples_per_class: int = 2000
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    hidden_dim: int = 128
    embed_dim: int = 64
    beta: float = 4.0              # temperature for coherence gate
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


cfg = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg.seed)


# ============================================================
# Synthetic Dataset
# ============================================================

class PhaseStructuredDataset(Dataset):
    """
    Synthetic dataset where each sample is a 1D signal made from two sinusoids.
    Classes differ in frequency pair + relative phase offset pattern.

    This is useful because:
    - magnitude/energy can be similar across classes
    - phase relationships actually matter
    """

    def __init__(self, train: bool = True, cfg: Config = cfg):
        super().__init__()
        self.cfg = cfg
        self.samples = []
        self.labels = []

        total = cfg.samples_per_class
        if not train:
            total = max(400, total // 5)

        t = torch.linspace(0, 1, cfg.seq_len)

        # Each class uses a different frequency pair and phase relation.
        class_specs = [
            {"f1": 5.0, "f2": 11.0, "phase_offset": 0.0},
            {"f1": 5.0, "f2": 11.0, "phase_offset": math.pi / 2},
            {"f1": 7.0, "f2": 13.0, "phase_offset": math.pi / 4},
            {"f1": 7.0, "f2": 13.0, "phase_offset": 3 * math.pi / 4},
        ]

        for label, spec in enumerate(class_specs):
            for _ in range(total):
                # Random global phase
                base_phase = random.uniform(0, 2 * math.pi)

                # Slight amplitude jitter
                a1 = random.uniform(0.8, 1.2)
                a2 = random.uniform(0.8, 1.2)

                # Small frequency jitter
                f1 = spec["f1"] + random.uniform(-0.2, 0.2)
                f2 = spec["f2"] + random.uniform(-0.2, 0.2)

                # Build signal
                x = (
                    a1 * torch.sin(2 * math.pi * f1 * t + base_phase)
                    + a2 * torch.sin(2 * math.pi * f2 * t + base_phase + spec["phase_offset"])
                )

                # Add noise
                x = x + 0.1 * torch.randn_like(x)

                # Normalize per sample
                x = (x - x.mean()) / (x.std() + 1e-6)

                self.samples.append(x.unsqueeze(0))  # [1, T]
                self.labels.append(label)

        self.samples = torch.stack(self.samples)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.samples[idx]
        y = self.labels[idx]
        return x, y


# ============================================================
# Augmentations
# ============================================================

def augment_signal(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, 1, T]
    Mild augmentations so the reference view is related but not identical.
    """
    # Small gaussian noise
    x = x + 0.03 * torch.randn_like(x)

    # Small gain jitter
    gain = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(0.9, 1.1)
    x = x * gain

    # Small random circular time shift
    shifts = torch.randint(low=-4, high=5, size=(x.size(0),), device=x.device)
    out = []
    for i in range(x.size(0)):
        out.append(torch.roll(x[i], shifts=int(shifts[i].item()), dims=-1))
    x = torch.stack(out, dim=0)

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

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        z = self.proj(h)
        z = F.normalize(z, dim=-1)
        logits = self.cls(z)
        return z, logits


# ============================================================
# Experimental Model
# ============================================================

class PhaseCoherenceModel(nn.Module):
    """
    Projection head outputs real + imaginary latent parts.
    We derive:
    - amplitude
    - phase
    - normalized embedding for classification
    """
    def __init__(self, hidden_dim: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.encoder = ConvEncoder(hidden_dim)
        self.proj_real = nn.Linear(hidden_dim, embed_dim)
        self.proj_imag = nn.Linear(hidden_dim, embed_dim)
        self.cls = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x: torch.Tensor):
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
# Coherence Functions
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
    coh = (amp_a * amp_b * torch.cos(phase_a - phase_b)).mean(dim=-1)
    return coh


def phase_alignment_loss(phase_a: torch.Tensor, phase_b: torch.Tensor) -> torch.Tensor:
    """
    L_phase = 1 - cos(angle(h) - angle(r))
    """
    return (1.0 - torch.cos(phase_a - phase_b)).mean()


def amplitude_alignment_loss(amp_a: torch.Tensor, amp_b: torch.Tensor) -> torch.Tensor:
    """
    L_amp = || |h| - |r| ||^2
    """
    return F.mse_loss(amp_a, amp_b)


# ============================================================
# Training / Eval
# ============================================================

@torch.no_grad()
def evaluate_baseline(model: BaselineModel, loader: DataLoader, device: str):
    model.eval()
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        _, logits = model(x)
        preds = logits.argmax(dim=-1)

        total += y.size(0)
        correct += (preds == y).sum().item()

    return correct / total


@torch.no_grad()
def evaluate_experimental(model: PhaseCoherenceModel, loader: DataLoader, device: str):
    model.eval()
    total = 0
    correct = 0
    coh_vals = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        x_ref = augment_signal(x)

        out = model(x)
        ref = model(x_ref)

        preds = out["logits"].argmax(dim=-1)
        total += y.size(0)
        correct += (preds == y).sum().item()

        coh = coherence_score(out["amplitude"], out["phase"], ref["amplitude"], ref["phase"])
        coh_vals.append(coh.mean().item())

    return correct / total, sum(coh_vals) / max(len(coh_vals), 1)


def train_baseline(cfg: Config, train_loader: DataLoader, val_loader: DataLoader):
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

        val_acc = evaluate_baseline(model, val_loader, cfg.device)
        print(
            f"[Baseline] Epoch {epoch+1:02d}/{cfg.epochs} | "
            f"train_loss={running_loss / len(train_loader):.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    return model


def train_experimental(cfg: Config, train_loader: DataLoader, val_loader: DataLoader):
    print("\n=== Training experimental model ===")
    model = PhaseCoherenceModel(cfg.hidden_dim, cfg.embed_dim, cfg.num_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    lambda_amp = 0.2
    lambda_phase = 0.2

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_coh = 0.0

        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            # Create reference view
            x_ref = augment_signal(x)

            out = model(x)
            ref = model(x_ref)

            # Main task loss
            ce = F.cross_entropy(out["logits"], y)

            # Auxiliary alignment losses
            l_amp = amplitude_alignment_loss(out["amplitude"], ref["amplitude"])
            l_phase = phase_alignment_loss(out["phase"], ref["phase"])

            # Coherence gate
            coh = coherence_score(
                out["amplitude"], out["phase"],
                ref["amplitude"], ref["phase"]
            )

            # alpha(x) = sigmoid(beta * C(x))
            alpha = torch.sigmoid(cfg.beta * coh).mean()

            # Total loss
            base_loss = ce + lambda_amp * l_amp + lambda_phase * l_phase

            # IMPORTANT:
            # Instead of doing per-sample gradient surgery right now,
            # we scale the batch loss by the mean coherence gate.
            # This is a simple first implementation of the idea.
            loss = alpha * base_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            running_coh += coh.mean().item()

        val_acc, val_coh = evaluate_experimental(model, val_loader, cfg.device)
        print(
            f"[Experimental] Epoch {epoch+1:02d}/{cfg.epochs} | "
            f"train_loss={running_loss / len(train_loader):.4f} | "
            f"train_coh={running_coh / len(train_loader):.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_coh={val_coh:.4f}"
        )

    return model


# ============================================================
# Main
# ============================================================

def main():
    print(f"Using device: {cfg.device}")

    train_ds = PhaseStructuredDataset(train=True, cfg=cfg)
    val_ds = PhaseStructuredDataset(train=False, cfg=cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    baseline = train_baseline(cfg, train_loader, val_loader)
    experimental = train_experimental(cfg, train_loader, val_loader)

    baseline_acc = evaluate_baseline(baseline, val_loader, cfg.device)
    experimental_acc, experimental_coh = evaluate_experimental(experimental, val_loader, cfg.device)

    print("\n=== Final Results ===")
    print(f"Baseline val acc:      {baseline_acc:.4f}")
    print(f"Experimental val acc:  {experimental_acc:.4f}")
    print(f"Experimental val coh:  {experimental_coh:.4f}")


if __name__ == "__main__":
    main()