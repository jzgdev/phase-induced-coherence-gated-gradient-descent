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
    beta: float = 4.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


cfg = Config()


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(cfg.seed)


# ============================================================
# Synthetic Dataset
# ============================================================

class PhaseStructuredDataset(Dataset):

    def __init__(self, train=True, cfg=cfg):

        self.samples = []
        self.labels = []

        total = cfg.samples_per_class if train else max(400, cfg.samples_per_class // 5)

        t = torch.linspace(0, 1, cfg.seq_len)

        class_specs = [
            {"f1": 5.0, "f2": 11.0, "phase_offset": 0.0},
            {"f1": 5.0, "f2": 11.0, "phase_offset": math.pi / 2},
            {"f1": 7.0, "f2": 13.0, "phase_offset": math.pi / 4},
            {"f1": 7.0, "f2": 13.0, "phase_offset": 3 * math.pi / 4},
        ]

        for label, spec in enumerate(class_specs):

            for _ in range(total):

                base_phase = random.uniform(0, 2 * math.pi)

                a1 = random.uniform(0.8, 1.2)
                a2 = random.uniform(0.8, 1.2)

                f1 = spec["f1"] + random.uniform(-0.2, 0.2)
                f2 = spec["f2"] + random.uniform(-0.2, 0.2)

                x = (
                    a1 * torch.sin(2 * math.pi * f1 * t + base_phase)
                    + a2 * torch.sin(
                        2 * math.pi * f2 * t + base_phase + spec["phase_offset"]
                    )
                )

                x += 0.1 * torch.randn_like(x)

                x = (x - x.mean()) / (x.std() + 1e-6)

                self.samples.append(x.unsqueeze(0))
                self.labels.append(label)

        self.samples = torch.stack(self.samples)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# ============================================================
# Augmentations
# ============================================================

def augment_signal(x):

    x = x + 0.03 * torch.randn_like(x)

    gain = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(0.9, 1.1)
    x = x * gain

    shifts = torch.randint(-4, 5, (x.size(0),), device=x.device)
    x = torch.stack([torch.roll(x[i], shifts=int(shifts[i]), dims=-1) for i in range(x.size(0))])

    return x


# ============================================================
# Encoder
# ============================================================

class ConvEncoder(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32, 64, 7, padding=3, stride=2),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, hidden_dim, 5, padding=2, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        h = self.net(x)
        return self.pool(h).squeeze(-1)


# ============================================================
# Baseline
# ============================================================

class BaselineModel(nn.Module):

    def __init__(self, hidden_dim, embed_dim, num_classes):

        super().__init__()

        self.encoder = ConvEncoder(hidden_dim)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.cls = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        h = self.encoder(x)

        z = F.normalize(self.proj(h), dim=-1)

        logits = self.cls(z)

        return z, logits


# ============================================================
# Experimental Model
# ============================================================

class PhaseCoherenceModel(nn.Module):

    def __init__(self, hidden_dim, embed_dim, num_classes):

        super().__init__()

        self.encoder = ConvEncoder(hidden_dim)

        self.real = nn.Linear(hidden_dim, embed_dim)
        self.imag = nn.Linear(hidden_dim, embed_dim)

        self.cls = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x):

        h = self.encoder(x)

        r = self.real(h)
        i = self.imag(h)

        amp = torch.sqrt(r ** 2 + i ** 2 + 1e-8)
        phase = torch.atan2(i, r)

        z = torch.cat([r, i], dim=-1)
        z = F.normalize(z, dim=-1)

        logits = self.cls(z)

        return r, i, amp, phase, z, logits


# ============================================================
# Coherence
# ============================================================

def coherence_score(amp_a, phase_a, amp_b, phase_b):

    return (amp_a * amp_b * torch.cos(phase_a - phase_b)).mean(dim=-1)


# ============================================================
# Training
# ============================================================

def train_experimental(cfg, train_loader, val_loader):

    model = PhaseCoherenceModel(cfg.hidden_dim, cfg.embed_dim, cfg.num_classes).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):

        model.train()

        loss_sum = 0
        coh_sum = 0

        for x, y in train_loader:

            x = x.to(cfg.device)
            y = y.to(cfg.device)

            x_ref = augment_signal(x)

            r, i, amp, phase, z, logits = model(x)
            r2, i2, amp2, phase2, _, _ = model(x_ref)

            per_sample_ce = F.cross_entropy(logits, y, reduction="none")

            coh = coherence_score(amp, phase, amp2.detach(), phase2.detach())

            alpha = torch.sigmoid(cfg.beta * coh)

            loss = (alpha * per_sample_ce).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item()
            coh_sum += coh.mean().item()

        val_acc, val_coh = evaluate_experimental(model, val_loader)

        print(
            f"Epoch {epoch+1}/{cfg.epochs} | "
            f"loss {loss_sum/len(train_loader):.4f} | "
            f"train_coh {coh_sum/len(train_loader):.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"val_coh {val_coh:.4f}"
        )

    return model


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_experimental(model, loader):

    model.eval()

    total = 0
    correct = 0
    coh_vals = []

    for x, y in loader:

        x = x.to(cfg.device)
        y = y.to(cfg.device)

        x_ref = augment_signal(x)

        r, i, amp, phase, z, logits = model(x)
        r2, i2, amp2, phase2, _, _ = model(x_ref)

        preds = logits.argmax(-1)

        total += y.size(0)
        correct += (preds == y).sum().item()

        coh = coherence_score(amp, phase, amp2, phase2)

        coh_vals.append(coh.mean().item())

    return correct / total, sum(coh_vals) / len(coh_vals)


# ============================================================
# Main
# ============================================================

def main():

    train_ds = PhaseStructuredDataset(True)
    val_ds = PhaseStructuredDataset(False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    train_experimental(cfg, train_loader, val_loader)


if __name__ == "__main__":
    main()