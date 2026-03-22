import math
import random
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

from matplotlib.pylab import beta
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

    base_f1: float = 6.0
    base_f2: float = 12.0
    freq_jitter: float = 0.5
    amp_low: float = 0.7
    amp_high: float = 1.3
    phase_jitter: float = 0.35
    noise_std: float = 0.2

    aug_noise_std: float = 0.05
    aug_gain_low: float = 0.9
    aug_gain_high: float = 1.1
    aug_shift_min: int = -6
    aug_shift_max: int = 6

    lambda_amp: float = 0.1
    lambda_phase: float = 0.1


cfg = Config()


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
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

    def __init__(self, train: bool, cfg: Config):

        total = cfg.samples_per_class_train if train else cfg.samples_per_class_val

        t = torch.linspace(0, 1, cfg.seq_len)

        phase_offsets = [0, math.pi/4, math.pi/2, 3*math.pi/4]

        samples = []
        labels = []

        for label, offset in enumerate(phase_offsets):

            for _ in range(total):

                base_phase = random.uniform(0, 2*math.pi)

                a1 = random.uniform(cfg.amp_low, cfg.amp_high)
                a2 = random.uniform(cfg.amp_low, cfg.amp_high)

                f1 = cfg.base_f1 + random.uniform(-cfg.freq_jitter, cfg.freq_jitter)
                f2 = cfg.base_f2 + random.uniform(-cfg.freq_jitter, cfg.freq_jitter)

                jitter = random.uniform(-cfg.phase_jitter, cfg.phase_jitter)

                x = (
                    a1 * torch.sin(2*math.pi*f1*t + base_phase) +
                    a2 * torch.sin(2*math.pi*f2*t + base_phase + offset + jitter)
                )

                x += cfg.noise_std * torch.randn_like(x)

                x = (x - x.mean()) / (x.std() + 1e-6)

                samples.append(x.unsqueeze(0))
                labels.append(label)

        self.samples = torch.stack(samples)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.samples[i], self.labels[i]


# ============================================================
# Augmentations
# ============================================================

def augment_signal(x, cfg):

    x = x + cfg.aug_noise_std * torch.randn_like(x)

    gain = torch.empty(x.size(0),1,1,device=x.device).uniform_(cfg.aug_gain_low,cfg.aug_gain_high)
    x = x * gain

    shifts = torch.randint(cfg.aug_shift_min,cfg.aug_shift_max+1,(x.size(0),),device=x.device)

    x = torch.stack([torch.roll(x[i],int(shifts[i]),dims=-1) for i in range(x.size(0))])

    return x


# ============================================================
# Encoder
# ============================================================

class ConvEncoder(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv1d(1,32,7,padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),

            nn.Conv1d(32,64,7,padding=3,stride=2),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64,hidden_dim,5,padding=2,stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),

            nn.Conv1d(hidden_dim,hidden_dim,5,padding=2,stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self,x):

        h = self.net(x)
        return self.pool(h).squeeze(-1)


# ============================================================
# Models
# ============================================================

class BaselineModel(nn.Module):

    def __init__(self, hidden_dim, embed_dim, num_classes):

        super().__init__()

        self.encoder = ConvEncoder(hidden_dim)

        self.proj = nn.Linear(hidden_dim,embed_dim)

        self.cls = nn.Linear(embed_dim,num_classes)

    def forward(self,x):

        h = self.encoder(x)

        z = F.normalize(self.proj(h),dim=-1)

        logits = self.cls(z)

        return z,logits


class PhaseModel(nn.Module):

    def __init__(self, hidden_dim, embed_dim, num_classes):

        super().__init__()

        self.encoder = ConvEncoder(hidden_dim)

        self.real = nn.Linear(hidden_dim,embed_dim)
        self.imag = nn.Linear(hidden_dim,embed_dim)

        self.cls = nn.Linear(embed_dim*2,num_classes)

    def forward(self,x):

        h = self.encoder(x)

        r = self.real(h)
        i = self.imag(h)

        amp = torch.sqrt(r**2 + i**2 + 1e-8)

        phase = torch.atan2(i,r)

        z = torch.cat([r,i],dim=-1)
        z = F.normalize(z,dim=-1)

        logits = self.cls(z)

        return r,i,amp,phase,z,logits


# ============================================================
# Coherence
# ============================================================

def coherence_score(amp_a,phase_a,amp_b,phase_b):

    return (amp_a * amp_b * torch.cos(phase_a - phase_b)).mean(dim=-1)


def amp_loss(a,b):
    return ((a-b)**2).mean(dim=-1)


def phase_loss(a,b):
    return (1-torch.cos(a-b)).mean(dim=-1)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def eval_baseline(model,loader,cfg):

    model.eval()

    total=0
    correct=0

    for x,y in loader:

        x=x.to(cfg.device)
        y=y.to(cfg.device)

        _,logits=model(x)

        pred=logits.argmax(-1)

        total+=y.size(0)
        correct+=(pred==y).sum().item()

    return correct/total


@torch.no_grad()
def eval_phase(model,loader,cfg):

    model.eval()

    total=0
    correct=0
    coh_vals=[]

    for x,y in loader:

        x=x.to(cfg.device)
        y=y.to(cfg.device)

        x_ref=augment_signal(x,cfg)

        r,i,a,p,z,logits=model(x)
        r2,i2,a2,p2,_,_=model(x_ref)

        pred=logits.argmax(-1)

        total+=y.size(0)
        correct+=(pred==y).sum().item()

        coh=coherence_score(a,p,a2,p2)

        coh_vals.append(coh.mean().item())

    return correct/total,sum(coh_vals)/len(coh_vals)


# ============================================================
# Training variants
# ============================================================

def train_baseline(cfg,train_loader,val_loader):

    print("\n=== Baseline ===")

    model=BaselineModel(cfg.hidden_dim,cfg.embed_dim,cfg.num_classes).to(cfg.device)

    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr)

    for e in range(cfg.epochs):

        model.train()

        for x,y in train_loader:

            x=x.to(cfg.device)
            y=y.to(cfg.device)

            _,logits=model(x)

            loss=F.cross_entropy(logits,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        val=eval_baseline(model,val_loader,cfg)

        print(f"[Baseline] epoch {e+1} val_acc {val:.4f}")

    return model


def train_complex(cfg,train_loader,val_loader):

    print("\n=== Complex Latent Only ===")

    model=PhaseModel(cfg.hidden_dim,cfg.embed_dim,cfg.num_classes).to(cfg.device)

    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr)

    for e in range(cfg.epochs):

        model.train()

        for x,y in train_loader:

            x=x.to(cfg.device)
            y=y.to(cfg.device)

            _,_,_,_,_,logits=model(x)

            loss=F.cross_entropy(logits,y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        val,_=eval_phase(model,val_loader,cfg)

        print(f"[Complex] epoch {e+1} val_acc {val:.4f}")

    return model


def train_alignment(cfg,train_loader,val_loader):

    print("\n=== Alignment Loss Only ===")

    model=PhaseModel(cfg.hidden_dim,cfg.embed_dim,cfg.num_classes).to(cfg.device)

    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr)

    for e in range(cfg.epochs):

        model.train()

        for x,y in train_loader:

            x=x.to(cfg.device)
            y=y.to(cfg.device)

            x_ref=augment_signal(x,cfg)

            r,i,a,p,z,logits=model(x)
            r2,i2,a2,p2,_,_=model(x_ref)

            ce=F.cross_entropy(logits,y)

            loss=ce + cfg.lambda_amp*amp_loss(a,a2).mean() + cfg.lambda_phase*phase_loss(p,p2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        val,_=eval_phase(model,val_loader,cfg)

        print(f"[Align] epoch {e+1} val_acc {val:.4f}")

    return model


def train_full(cfg,train_loader,val_loader):

    print("\n=== Full Method (Coherence Gating) ===")

    model=PhaseModel(cfg.hidden_dim,cfg.embed_dim,cfg.num_classes).to(cfg.device)

    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr)

    for e in range(cfg.epochs):

        model.train()

        for x,y in train_loader:

            x=x.to(cfg.device)
            y=y.to(cfg.device)

            x_ref=augment_signal(x,cfg)

            r,i,a,p,z,logits=model(x)
            r2,i2,a2,p2,_,_=model(x_ref)

            ce=F.cross_entropy(logits,y,reduction="none")

            coh=coherence_score(a,p,a2.detach(),p2.detach())

            alpha = 1 + cfg.beta * coh
            alpha = torch.clamp(alpha, 0.5, 1.5)

            loss=(alpha*ce).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        val,coh_val=eval_phase(model,val_loader,cfg)

        print(f"[Full] epoch {e+1} val_acc {val:.4f} coh {coh_val:.4f}")

    return model


# ============================================================
# Experiment
# ============================================================

def run_once(cfg):

    set_seed(cfg.seed)

    train_ds=PhaseStructuredDataset(True,cfg)
    val_ds=PhaseStructuredDataset(False,cfg)

    train_loader=DataLoader(train_ds,batch_size=cfg.batch_size,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=cfg.batch_size)

    base=train_baseline(cfg,train_loader,val_loader)
    complex_model=train_complex(cfg,train_loader,val_loader)
    align=train_alignment(cfg,train_loader,val_loader)
    full=train_full(cfg,train_loader,val_loader)

    base_acc=eval_baseline(base,val_loader,cfg)
    complex_acc,_=eval_phase(complex_model,val_loader,cfg)
    align_acc,_=eval_phase(align,val_loader,cfg)
    full_acc,coh=eval_phase(full,val_loader,cfg)

    print("\nRESULTS")
    print("Baseline:",base_acc)
    print("Complex:",complex_acc)
    print("Align:",align_acc)
    print("Full:",full_acc)

    return base_acc,complex_acc,align_acc,full_acc


# ============================================================
# Main
# ============================================================

def main():

    results=[]

    for i in range(cfg.num_runs):

        run_cfg=replace(cfg,seed=cfg.seed+i)

        print("\n"+"="*60)
        print("RUN",run_cfg.seed)
        print("="*60)

        results.append(run_once(run_cfg))

    r=torch.tensor(results)

    print("\nSUMMARY")

    names=["Baseline","Complex","Align","Full"]

    for i,n in enumerate(names):

        vals=r[:,i]

        print(n,vals.mean().item(),"±",vals.std().item())


if __name__=="__main__":
    main()