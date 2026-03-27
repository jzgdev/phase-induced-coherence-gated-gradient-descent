import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import soundfile as sf


# ============================================================
# Synthetic dataset
# ============================================================

class PhaseStructuredDataset(Dataset):
    """
    Synthetic task:
    - same base frequencies for all classes
    - class identity differs mainly by relative phase offset
    """

    def __init__(self, train: bool, cfg):
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

    def __getitem__(self, idx: int):
        return self.samples[idx], self.labels[idx]


# ============================================================
# MedleyDB helpers
# ============================================================

def _is_junk(path: Path) -> bool:
    name = path.name
    return name.startswith("._") or name == ".DS_Store"


def _resolve_audio_root(root: str) -> Path:
    """
    Accept either:
    - /workspace/data/MedleyDB_sample
    - /workspace/data/MedleyDB_sample/Audio
    """
    root_path = Path(root)

    if (root_path / "Audio").exists():
        return root_path / "Audio"

    if root_path.name == "Audio":
        return root_path

    raise FileNotFoundError(
        f"Could not find MedleyDB audio root under: {root_path}. "
        f"Expected either root/Audio or root itself to be Audio."
    )


def _find_mix_file(track_dir: Path) -> Path:
    candidates = [
        p for p in sorted(track_dir.glob("*_MIX.wav"))
        if not _is_junk(p)
    ]
    if not candidates:
        raise FileNotFoundError(f"No *_MIX.wav found in {track_dir}")
    return candidates[0]


def _find_stem_dir(track_dir: Path) -> Path:
    candidates = [
        p for p in sorted(track_dir.glob("*_STEMS"))
        if not _is_junk(p)
    ]
    if not candidates:
        raise FileNotFoundError(f"No *_STEMS directory found in {track_dir}")
    return candidates[0]


def _find_stem_files(stem_dir: Path) -> List[Path]:
    stems = [
        p for p in sorted(stem_dir.glob("*.wav"))
        if not _is_junk(p)
    ]
    if not stems:
        raise FileNotFoundError(f"No stem wavs found in {stem_dir}")
    return stems


def _load_mono(path: Path, expected_sr: int) -> torch.Tensor:
    audio, sr = sf.read(str(path), always_2d=False)

    if sr != expected_sr:
        raise ValueError(
            f"Sample rate mismatch for {path}: got {sr}, expected {expected_sr}"
        )

    audio = torch.tensor(audio, dtype=torch.float32)

    if audio.ndim == 2:
        audio = audio.mean(dim=1)

    return audio


def _extract_aligned_pair(
    mix_audio: torch.Tensor,
    stem_audio: torch.Tensor,
    segment_samples: int,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = min(len(mix_audio), len(stem_audio))

    if max_len <= 1:
        raise ValueError("Audio too short to extract a segment.")

    if max_len < segment_samples:
        pad = segment_samples - max_len
        mix_audio = F.pad(mix_audio[:max_len], (0, pad))
        stem_audio = F.pad(stem_audio[:max_len], (0, pad))
        start = 0
    else:
        start = rng.randint(0, max_len - segment_samples)

    mix_seg = mix_audio[start:start + segment_samples]
    stem_seg = stem_audio[start:start + segment_samples]

    mix_seg = (mix_seg - mix_seg.mean()) / (mix_seg.std() + 1e-6)
    stem_seg = (stem_seg - stem_seg.mean()) / (stem_seg.std() + 1e-6)

    return mix_seg.unsqueeze(0), stem_seg.unsqueeze(0)


# ============================================================
# MedleyDB Sample dataset
# ============================================================

class MedleyDBSamplePairs(Dataset):
    """
    Each item:
    - x     = random segment from song mix
    - x_ref = aligned segment from one randomly selected stem
    - y     = track label

    Train/val use the SAME track label space.
    Repeated sampling is handled by samples_per_epoch.

    IMPORTANT:
    This class caches audio in RAM to avoid repeated disk I/O.
    """

    def __init__(
        self,
        root: str,
        sample_rate: int = 44100,
        segment_seconds: float = 2.0,
        max_tracks: int = 0,
        samples_per_epoch: int = 1024,
        seed: int = 42,
        track_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()

        self.audio_root = _resolve_audio_root(root)
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_seconds)
        self.samples_per_epoch = samples_per_epoch
        self.rng = random.Random(seed)

        all_track_dirs = sorted([
            p for p in self.audio_root.iterdir()
            if p.is_dir() and not _is_junk(p)
        ])

        if max_tracks > 0:
            all_track_dirs = all_track_dirs[:max_tracks]

        if track_names is None:
            self.track_names = [p.name for p in all_track_dirs]
        else:
            self.track_names = list(track_names)
            wanted = set(self.track_names)
            all_track_dirs = [p for p in all_track_dirs if p.name in wanted]

        if not all_track_dirs:
            raise ValueError(f"No MedleyDB track folders found in {self.audio_root}")

        label_map = {name: idx for idx, name in enumerate(self.track_names)}

        # Metadata: file paths
        self.items: List[Tuple[Path, List[Path], int, str]] = []

        for track_dir in all_track_dirs:
            if track_dir.name not in label_map:
                continue

            mix_file = _find_mix_file(track_dir)
            stem_dir = _find_stem_dir(track_dir)
            stem_files = _find_stem_files(stem_dir)
            label = label_map[track_dir.name]

            self.items.append((mix_file, stem_files, label, track_dir.name))

        if not self.items:
            raise ValueError("No usable MedleyDB Sample items found.")

        self.num_classes = len(self.track_names)

        # Audio cache: one entry per track
        # Each entry is:
        # {
        #   "mix": Tensor[T],
        #   "stems": List[Tensor[T]],
        #   "label": int,
        #   "name": str,
        # }
        self.audio_cache: List[Dict[str, object]] = []

        print(
            f"[MedleyDBSamplePairs] caching {len(self.items)} tracks into RAM "
            f"(split-independent label space: {self.num_classes} classes)",
            flush=True,
        )

        for idx, (mix_file, stem_files, label, track_name) in enumerate(self.items):
            mix_audio = _load_mono(mix_file, self.sample_rate)
            stem_audios = [_load_mono(stem_file, self.sample_rate) for stem_file in stem_files]

            self.audio_cache.append(
                {
                    "mix": mix_audio,
                    "stems": stem_audios,
                    "label": label,
                    "name": track_name,
                }
            )

            if idx % 5 == 0 or idx == len(self.items) - 1:
                print(
                    f"[MedleyDBSamplePairs] cached {idx + 1}/{len(self.items)} tracks",
                    flush=True,
                )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        entry = self.audio_cache[self.rng.randrange(len(self.audio_cache))]

        mix_audio = entry["mix"]
        stem_audio = self.rng.choice(entry["stems"])
        label = entry["label"]

        x, x_ref = _extract_aligned_pair(
            mix_audio,
            stem_audio,
            self.segment_samples,
            self.rng,
        )

        return x, x_ref, torch.tensor(label, dtype=torch.long)