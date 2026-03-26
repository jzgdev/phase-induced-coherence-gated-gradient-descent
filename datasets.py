import math
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
import soundfile as sf


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


def _resolve_audio_root(root: str) -> Path:
    """
    Accept either:
    - /workspace/medleydb/MedleyDB_sample
    - /workspace/medleydb/MedleyDB_sample/Audio
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
    candidates = sorted(track_dir.glob("*_MIX.wav"))
    if not candidates:
        raise FileNotFoundError(f"No *_MIX.wav found in {track_dir}")
    return candidates[0]


def _find_stem_dir(track_dir: Path) -> Path:
    candidates = sorted(track_dir.glob("*_STEMS"))
    if not candidates:
        raise FileNotFoundError(f"No *_STEMS directory found in {track_dir}")
    return candidates[0]


def _find_stem_files(stem_dir: Path) -> List[Path]:
    stems = sorted(stem_dir.glob("*.wav"))
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = min(len(mix_audio), len(stem_audio))

    if max_len <= 1:
        raise ValueError("Audio too short to extract a segment.")

    if max_len < segment_samples:
        pad = segment_samples - max_len
        mix_audio = torch.nn.functional.pad(mix_audio[:max_len], (0, pad))
        stem_audio = torch.nn.functional.pad(stem_audio[:max_len], (0, pad))
        start = 0
    else:
        start = random.randint(0, max_len - segment_samples)

    mix_seg = mix_audio[start:start + segment_samples]
    stem_seg = stem_audio[start:start + segment_samples]

    mix_seg = (mix_seg - mix_seg.mean()) / (mix_seg.std() + 1e-6)
    stem_seg = (stem_seg - stem_seg.mean()) / (stem_seg.std() + 1e-6)

    return mix_seg.unsqueeze(0), stem_seg.unsqueeze(0)


class MedleyDBSamplePairs(Dataset):
    """
    Each item:
    - x     = random segment from song mix
    - x_ref = aligned segment from one randomly selected stem
    - y     = track label

    Train/val use the SAME track label space.
    They differ only in RNG seed + repeated random segment sampling.
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

        all_track_dirs = sorted([p for p in self.audio_root.iterdir() if p.is_dir()])

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

        self.items: List[Tuple[Path, List[Path], int]] = []

        for track_dir in all_track_dirs:
            if track_dir.name not in label_map:
                continue

            mix_file = _find_mix_file(track_dir)
            stem_dir = _find_stem_dir(track_dir)
            stem_files = _find_stem_files(stem_dir)
            label = label_map[track_dir.name]

            self.items.append((mix_file, stem_files, label))

        if not self.items:
            raise ValueError("No usable MedleyDB Sample items found.")

        self.num_classes = len(self.track_names)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        mix_file, stem_files, label = self.items[self.rng.randrange(len(self.items))]

        mix_audio = _load_mono(mix_file, self.sample_rate)
        stem_audio = _load_mono(self.rng.choice(stem_files), self.sample_rate)

        x, x_ref = _extract_aligned_pair(
            mix_audio,
            stem_audio,
            self.segment_samples,
        )

        return x, x_ref, torch.tensor(label, dtype=torch.long)