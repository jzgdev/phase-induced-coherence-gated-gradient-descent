import math
import random
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _normalize_waveform(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std() + 1e-6)


# ============================================================
# Synthetic dataset
# ============================================================


class PhaseStructuredDataset(Dataset):
    """
    Synthetic paired task:
    - class identity differs mainly by relative phase offset
    - x and x_ref share the same class template and base phase
    - x_ref can either mirror x exactly or be a correlated non-identical view
    """

    def __init__(self, train: bool, cfg):
        super().__init__()

        total = cfg.samples_per_class_train if train else cfg.samples_per_class_val
        phase_offsets = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
        rng = random.Random(cfg.seed + (0 if train else 10_000))
        torch_gen = torch.Generator().manual_seed(cfg.seed + (0 if train else 10_000))

        xs: List[torch.Tensor] = []
        x_refs: List[torch.Tensor] = []
        labels: List[int] = []

        for label, offset in enumerate(phase_offsets):
            for _ in range(total):
                base_phase = rng.uniform(0.0, 2.0 * math.pi)
                f1 = cfg.base_f1 + rng.uniform(-cfg.freq_jitter, cfg.freq_jitter)
                f2 = cfg.base_f2 + rng.uniform(-cfg.freq_jitter, cfg.freq_jitter)

                x = self._generate_view(
                    cfg=cfg,
                    base_phase=base_phase,
                    phase_offset=offset,
                    f1=f1,
                    f2=f2,
                    rng=rng,
                    torch_gen=torch_gen,
                )

                if cfg.synthetic_reference_mode == "self":
                    x_ref = x.clone()
                elif cfg.synthetic_reference_mode == "paired_view":
                    x_ref = self._generate_view(
                        cfg=cfg,
                        base_phase=base_phase,
                        phase_offset=offset,
                        f1=f1,
                        f2=f2,
                        rng=rng,
                        torch_gen=torch_gen,
                    )
                else:
                    raise ValueError(
                        f"Unknown synthetic_reference_mode: {cfg.synthetic_reference_mode}"
                    )

                xs.append(x.unsqueeze(0))
                x_refs.append(x_ref.unsqueeze(0))
                labels.append(label)

        perm = torch.randperm(len(labels), generator=torch_gen)

        self.samples = torch.stack(xs)[perm]
        self.references = torch.stack(x_refs)[perm]
        self.labels = torch.tensor(labels, dtype=torch.long)[perm]

    @staticmethod
    def _generate_view(
        cfg,
        base_phase: float,
        phase_offset: float,
        f1: float,
        f2: float,
        rng: random.Random,
        torch_gen: torch.Generator,
    ) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, cfg.seq_len)
        a1 = rng.uniform(cfg.amp_low, cfg.amp_high)
        a2 = rng.uniform(cfg.amp_low, cfg.amp_high)
        jitter = rng.uniform(-cfg.phase_jitter, cfg.phase_jitter)

        x = (
            a1 * torch.sin(2.0 * math.pi * f1 * t + base_phase)
            + a2 * torch.sin(2.0 * math.pi * f2 * t + base_phase + phase_offset + jitter)
        )

        noise = torch.randn(cfg.seq_len, generator=torch_gen) * cfg.noise_std
        return _normalize_waveform(x + noise)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.samples[idx], self.references[idx], self.labels[idx]


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
    try:
        audio, sr = sf.read(str(path), always_2d=False)
        audio = torch.tensor(audio, dtype=torch.float32)
    except RuntimeError:
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError(
                f"Could not read audio file {path} with soundfile, and torchaudio "
                "is not installed as a fallback decoder."
            ) from exc

        audio, sr = torchaudio.load(str(path))
        audio = audio.to(dtype=torch.float32)
        if audio.ndim == 2:
            audio = audio.mean(dim=0)

    if sr != expected_sr:
        raise ValueError(
            f"Sample rate mismatch for {path}: got {sr}, expected {expected_sr}"
        )

    if audio.ndim == 2:
        audio = audio.mean(dim=1)

    return audio


def _slice_aligned_pair(
    mix_audio: torch.Tensor,
    stem_audio: torch.Tensor,
    segment_samples: int,
    start: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = min(len(mix_audio), len(stem_audio))

    if max_len <= 1:
        raise ValueError("Audio too short to extract a segment.")

    if max_len < segment_samples:
        pad = segment_samples - max_len
        mix_seg = F.pad(mix_audio[:max_len], (0, pad))
        stem_seg = F.pad(stem_audio[:max_len], (0, pad))
    else:
        mix_seg = mix_audio[start:start + segment_samples]
        stem_seg = stem_audio[start:start + segment_samples]

    return _normalize_waveform(mix_seg).unsqueeze(0), _normalize_waveform(stem_seg).unsqueeze(0)


def _slice_single_segment(
    audio: torch.Tensor,
    segment_samples: int,
    start: int,
) -> torch.Tensor:
    max_len = len(audio)

    if max_len <= 1:
        raise ValueError("Audio too short to extract a segment.")

    if max_len < segment_samples:
        pad = segment_samples - max_len
        segment = F.pad(audio[:max_len], (0, pad))
    else:
        segment = audio[start:start + segment_samples]

    return _normalize_waveform(segment).unsqueeze(0)


# ============================================================
# FMA small helpers
# ============================================================


def _resolve_fma_audio_root(root: str) -> Path:
    root_path = Path(root)

    if (root_path / "fma_small").exists():
        return root_path / "fma_small"

    if root_path.name == "fma_small":
        return root_path

    raise FileNotFoundError(
        f"Could not find FMA audio root under: {root_path}. "
        f"Expected either root/fma_small or root itself to be fma_small."
    )


def _resolve_fma_metadata_root(audio_root: Path, metadata_root: Optional[str]) -> Path:
    candidates: List[Path] = []

    if metadata_root is not None:
        explicit = Path(metadata_root)
        candidates.append(explicit)
        candidates.append(explicit / "fma_metadata")

    candidates.append(audio_root.parent / "fma_metadata")
    candidates.append(audio_root.parent.parent / "fma_metadata")

    for candidate in candidates:
        if (candidate / "tracks.csv").exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find FMA metadata root for audio root {audio_root}. "
        "Expected a directory containing tracks.csv."
    )


def _find_fma_audio_file(audio_root: Path, track_id: int) -> Path:
    stem = f"{track_id:06d}"
    subdir = stem[:3]

    for ext in (".mp3", ".wav"):
        candidate = audio_root / subdir / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find audio for FMA track {track_id} under {audio_root}"
    )


def _load_fma_small_rows(
    audio_root: Path,
    metadata_root: Path,
    split: str,
    max_tracks: int = 0,
    label_names: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, object]], List[str]]:
    split_map = {
        "train": "training",
        "val": "validation",
        "test": "test",
    }

    if split not in split_map:
        raise ValueError(f"Unknown FMA split: {split}")

    tracks_csv = metadata_root / "tracks.csv"

    with tracks_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header_top = next(reader)
        header_sub = next(reader)

        columns: Dict[Tuple[str, str], int] = {}
        for idx, (top, sub) in enumerate(zip(header_top, header_sub)):
            columns[(top.strip(), sub.strip())] = idx

        required = [
            ("track", "genre_top"),
            ("set", "subset"),
            ("set", "split"),
        ]
        missing = [key for key in required if key not in columns]
        if missing:
            raise ValueError(
                f"FMA tracks.csv is missing required columns: {missing}"
            )

        all_rows: List[Dict[str, object]] = []
        all_labels: List[str] = []

        for row in reader:
            if not row or not row[0].strip().isdigit():
                continue

            subset = row[columns[("set", "subset")]].strip()
            split_name = row[columns[("set", "split")]].strip()
            genre_top = row[columns[("track", "genre_top")]].strip()

            if subset != "small" or not genre_top:
                continue

            track_id = int(row[0].strip())
            audio_path = _find_fma_audio_file(audio_root, track_id)

            all_rows.append(
                {
                    "track_id": track_id,
                    "genre_top": genre_top,
                    "split": split_name,
                    "audio_path": audio_path,
                }
            )
            all_labels.append(genre_top)

    if label_names is None:
        resolved_label_names = sorted(set(all_labels))
    else:
        resolved_label_names = list(label_names)

    wanted_split = split_map[split]
    rows = [row for row in all_rows if row["split"] == wanted_split]

    if max_tracks > 0:
        rows = rows[:max_tracks]

    if not rows:
        raise ValueError(
            f"No FMA small rows found for split '{split}' in {tracks_csv}"
        )

    return rows, resolved_label_names


# ============================================================
# FMA small dataset
# ============================================================


class FMASmallPairs(Dataset):
    """
    Each item:
    - x     = segment from one FMA track
    - x_ref = second segment from the SAME track
    - y     = top-level genre label

    Train/val use the official FMA split from tracks.csv.
    Validation uses a fixed deterministic schedule.
    Training uses an epoch-indexed deterministic schedule.
    """

    def __init__(
        self,
        root: str,
        metadata_root: Optional[str] = None,
        sample_rate: int = 44100,
        segment_seconds: float = 2.0,
        max_tracks: int = 0,
        samples_per_epoch: int = 1024,
        seed: int = 42,
        split: str = "train",
        label_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()

        self.audio_root = _resolve_fma_audio_root(root)
        self.metadata_root = _resolve_fma_metadata_root(self.audio_root, metadata_root)
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_seconds)
        self.samples_per_epoch = samples_per_epoch
        self.base_seed = seed
        self.split = split

        rows, resolved_label_names = _load_fma_small_rows(
            audio_root=self.audio_root,
            metadata_root=self.metadata_root,
            split=split,
            max_tracks=max_tracks,
            label_names=label_names,
        )

        self.label_names = list(resolved_label_names)
        label_map = {name: idx for idx, name in enumerate(self.label_names)}
        self.num_classes = len(self.label_names)

        self.audio_cache: List[Dict[str, object]] = []

        print(
            f"[FMASmallPairs] caching {len(rows)} tracks into RAM "
            f"({self.num_classes} genre classes, split={split})",
            flush=True,
        )

        for idx, row in enumerate(rows):
            audio = _load_mono(row["audio_path"], self.sample_rate)
            self.audio_cache.append(
                {
                    "audio": audio,
                    "label": label_map[row["genre_top"]],
                    "genre_top": row["genre_top"],
                    "track_id": row["track_id"],
                }
            )

            if idx % 250 == 0 or idx == len(rows) - 1:
                print(
                    f"[FMASmallPairs] cached {idx + 1}/{len(rows)} tracks",
                    flush=True,
                )

        self.schedule = self._build_schedule(epoch=0)

    def _sample_start(self, rng: random.Random, max_len: int) -> int:
        if max_len <= 1 or max_len < self.segment_samples:
            return 0
        return rng.randint(0, max_len - self.segment_samples)

    def _build_schedule(self, epoch: int) -> List[Tuple[int, int, int]]:
        seed = self.base_seed if self.split != "train" else self.base_seed + epoch
        rng = random.Random(seed)
        schedule: List[Tuple[int, int, int]] = []

        for _ in range(self.samples_per_epoch):
            track_idx = rng.randrange(len(self.audio_cache))
            entry = self.audio_cache[track_idx]
            audio = entry["audio"]
            max_len = len(audio)

            start_a = self._sample_start(rng, max_len)
            start_b = self._sample_start(rng, max_len)

            if max_len > self.segment_samples and start_b == start_a:
                start_b = self._sample_start(rng, max_len)

            schedule.append((track_idx, start_a, start_b))

        return schedule

    def set_epoch(self, epoch: int) -> None:
        if self.split != "train":
            return

        self.schedule = self._build_schedule(epoch=epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        track_idx, start_a, start_b = self.schedule[idx]
        entry = self.audio_cache[track_idx]

        x = _slice_single_segment(
            audio=entry["audio"],
            segment_samples=self.segment_samples,
            start=start_a,
        )
        x_ref = _slice_single_segment(
            audio=entry["audio"],
            segment_samples=self.segment_samples,
            start=start_b,
        )

        return x, x_ref, torch.tensor(entry["label"], dtype=torch.long)


# ============================================================
# MedleyDB Sample dataset
# ============================================================


class MedleyDBSamplePairs(Dataset):
    """
    Each item:
    - x     = aligned segment from song mix
    - x_ref = aligned segment from one selected stem
    - y     = track label

    Train/val use the same track label space.
    Validation uses a fixed deterministic schedule.
    Training uses an epoch-indexed deterministic schedule.

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
        split: str = "train",
    ):
        super().__init__()

        if split not in {"train", "val"}:
            raise ValueError(f"Unknown split: {split}")

        self.audio_root = _resolve_audio_root(root)
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_seconds)
        self.samples_per_epoch = samples_per_epoch
        self.base_seed = seed
        self.split = split
        self.current_epoch = 0

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

        self.schedule = self._build_schedule(epoch=0)

    def _build_schedule(self, epoch: int) -> List[Tuple[int, int, int]]:
        seed = self.base_seed if self.split == "val" else self.base_seed + epoch
        rng = random.Random(seed)
        schedule: List[Tuple[int, int, int]] = []

        for _ in range(self.samples_per_epoch):
            track_idx = rng.randrange(len(self.audio_cache))
            entry = self.audio_cache[track_idx]
            stems = entry["stems"]
            stem_idx = rng.randrange(len(stems))

            mix_audio = entry["mix"]
            stem_audio = stems[stem_idx]
            max_len = min(len(mix_audio), len(stem_audio))

            if max_len <= 1 or max_len < self.segment_samples:
                start = 0
            else:
                start = rng.randint(0, max_len - self.segment_samples)

            schedule.append((track_idx, stem_idx, start))

        return schedule

    def set_epoch(self, epoch: int) -> None:
        if self.split != "train":
            return

        self.current_epoch = epoch
        self.schedule = self._build_schedule(epoch=epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int):
        track_idx, stem_idx, start = self.schedule[idx]
        entry = self.audio_cache[track_idx]

        mix_audio = entry["mix"]
        stem_audio = entry["stems"][stem_idx]
        label = entry["label"]

        x, x_ref = _slice_aligned_pair(
            mix_audio=mix_audio,
            stem_audio=stem_audio,
            segment_samples=self.segment_samples,
            start=start,
        )

        return x, x_ref, torch.tensor(label, dtype=torch.long)
