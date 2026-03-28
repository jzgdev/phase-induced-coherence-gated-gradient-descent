import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import soundfile as sf
import torch
from torch.utils.data import DataLoader

from datasets import FMASmallPairs, MedleyDBSamplePairs, PhaseStructuredDataset
from phase_coherence_test import (
    Config,
    PhaseModel,
    amplitude_alignment_loss,
    capture_initial_states,
    compute_summary_stats,
    evaluate_phase,
    format_summary_line,
)


class PhaseCoherenceTests(unittest.TestCase):
    def test_synthetic_paired_view_preserves_labels_and_nonzero_alignment(self) -> None:
        cfg = Config(
            samples_per_class_train=4,
            samples_per_class_val=2,
            seq_len=64,
            noise_std=0.1,
            synthetic_reference_mode="paired_view",
        )
        dataset = PhaseStructuredDataset(train=True, cfg=cfg)

        xs = []
        x_refs = []
        labels = []
        for idx in range(len(dataset)):
            x, x_ref, y = dataset[idx]
            xs.append(x)
            x_refs.append(x_ref)
            labels.append(int(y))

        self.assertTrue(all(label in {0, 1, 2, 3} for label in labels))

        batch_x = torch.stack(xs[:8])
        batch_x_ref = torch.stack(x_refs[:8])
        self.assertGreater((batch_x - batch_x_ref).abs().mean().item(), 1e-4)

        model = PhaseModel(cfg.hidden_dim, cfg.embed_dim, cfg.num_classes)
        with torch.no_grad():
            _, _, amp, _, _, _ = model(batch_x)
            _, _, amp_ref, _, _, _ = model(batch_x_ref)
            amp_loss = amplitude_alignment_loss(amp, amp_ref).mean().item()

        self.assertGreater(amp_loss, 0.0)

    def test_medleydb_validation_schedule_and_metrics_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "MedleyDB_sample" / "Audio"
            self._write_mock_track(root, "TrackA", frequency=220.0)
            self._write_mock_track(root, "TrackB", frequency=330.0)

            cfg = Config(
                dataset="medleydb_sample",
                medleydb_root=str(root.parent),
                sample_rate=8000,
                segment_seconds=0.1,
                batch_size=2,
                num_workers=0,
                val_samples_per_epoch=6,
            )

            ds1 = MedleyDBSamplePairs(
                root=cfg.medleydb_root,
                sample_rate=cfg.sample_rate,
                segment_seconds=cfg.segment_seconds,
                samples_per_epoch=cfg.val_samples_per_epoch,
                seed=123,
                split="val",
            )
            ds2 = MedleyDBSamplePairs(
                root=cfg.medleydb_root,
                sample_rate=cfg.sample_rate,
                segment_seconds=cfg.segment_seconds,
                samples_per_epoch=cfg.val_samples_per_epoch,
                seed=123,
                split="val",
            )

            self.assertEqual(ds1.schedule, ds2.schedule)

            x1, x_ref1, y1 = ds1[0]
            x2, x_ref2, y2 = ds2[0]
            self.assertTrue(torch.allclose(x1, x2))
            self.assertTrue(torch.allclose(x_ref1, x_ref2))
            self.assertEqual(int(y1), int(y2))

            loader1 = DataLoader(ds1, batch_size=cfg.batch_size, shuffle=False)
            loader2 = DataLoader(ds2, batch_size=cfg.batch_size, shuffle=False)
            model = PhaseModel(cfg.hidden_dim, cfg.embed_dim, ds1.num_classes)

            metrics1 = evaluate_phase(model, loader1, cfg)
            metrics2 = evaluate_phase(model, loader2, cfg)

            self.assertAlmostEqual(float(metrics1["val_accuracy"]), float(metrics2["val_accuracy"]), places=7)
            self.assertAlmostEqual(float(metrics1["val_coherence"]), float(metrics2["val_coherence"]), places=7)
            self.assertIn("val_loss", metrics1)
            self.assertIn("val_loss", metrics2)

    def test_phase_initial_state_cloning_is_identical(self) -> None:
        cfg = Config()
        _, phase_state = capture_initial_states(cfg, cfg.num_classes)

        model_a = PhaseModel(cfg.hidden_dim, cfg.embed_dim, cfg.num_classes)
        model_b = PhaseModel(cfg.hidden_dim, cfg.embed_dim, cfg.num_classes)
        model_a.load_state_dict(phase_state)
        model_b.load_state_dict(phase_state)

        for (name_a, tensor_a), (name_b, tensor_b) in zip(
            model_a.state_dict().items(), model_b.state_dict().items()
        ):
            self.assertEqual(name_a, name_b)
            self.assertTrue(torch.equal(tensor_a, tensor_b))

    def test_single_run_summary_does_not_emit_nan(self) -> None:
        stats = compute_summary_stats([0.75])
        line = format_summary_line("baseline", [0.75])

        self.assertIsNone(stats["std"])
        self.assertIn("(n=1)", line)
        self.assertNotIn("nan", line.lower())

    def test_fma_small_metadata_and_schedule_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "fma"
            audio_root = root / "fma_small"
            metadata_root = root / "fma_metadata"

            self._write_mock_fma_track(audio_root, 1, frequency=220.0, sample_rate=16000)
            self._write_mock_fma_track(audio_root, 2, frequency=330.0)
            self._write_mock_fma_track(audio_root, 3, frequency=440.0)
            self._write_mock_fma_track(audio_root, 4, frequency=550.0)
            self._write_mock_fma_tracks_csv(metadata_root)

            train_ds = FMASmallPairs(
                root=str(root),
                metadata_root=str(metadata_root),
                sample_rate=8000,
                segment_seconds=0.1,
                samples_per_epoch=4,
                seed=7,
                split="train",
            )
            val_ds_a = FMASmallPairs(
                root=str(root),
                metadata_root=str(metadata_root),
                sample_rate=8000,
                segment_seconds=0.1,
                samples_per_epoch=4,
                seed=7,
                split="val",
                label_names=train_ds.label_names,
            )
            val_ds_b = FMASmallPairs(
                root=str(root),
                metadata_root=str(metadata_root),
                sample_rate=8000,
                segment_seconds=0.1,
                samples_per_epoch=4,
                seed=7,
                split="val",
                label_names=train_ds.label_names,
            )

            self.assertEqual(train_ds.num_classes, 2)
            self.assertEqual(val_ds_a.label_names, train_ds.label_names)
            self.assertEqual(val_ds_a.schedule, val_ds_b.schedule)

            x_a, x_ref_a, y_a = val_ds_a[0]
            x_b, x_ref_b, y_b = val_ds_b[0]
            self.assertTrue(torch.allclose(x_a, x_b))
            self.assertTrue(torch.allclose(x_ref_a, x_ref_b))
            self.assertEqual(int(y_a), int(y_b))
            self.assertEqual(x_a.shape[-1], int(0.1 * 8000))
            self.assertEqual(x_ref_a.shape[-1], int(0.1 * 8000))

    def test_fma_small_skips_unreadable_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "fma"
            audio_root = root / "fma_small"
            metadata_root = root / "fma_metadata"

            self._write_mock_fma_track(audio_root, 1, frequency=220.0)
            self._write_mock_fma_track(audio_root, 3, frequency=440.0)
            self._write_mock_fma_tracks_csv(metadata_root)
            self._write_corrupt_fma_track(audio_root, 2)

            train_ds = FMASmallPairs(
                root=str(root),
                metadata_root=str(metadata_root),
                sample_rate=8000,
                segment_seconds=0.1,
                samples_per_epoch=4,
                seed=7,
                split="train",
            )

            self.assertEqual(len(train_ds.audio_cache), 1)
            self.assertEqual(train_ds.audio_cache[0]["track_id"], 1)

    def test_fma_small_runtime_decode_failure_falls_back_instead_of_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "fma"
            audio_root = root / "fma_small"
            metadata_root = root / "fma_metadata"

            self._write_mock_fma_track(audio_root, 1, frequency=220.0)
            self._write_mock_fma_track(audio_root, 2, frequency=330.0)
            self._write_mock_fma_tracks_csv(metadata_root)

            train_ds = FMASmallPairs(
                root=str(root),
                metadata_root=str(metadata_root),
                sample_rate=8000,
                segment_seconds=0.1,
                samples_per_epoch=2,
                seed=7,
                split="train",
            )

            track_a = train_ds.audio_cache[0]["track_id"]
            track_b = train_ds.audio_cache[1]["track_id"]
            self.assertEqual((track_a, track_b), (1, 2))
            train_ds.schedule = [(0, 0, 10), (1, 0, 10)]

            original_load_mono = __import__("datasets")._load_mono
            failing_path = str(train_ds.audio_cache[0]["audio_path"])

            def fake_load_mono(path, expected_sr):
                if str(path) == failing_path:
                    raise RuntimeError("forced decode failure")
                return original_load_mono(path, expected_sr)

            with mock.patch("datasets._load_mono", side_effect=fake_load_mono):
                x, x_ref, y = train_ds[0]

            self.assertEqual(int(y), int(train_ds.audio_cache[1]["label"]))
            self.assertIn(1, train_ds.bad_track_ids)
            self.assertEqual(x.shape[-1], int(0.1 * 8000))
            self.assertEqual(x_ref.shape[-1], int(0.1 * 8000))

    def _write_mock_track(self, audio_root: Path, track_name: str, frequency: float) -> None:
        track_dir = audio_root / track_name
        stem_dir = track_dir / f"{track_name}_STEMS"
        stem_dir.mkdir(parents=True, exist_ok=True)

        sr = 8000
        duration = 0.5
        num_samples = int(sr * duration)
        t = torch.linspace(0.0, duration, num_samples)

        stem_a = 0.6 * torch.sin(2.0 * math.pi * frequency * t)
        stem_b = 0.4 * torch.sin(2.0 * math.pi * (frequency * 1.5) * t + 0.3)
        mix = stem_a + stem_b

        sf.write(track_dir / f"{track_name}_MIX.wav", mix.numpy(), sr)
        sf.write(stem_dir / f"{track_name}_STEM_01.wav", stem_a.numpy(), sr)
        sf.write(stem_dir / f"{track_name}_STEM_02.wav", stem_b.numpy(), sr)

    def _write_mock_fma_track(
        self,
        audio_root: Path,
        track_id: int,
        frequency: float,
        sample_rate: int = 8000,
    ) -> None:
        sr = sample_rate
        duration = 0.5
        num_samples = int(sr * duration)
        t = torch.linspace(0.0, duration, num_samples)
        audio = 0.6 * torch.sin(2.0 * math.pi * frequency * t)

        stem = f"{track_id:06d}"
        track_dir = audio_root / stem[:3]
        track_dir.mkdir(parents=True, exist_ok=True)
        sf.write(track_dir / f"{stem}.wav", audio.numpy(), sr)

    def _write_mock_fma_tracks_csv(self, metadata_root: Path) -> None:
        metadata_root.mkdir(parents=True, exist_ok=True)
        tracks_csv = metadata_root / "tracks.csv"
        tracks_csv.write_text(
            "\n".join(
                [
                    "track_id,track,set,set",
                    ",genre_top,subset,split",
                    "1,Rock,small,training",
                    "2,Jazz,small,training",
                    "3,Rock,small,validation",
                    "4,Jazz,small,validation",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_corrupt_fma_track(self, audio_root: Path, track_id: int) -> None:
        stem = f"{track_id:06d}"
        track_dir = audio_root / stem[:3]
        track_dir.mkdir(parents=True, exist_ok=True)
        (track_dir / f"{stem}.wav").write_bytes(b"not-audio")


if __name__ == "__main__":
    unittest.main()
