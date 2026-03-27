# Phase-Induced Coherence-Gated Gradient Descent

Research prototype for testing whether **phase coherence between latent representations** can improve supervised learning.

This repo compares standard training against variants that:
- represent latent features as complex signals
- align latent amplitude and phase to a paired reference
- scale per-sample classification loss by a coherence-derived gate

## Method

Each latent component is treated as a complex signal

$$
h_i = A_i e^{i\phi_i}
$$

with model and reference signals

$$
\psi_\theta(x) = A_\theta(x)e^{i\phi_\theta(x)}, \quad
\psi_r(x) = A_r(x)e^{i\phi_r(x)}
$$

The coherence score is

$$
C(x) = \frac{1}{n}\sum_i |h_i||r_i|\cos(\phi_i - \phi_{r,i})
$$

and the coherence-gated update is

$$
g' = \alpha(x) g
$$

$$
\alpha(x) = \sigma(\beta C(x))
$$

The implementation uses a centered, bounded gate during training so gating is gentle rather than unstable.

## Current Protocol

The repo currently supports two paired-reference settings:

### Synthetic paired-view dataset

- Classes differ mainly by relative phase offset between sinusoidal components.
- Each item returns `(x, x_ref, y)`.
- In the default `paired_view` mode, `x` and `x_ref` share the same class template and base phase but use independent amplitude jitter and noise, so the reference is correlated but non-identical.
- `self` mode is still available for diagnostics, but it is not the default because it makes alignment too trivial.

### MedleyDB sample dataset

- Each item returns a song mix segment, an aligned stem segment, and the track label.
- Audio is cached in RAM to reduce repeated WAV loading.
- Validation uses a fixed deterministic segment schedule.
- Training uses a deterministic epoch-indexed schedule so all variants see the same sample order for a given seed and epoch.

Important: the current MedleyDB claim is limited to **same-track segment classification/learning**. This repo does **not** currently benchmark held-out track or artist generalization.

## Variants

| Variant | Description |
| --- | --- |
| `baseline` | Standard real-valued embedding and classifier |
| `complex` | Complex latent representation without alignment or gating |
| `align` | Complex latent representation with amplitude and phase alignment losses |
| `gate_only` | Complex latent representation with coherence-gated cross-entropy only |
| `full` | Alignment losses plus coherence-gated cross-entropy |

For fairness, all phase-based variants start each run from the same initial weights.

## Metrics And Artifacts

Per epoch, the runner logs:
- training loss
- training coherence
- mean gate value
- amplitude alignment loss
- phase alignment loss
- validation accuracy
- validation coherence
- epoch time

Artifacts are written under:

```text
results/<dataset>/<eval_protocol>/seed_<seed>/
```

with one `epochs.jsonl` file per variant plus `config.json` and `run_summary.json`. A cross-run `summary.json` is written at:

```text
results/<dataset>/<eval_protocol>/summary.json
```

Checkpoints remain optional and are saved as:

```text
checkpoints/baseline_best.pt
checkpoints/complex_best.pt
checkpoints/align_best.pt
checkpoints/gate_only_best.pt
checkpoints/full_best.pt
```

[HuggingFace Repo](https://huggingface.co/jzgdev/phase-induced-coherence-gated-gradient-descent)


## Installation

Requires Python 3.9+.

```bash
pip install torch soundfile
```

## Running Experiments

Default synthetic run:

```bash
python phase_coherence_test.py
```

Run all publication-style ablations explicitly:

```bash
python phase_coherence_test.py \
  --dataset synthetic \
  --variants baseline,complex,align,gate_only,full \
  --num_runs 3 \
  --results_dir results
```

Run on MedleyDB sample data:

```bash
python phase_coherence_test.py \
  --dataset medleydb_sample \
  --medleydb_root /path/to/MedleyDB_sample \
  --batch_size 16 \
  --num_workers 4 \
  --eval_protocol same_track_fixed
```

Useful flags:
- `--variants baseline,complex,align,gate_only,full`
- `--results_dir PATH`
- `--synthetic_reference_mode self|paired_view`
- `--eval_protocol same_track_fixed`
- `--save_checkpoints`

## Repository Layout

```text
phase-induced-coherence-gated-gradient-descent/
├── phase_coherence_test.py
├── datasets.py
├── test_phase_coherence.py
└── README.md
```

## Notes

- This is a research prototype, not a production training framework.
- Reported comparisons are intended to be seed-matched and initialization-matched.
- Publication-style reporting in this repo uses paired per-seed deltas, not significance testing.

## License

MIT
