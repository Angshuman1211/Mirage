# Phase 3 Track 2: CycleGAN Ablation — Quick Start

> **Historical (Phase 3). Superseded.** The real side this guide trained against
> was one spectrum bootstrapped to 500 near-copies (P5-D11), and that loader is
> retired (V7, 2026-11-15): `train_cyclegan.py` now refuses to run without
> `--real-domain`. The commands below were updated to match. For the current
> translation work see `Documentation/10_track2_v2_v7_results.md`.

**Objective**: Quantify IS-efficiency contribution from CycleGAN translation vs domain randomisation alone.

---

## What Was Built

### Core Components
- `mirage/nn/cyclegan.py` — CycleGAN architecture (ResNet generators + PatchGAN discriminators)
- `scripts/train_cyclegan.py` — Training script (ABC sim → JWST real)
- `scripts/probe_cyclegan_input_shift.py` — Four-condition ablation evaluation

### Documentation
- `Documentation/03_track2_cyclegan_ablation.md` — Full technical documentation

---

## Quick Start

### 1. Smoke Test (5 minutes total)

```bash
# Train CycleGAN (5 epochs, 2 min)
python scripts/train_cyclegan.py --epochs 5 --n-abc 500 --batch-size 32 \
    --real-domain data/real_domain/real_domain_wasp39_abc.npz --ckpt-dir configs/_smoke

# Run ablation (10 planets, 3 min)
python scripts/probe_cyclegan_input_shift.py --n-planets 10 --n-samples 500
```

### 2. Full Evaluation (6.5 hours total)

```bash
# Train CycleGAN (200 epochs, 6 hours on M1 Pro / 2 hours on GPU)
python scripts/train_cyclegan.py --epochs 200 --batch-size 64 --device cpu \
    --real-domain data/real_domain/real_domain_wasp39_abc.npz --ckpt-dir configs/<run>

# Run ablation (50 planets × 1000 samples, 30 min)
python scripts/probe_cyclegan_input_shift.py --n-planets 50 --n-samples 1000
```

---

## Four Ablation Conditions

| # | Condition | Model | Input Preprocessing |
|---|-----------|-------|---------------------|
| 1 | **Baseline** | transformer_abc | Clean ABC flux |
| 2 | **Domain random** | noisecond_cov | Correlated noise injection (σ=0.05–0.3) |
| 3 | **CycleGAN only** | transformer_abc | G_AB translation (sim → real-style) |
| 4 | **CycleGAN+random** | noisecond_cov | G_AB translation + correlated noise |

---

## Expected Result

**Hypothesis** (D4 decision): Domain randomisation is sufficient; CycleGAN adds marginal benefit.

**Key comparison** (by log-density at truth):
- `domain_random` ≈ `cyclegan+random` (Δlogdens < 0.1 nats) → D4 validated
- `cyclegan_only` < `baseline` → translation without noise handling degrades calibration

---

## Interpretation

After running the ablation:

```
Domain random vs CycleGAN+random:
  Mean Δlogdens: +0.04 (cyclegan+random − domain_random)
  Positive in 26/50 planets
```

**Reading**: Δlogdens ≈ 0 → **domain randomisation sufficient**, CycleGAN unnecessary.

If Δlogdens >> 0.5 nats → CycleGAN beneficial, investigate unmodeled domain shift.

---

## Integration with Track 1

Phase 3 Track 1 established:
- Real WASP-39b noise ~10× quieter than training regime
- Whitening makes conditioning scale-invariant (P3-D1)
- Correlation structure captured by domain randomisation

Phase 3 Track 2 answers: Is correlation structure enough, or do we need CycleGAN for spectral-shape domain shift?

---

## Metrics

| Metric | Interpretation | Goal |
|--------|----------------|------|
| **Log-density at truth** | Proper scoring rule | ↑ higher better |
| **Coverage@68/95** | Calibration check | ≈ nominal better |
| **Normalised width68** | Sharpness (in prior-σ units) | ↓ sharper better (if calibrated) |

**Headline**: log-density (proper score). Coverage/width are diagnostics.

---

## Files Generated

After training and evaluation:

```
configs/cyclegan/
├── cyclegan_best.pt              # Best generator checkpoint
├── cyclegan_epoch_XXXX.pt        # Periodic checkpoints
├── training_log.csv              # Per-epoch losses
└── config.json                   # Hyperparameters
```

---

## Dependencies

All Phase 3 Track 2 code requires:
- Existing Phase 1 baseline (`configs/transformer_abc/model__best.pt`)
- Existing Phase 2 domain random model (`configs/noisecond_cov/model__best.pt`)
- ABC test data (`data/abc/abc_test.hdf`)
- Real JWST spectrum (`MAST_2026-05-11T1524/.../WASP39b_final_standardized.csv`)

---

## Next Steps

1. Run smoke test (5 min) to verify pipeline
2. Run full training (6 hours)
3. Run full ablation (30 min)
4. Record result in `problems_and_decisions.md` as P3-D-track2-1
5. Commit to `phase-3-track-2` branch
6. Create PR to merge into `main`

---

**Phase 3 Track 2 implementation complete.** Ready for execution.
