# Kumo Tabular output distillation V1

This package is an isolated mechanical experiment for transferring raw output
behavior from pretrained Kumo Tabular Large to pretrained Kumo Tabular Small.
It does not modify Kumo model classes, reproduce NVIDIA's pretraining prior, or
establish a benchmark improvement.

## What is implemented

- Frozen Large teacher and trainable pretrained Small student.
- Classification KD for 2-10 classes, before ECOC and output processing.
- Regression KD across all 999 fixed quantile channels.
- Mixed hard-label and teacher losses, including hard-label-only `--alpha 0`.
- A replaceable `TaskSource` protocol and deterministic random SCM source.
- AdamW, warmup/cosine decay, gradient clipping, accumulation, and AMP.
- Atomic checkpoints with optimizer, scheduler, global RNG, and task RNG state.
- Structured JSONL metrics and provenance metadata.
- One held-out synthetic evaluator for teacher, baseline Small, hard-label
  continuation, and distilled Small.

The synthetic SCM source exists to validate mechanics. It is not a substitute
for the unavailable Kumo synthetic pretraining generator, and results from it
must not be described as reproducing Kumo pretraining.

## Train

Classification distillation:

```bash
python -m experiments.kumo_distill.train \
  --task classification \
  --teacher-size large \
  --student-size small \
  --steps 10000 \
  --learning-rate 1e-5 \
  --alpha 0.7 \
  --temperature 2.0 \
  --seed 0 \
  --output-dir outputs/kumo-large-to-small/classification
```

Run the matched hard-label continuation with the same arguments and
`--alpha 0`, writing to a separate output directory.

Regression distillation:

```bash
python -m experiments.kumo_distill.train \
  --task regression \
  --teacher-size large \
  --student-size small \
  --steps 10000 \
  --learning-rate 1e-5 \
  --alpha 0.5 \
  --seed 0 \
  --output-dir outputs/kumo-large-to-small/regression
```

Resume with the original arguments plus
`--resume outputs/.../latest.pt`. The saved task-generator state ensures that
the next sampled episode matches an uninterrupted run.

For the first overfit smoke test, use one shape and frequent checkpoints:

```bash
python -m experiments.kumo_distill.train \
  --task classification \
  --steps 100 \
  --fixed-shape \
  --context-rows 64 \
  --query-rows 32 \
  --num-features 8 \
  --log-interval 1 \
  --checkpoint-interval 100 \
  --output-dir outputs/kumo-smoke
```

## Evaluate

Use a seed not used by training and compare all required conditions on the
same fixed episodes:

```bash
python -m experiments.kumo_distill.evaluate \
  --task classification \
  --hard-label-checkpoint outputs/kumo-hard/latest.pt \
  --distilled-checkpoint outputs/kumo-kd/latest.pt \
  --seed 10000 \
  --output outputs/kumo-evaluation/classification.json
```

Public TabArena, TALENT, and ScoringBench runs remain a separate evidence
milestone. They must use untouched test splits and matched `num_estimators=8`
before native-default comparisons or performance claims.

## Tests

```bash
python -m pytest test/experiments/kumo_distill
```

The CPU tests use tiny networks and do not download pretrained weights. A real
Large-to-Small smoke run requires enough accelerator memory and network access
for the released checkpoints.

## Weight license

Kumo Tabular weights at `nvidia/Kumo-Tabular`, revision `v1.0.1`, are released
under OpenMDW 1.1 rather than this repository's Apache-2.0 source license. If
distilled weights contain or derive from those model materials, redistribution
must retain a copy of OpenMDW 1.1 and all applicable copyright and origin
notices from the model materials. OpenMDW 1.1 says it does not restrict outputs
generated using the model materials, but that output clause should not be used
to relabel derivative weights without legal review. Include the teacher model,
revision, this experiment configuration, and NVIDIA attribution in any model
card. This is an engineering summary, not legal advice; review the current
license text before publication.