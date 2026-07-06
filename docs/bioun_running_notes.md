# Running BioUnlearn experiments (practical notes)

## Correct Hydra entrypoint for unlearning runs

Use `--config-name=unlearn.yaml`, **not** `train.yaml`, for any unlearning experiment:

```
python src/train.py --config-name=unlearn.yaml experiment=unlearn/bioun/RGU_ga task_name=RGU_ga_pilot
```

`configs/train.yaml`'s defaults (`trainer: finetune`, `data: finetune`) don't set `trainer.args.remove_unused_columns: False`. Without that override, HF Trainer strips the custom `forget`/`retain` keys our `ForgetRetainDataset` produces (they aren't recognized model-forward columns), and training crashes on the first real step with `ValueError: The batch received was empty` -- confusingly, *after* the checkpoint-0 eval succeeds, since eval doesn't go through this code path. `configs/unlearn.yaml` sets `mode: unlearn` and the required override already; use it directly instead of also passing `mode=unlearn` on the CLI.

For eval-only runs (no training), `src/eval.py --config-name=eval.yaml` is correct as-is.

## Eval suite composes both scenarios regardless of what you're training

`configs/eval/bioun.yaml` includes RGU_* and IFE_* metrics unconditionally. Training on RGU alone still runs the full IFE eval at every checkpoint (a free cross-scenario collateral-damage check, but expensive). For faster iteration once you don't need per-epoch trajectories, add to the experiment config:

```yaml
trainer:
  args:
    eval_strategy: 'no'
    eval_on_start: False
```

This keeps the one required post-training eval (triggered unconditionally by `do_eval: True` in `src/train.py`) but skips the ~7x redundant per-epoch/start evals. Already applied to the NPO/RMU bioun configs; the GA configs still run the full per-epoch trace since that data was useful to keep once already in flight.

## Pre_compute cache keys must be scenario-qualified

Any new `configs/eval/bioun_metrics/*.yaml` metric config that references a `pre_compute` sub-metric must mount it under a scenario-qualified key (e.g. `.@pre_compute.RGU_forget_gen: RGU_forget_gen`, with `access_key: forget_gen` to remap for the metric function) -- **not** a generic key like `forget_gen`. The precompute cache is shared across the whole eval run; a generic key collides across scenarios and silently reuses the wrong scenario's cached generation (this actually happened -- see commit `62e4ca7`).

## Single-GPU parallelism -- eval yes, training no

This box has one A100 40GB. **Inference/eval-only** processes (`src/eval.py`, or the checkpoint-0 eval inside a training run before the first backward pass) use ~15GB each for BioMistral-7B; two run comfortably in parallel (~30GB, confirmed working).

**Training does not have the same headroom.** Gradients + optimizer states (`paged_adamw_32bit`) push a single BioMistral-7B fine-tune to ~24GB+ once the backward pass starts. Two simultaneous training runs OOM'd at the first backward pass (`torch.OutOfMemoryError`, ~39GB already in use before either could allocate more) even though both had looked fine during their eval-only startup phase -- the eval phase memory usage is not a reliable predictor of training memory usage. Run training jobs for this model size **sequentially**, one at a time; reserve parallel execution for eval-only workloads.
