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

## Single-GPU parallelism

This box has one A100 40GB. A single BioMistral-7B eval or training process uses ~15GB; two fit comfortably in parallel (~30GB, confirmed working). Useful for running two scenarios or two methods concurrently rather than serially -- just watch `nvidia-smi` before adding a third.
