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

## This GPU is shared with other users -- check `nvidia-smi` before assuming full capacity

This is a multi-tenant box, not exclusively ours. Mid-session, another user's unrelated job (`train_stage1.py --config configs/config_isles22.yaml`, a medical imaging segmentation training run, PID owned by a different Linux user) started up and held ~8.3GB of GPU memory for an extended period, which was enough to OOM our training runs even at settings that had fit comfortably before. **Never kill a process you don't recognize** -- check `ps aux` for the owning user first. Instead, reduce our own footprint: `per_device_train_batch_size: 2` (was 4), `gradient_accumulation_steps: 4` (was 2, keeps effective batch size the same), and `gradient_checkpointing: True` (was implicitly False via the `finetune.yaml` default) across all `configs/experiment/unlearn/bioun/*.yaml`. Always check `nvidia-smi --query-gpu=memory.free --format=csv` before launching a training run, not just before launching the first one of a session.

## NPO/RMU: resolved via LoRA + 8-bit reference model (do not wait for exclusive GPU access -- we will not get it)

Originally blocked: with the other tenant's job holding ~8.3GB, NPO/RMU OOM'd identically (`torch.OutOfMemoryError`, exactly 30.88 GiB in use) regardless of batch size or `max_length` -- confirming the ~30GB is fixed cost from the trainable model + frozen reference-model deep-copy + optimizer state, not activation memory. Quantizing *only* the reference model to 8-bit didn't move the needle (same 30.88GB), because the trainable model's own full-fine-tuning footprint was already the dominant cost.

**Resolved by combining two changes** (neither alone was enough):
1. `src/model/__init__.py`: opt-in LoRA wrapping, `BIOUNLEARN_USE_LORA=1` (plus `BIOUNLEARN_LORA_R`/`BIOUNLEARN_LORA_ALPHA` env vars, default r=16/alpha=32). Cuts trainable parameters to ~0.58% of the full model. Requires `model.enable_input_require_grads()` alongside `gradient_checkpointing=True`, and explicit dtype normalization on adapter params (peft's default LoRA dtype doesn't always match the base model's `torch_dtype`).
2. `src/trainer/unlearn/grad_diff.py`: the frozen KL/DPO reference-model copy now loads via `BitsAndBytesConfig(load_in_8bit=True)` by default instead of `copy.deepcopy`, opt-out via `BIOUNLEARN_REF_MODEL_FULL_PRECISION=1`.

**RMU needed three more fixes** on top of those two (see commit history for exact diffs): `module_regex` needs an optional `(base_model\.model\.)?` prefix to match PEFT-wrapped module names (RMU uses `re.fullmatch`); `trainable_params_regex` must be `.*lora.*` not `.*`, since RMU's `create_optimizer` explicitly re-enables `requires_grad` on every regex-matched param and `.*` would silently undo LoRA's savings by re-enabling gradients on the whole frozen base model; and `ref_act` needs an explicit `dtype=model_act.dtype` cast before `compute_activation_loss`, since bitsandbytes' 8-bit layers dequantize to fp16 internally regardless of the trainable model's bf16 dtype.

**Takeaway for future work on this box**: don't design around "wait for the GPU to free up" -- it's a shared resource and other tenants' jobs are unpredictable. LoRA + 8-bit reference models is now the standard way to run any KL/DPO-based method (NPO, RMU, GradDiff-KL, UNDIAL, WGA, SatImp, DPO) here, not a one-off workaround.

## GradAscent collapses completely at the original hyperparameters

The original GA configs (`lr=1e-5`, 5 epochs over ~58 examples, ~40 optimizer steps) drove `train_loss` from -30 to -222 with grad norms in the thousands -- unconstrained gradient ascent has no lower bound on loss, and it blew up. Post-training eval showed `RGU_forget_gen = RGU_retain_gen = IFE_forget_gen = IFE_retain_gen = 0.0` (the model's generations matched *nothing*, not even the retain answer -- pure gibberish), giving the degenerate pattern `FA=1.0, DEF=0.0` identically across both scenarios. This is GA's well-documented catastrophic-collapse failure mode, but it's scientifically useless for the diagnostic comparison Gate 2 needs: a fully collapsed model fails identically regardless of dataset, so it can't show whether BioUnlearn-Bench is harder than TOFU. Fixed by dropping to `lr=2e-6` and capping `max_steps=8` (~1 epoch) instead of running full 5-epoch schedules -- matching the master plan's own hyperparameter grid, which specifies small step counts (10/30/50) for GA specifically, not epoch counts. NPO/RMU include retain-loss regularization terms and are expected to be more stable at the original settings; left as-is pending their own results rather than pre-emptively retuning without evidence.
