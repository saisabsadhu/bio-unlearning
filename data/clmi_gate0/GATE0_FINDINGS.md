# Gate 0 Results: CLMI Validity Pre-Check

Ran 2026-07-07 on BioMistral-7B (base/pre-unlearning checkpoint), 10 pilot concepts, 3 pooling schemes x {real labels, label-swapped labels}, 40 prompts/class, 5-fold CV. Full data: `gate0_summary.json`, per-concept: `<CUI>_gate0.json`.

## Results

| Pooling scheme | Mean real AUROC (10 concepts) | Mean label-swap AUROC |
|---|---|---|
| `original` (the existing clmi_prescreen_v2.py method: concept-word token search, last-token fallback for negatives) | 1.0000 | 0.5950 |
| `last_token` (fixed: last non-pad token, same rule both classes) | 1.0000 | 0.4866 |
| `mean_pool` (fixed: mean over all non-pad tokens, same rule both classes) | 1.0000 | 0.5778 |

## Interpretation

**Two separate questions were being conflated, and this run only answers one of them.**

1. **"Is the probe a degenerate/broken classifier?"** -- No. Label-swap AUROC sits close to chance (0.49-0.60) for every pooling scheme, which is the correct null-condition behavior for a valid probe. This was the original concern raised in the master plan (the suspicious `clmi_mean=1.0, std=0.0` in `clmi_prescreen_v2.py`), and it does NOT indicate a broken probe.

2. **"Does the original method's positional asymmetry (concept-word search for positives, last-token fallback for negatives) inflate the score?"** -- Inconclusive from this run, because real AUROC is 1.0 under *every* scheme, including the two symmetric, artifact-free ones (`last_token`, `mean_pool`). On an untouched base model, "sentences about concept A" vs. "sentences about a different concept B" are trivially linearly separable in residual-stream activations by topic/lexical content alone -- that is expected and unremarkable, and isn't yet a test of parametric knowledge erasure.

**The test CLMI actually needs to pass has not been run yet**, and can't be run until Gate 2 produces an actual post-unlearning checkpoint: does CLMI *drop* toward chance after a method that plausibly achieves genuine parametric erasure is applied, while *staying high* for a method known to only achieve behavioral suppression (e.g. WHP-style refusal training)? That pre-vs-post comparison, not a single pre-unlearning snapshot, is the real validity criterion, and it's a natural extension of Gate 2 rather than a separate step.

## Decision

- Adopt `mean_pool` as the standard CLMI activation-extraction method going forward (more principled than the original word-position search, which has a real asymmetry bug even though it didn't change the outcome in this particular pre-unlearning test -- it could matter once real unlearning perturbs the token-level surface features differently for pos/neg classes).
- Gate 0 is **partially cleared**: the "probe isn't degenerate" check passed. The "CLMI responds to genuine erasure" check is deferred to immediately after Gate 2 training, using the same 10 concepts, comparing pre- vs. post-unlearning CLMI under `mean_pool`.
- Do not yet treat any CLMI number as a finished paper metric -- this is pre-registration-style documentation of what's been checked and what's still open, per MASTER_RESEARCH_PLAN.md Section 5.
