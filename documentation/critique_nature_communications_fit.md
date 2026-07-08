# Critique: Is the BioUnlearn Research Plan Nature Communications-Ready?

Source reviewed: `documentation/BioUnlearn_EMNLP2026_FINAL_Proposal (1).docx` (1512 lines), cross-checked against repo state (pilot data, CLMI prescreen results, splits) as of 2026-07-07.

## Headline problem: venue mismatch

The document is titled, structured, and written entirely as an **EMNLP 2026 submission** — it references "ARR" (ACL Rolling Review), a "double-blind policy," an "ARR-required Responsible NLP Research Checklist," and has a section literally titled "Anticipated Reviewer Objections" written in the adversarial rebuttal style specific to NLP conference reviewing. But the repo's own `README.md` says "Targeting Nature Communications." These are not the same paper.

If Nature Communications (NC) is really the target, roughly a third of this document needs to be rebuilt, not edited:

- **Audience**: NC readers are broad scientists/clinicians, not NLP specialists. The current draft assumes familiarity with DPO, gradient ascent internals, PCA-on-residual-streams — this needs to lead with clinical stakes and be readable by a general science audience, with ML mechanics moved to methods/supplement.
- **Novelty bar**: NC wants *general significance*, not "three new metrics + method + benchmark" framed as incremental ML contributions. Right now this reads as a very good EMNLP/ACL paper, not obviously a Nature-tier one. Sharpen the "why should a non-NLP scientist care" story (e.g., frame around patient-safety consequences of guideline persistence, not metric novelty).
- **Review process artifacts**: strip ARR/double-blind language, add NC's actual requirements — Reporting Summary, Data Availability Statement, Code Availability Statement, Competing Interests, and an ethics/IRB statement that goes beyond "we have a PhysioNet DUA."
- **Human validation depth**: 2 annotators (MD-2 students, $20/hr, 40h each) validating 9.1% of the data is defensible for an EMNLP dataset paper, but thin for a journal making clinical-safety claims. NC reviewers (often including clinician referees) will likely want board-certified physician review, a larger/more representative validated subset, and a pre-registered protocol.

**Recommendation: clarify the venue decision before investing more in this document.** It's currently optimized for EMNLP; retargeting to NC without a rewrite pass risks desk rejection on fit/scope grounds regardless of the science's quality.

## Genuine strengths (keep these regardless of venue)

- The three-failure-mode structure (ontological entanglement, temporal directionality, verification opacity) is a clean, falsifiable diagnostic frame — each failure mode has its own formal metric, ablation, and a "what would disprove this" built in.
- The 3-day feasibility gate before building the full dataset (Section 10) is excellent research hygiene — pre-registering decision criteria before sinking 9 weeks into dataset construction is rare to see planned this explicitly.
- Cross-model control (Llama-3.1-8B as a non-clinical control) is a strong design choice — it's the one thing that lets the paper claim domain-specificity rather than a universal method quirk.
- The two "critical ablations" (OGFR-Random, DGP-NaiveSum) directly pre-empt the two most obvious reviewer objections with data rather than argument.

## Concerns independent of venue

1. **Circularity / construct validity risk.** Claude 3.5 Sonnet is used as (a) the oracle that generates BioUnlearn-Bench instances, and (b) the oracle that generates OGFR's retain-expansion instances. If the same LLM constructs both the test and the fix's training signal, a reviewer will ask whether OCD improvements reflect genuine UMLS-structural benefit or just "more data from the same generator that built the benchmark." The OGFR-Random ablation helps but doesn't fully address it — consider an independent oracle or template-based fallback for at least a subset.

2. **The "Expected Results" tables are fully fabricated to two-decimal precision** (Table 1: 89% FA, 2.8% OCD, etc.) before any experiment has run. Fine as an internal planning artifact, but risky: precise numbers are easy to anchor on and can create pressure to "hit" them rather than report what's actually observed. Strip or clearly mark as illustrative before this becomes a real draft; don't let hyperparameter sweeps be tuned toward matching them.

3. **No statistical rigor plan.** Tables 1-4 are point estimates with no seeds, no variance, no significance tests. The paper's entire argument rests on "baselines fail, BioUnlearn doesn't" — needs multiple seeds/hyperparameter reruns and confidence intervals; a 2-3 point OCD gap without variance bars isn't a claim in any venue.

4. **Missing an obvious baseline class**: knowledge-editing methods (ROME/MEMIT-style) and RAG/retrieval suppression aren't in the baseline set. For the RGU (temporally-superseded-guideline) scenario specifically, "just use RAG to override the old guideline" is the most obvious competing solution a reviewer will raise, and the proposal doesn't address why unlearning is preferable to editing/retrieval for this use case.

## Feasibility check against what's actually in the repo

Pilot is real but partial relative to the plan: `git log` shows 192 filtered pilot instances (95 RGU + 95 IFE) — **no PAC scenario yet**, against a planned 4,400 (2,800 forget). The CLMI prescreen (`data/clmi_prescreen_v2/confirmed_concepts_v2.json`) shows `clmi_mean: 1.0, std: 0.0` for essentially every concept at n=50/50. Worth checking before trusting the CLMI metric: perfect AUROC on every single pilot concept is suspicious for a metric that's supposed to discriminate "genuine erasure" (~0.5) from "still present" (~1.0) — could mean the probe is picking up a template/surface artifact between positive and negative prompt construction rather than true concept-level representation. Sanity-check with a scrambled/control condition (e.g., swap positive/negative labels, or use paraphrase-matched negatives) before relying on CLMI as a load-bearing metric.

## Bottom line

The underlying research design is strong — better than most conference submissions in this space, with real falsifiability and pre-registered decision gates. But as written, this is an EMNLP proposal, not an NC one, and treating it as "the plan" for NC submission would be a mistake without a deliberate reframing pass (audience, significance framing, clinical validation depth, ethics/IRB language) plus closing the statistical-rigor and circularity gaps above.
