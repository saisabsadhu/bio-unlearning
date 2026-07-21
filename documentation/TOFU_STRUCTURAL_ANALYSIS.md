# Why TOFU cannot test ontology-aware collateral damage (structural analysis)

**Question this answers**: the paper's core diagnostic claim is that clinical concepts'
ontological entanglement (shared UMLS/RxNorm neighbor structure) causes *collateral* damage
on retain-critical neighbor concepts when a target fact is unlearned. The cross-dataset
GA/NPO/RMU runs (see `PI_STATUS_REPORT.md` Section 3) established that TOFU and
BioUnlearn-Bench show the same collapse-vs-non-collapse behavior at matched doses — but that
doesn't yet show whether TOFU could, even in principle, run the *specific* OCD-style
neighbor-collateral-damage comparison that is BioUnlearn-Bench's actual contribution. This
document settles that question directly, from the dataset's own structure, no GPU required.

## Method

Loaded `locuslab/TOFU` configs `forget10` (400 examples) and `retain90` (3600 examples) via
`datasets.load_dataset`. Extracted each example's underlying fictitious author identity two
ways:

1. Split sizes alone: 400 = 20 × 20, 3600 = 180 × 20, and 400 + 3600 = 4000 = 200 × 20 —
   consistent with TOFU's documented design of exactly 200 synthetic author profiles at 20
   QA pairs each (Maini et al. 2024).
2. Direct extraction: regex `full name is ([A-Z][a-zA-Z\-' ]+)\.` applied to every answer
   in both splits, pulling the literal author name out of the subset of questions that ask
   for it directly ("What is the full name of the author born in ... ?").

## Result

- forget10 contains 2 authors identifiable via the full-name-question pattern
  (Hsiao Yun-Hwa, Xin Lee Williams); retain90 contains 16 identifiable the same way.
- **Zero overlap** between the two name sets.
- Combined with the exact 20/180-author arithmetic above, this confirms TOFU's forget/retain
  split partitions *whole authors*, not individual facts within a shared entity — every fact
  about a given fictitious author sits entirely on one side of the split. None of an author's
  facts are held back on the retain side for comparison against what's forgotten about them.

## Why this matters for the paper

For an OCD-style (ontology-collateral-damage) comparison to be possible, a benchmark needs
two things TOFU does not have:

1. **A retained entity relationally close to the forgotten one** (e.g., a pharmacologically
   similar drug, a comorbid condition, a drug-class sibling) — so that an imprecise edit to
   the forgotten fact has somewhere nearby to leak into.
2. **A real, external relational structure connecting entities** (UMLS's drug-class
   hierarchies, ATC codes, ICD-10 comorbidities) — not just textual similarity, but a
   graph humans and clinical systems already treat as connected.

TOFU has neither by construction. Its 200 author profiles are independently generated
fictions (birthplace, genre, awards, book titles) with no shared graph linking one author to
another — the dataset was deliberately built this way so authors *wouldn't* interfere with
each other, which is exactly the opposite property BioUnlearn-Bench needs to test collateral
damage. TOFU's closest analog, `retain_perturbed`, is paraphrases of the *same* fact (tests
robustness to rewording), not a *different* but *related* fact (tests collateral damage) —
these are not the same question.

BioUnlearn-Bench's RGU/IFE scenarios are built the other way around: aspirin's reversed
cardiovascular-guideline fact (RGU) and formulation-equivalence claims (IFE) are checked
against retained facts about pharmacologically neighboring NSAIDs/anticoagulants and
dosage-equivalent formulations — a real UMLS-graph-connected retain set that a sloppy edit
could plausibly damage, which is exactly what OGDA's protected subspace exists to prevent and
what the FA/DEF behavioral eval measures on the untargeted scenario.

## Conclusion

This is a positive finding, not a null result: TOFU structurally cannot serve as a benchmark
for ontology-aware collateral damage, independent of whether anyone ran the experiment on it.
The GA/NPO/RMU cross-dataset runs already completed establish *method*-behavior parity
(collapse vs. non-collapse generalizes across datasets); this analysis establishes that the
*benchmark*-level question BioUnlearn-Bench is built to answer is one TOFU cannot pose at
all, for a structural reason, not an oversight. That is itself evidence for why a
UMLS-graph-grounded benchmark is a necessary contribution rather than a redundant one —
it should be stated in the paper as such, not treated as an unresolved gap.

## Reproduction

```python
from datasets import load_dataset
import re

forget = load_dataset('locuslab/TOFU', 'forget10')['train']
retain = load_dataset('locuslab/TOFU', 'retain90')['train']
pat = re.compile(r"full name is ([A-Z][a-zA-Z\-' ]+)\.")

def get_author_names(ds):
    return {m.group(1).strip() for ex in ds if (m := pat.search(ex['answer']))}

fn, rn = get_author_names(forget), get_author_names(retain)
print(len(forget), len(retain))          # 400 3600
print(len(fn), len(rn), fn & rn)         # 2 16 set()
```
