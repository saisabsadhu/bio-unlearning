# Research Memo: Concretizing Forget/Retain Ground Truth (Dr. Vindo, UMich collaboration discussion)

Context: Dr. Vindo (UMich) proposed shifting toward (A) a more concretized, documented ground truth for "what to forget" — analogous to UMLS changes over time — while noting most such changes aren't mistakes and often aren't documented anywhere; and (B) building the unlearning dataset directly from EHR data or clinical guidelines rather than (or in addition to) an LLM annotation oracle. This memo researches what actually exists for each direction, so we can decide what to bring back to him as a concrete proposal.

## TL;DR

Dr. Vindo's intuition is half right and half already solved:

- He's right that most UMLS/vocabulary changes are **not clinically meaningful reversals** — they're mostly terminology housekeeping (CUI merges, source-vocabulary version bumps). Using raw UMLS diffs as forget targets would mostly generate noise.
- He's wrong that changes are "not documented anywhere" for the *clinically meaningful* subset — they are documented, just not inside UMLS. **SNOMED CT** (a UMLS source vocabulary you already have access to) has an explicit, reason-coded inactivation mechanism that separates "erroneous" from "outdated" from "duplicate." And outside ontologies entirely, there are **peer-reviewed, citation-backed catalogs of real medical reversals** (Prasad et al. 2013; Herrera-Perez et al. 2019) and **official regulatory change databases** (FDA SrLC) that are far stronger ground truth than anything UMLS offers.
- On direction (B): "go to EHR" most plausibly does NOT mean mining raw EHR audit logs (that literature is about clinician workflow behavior, not fact-level ground truth). It more likely converges with (A): anchor forget targets in **documented guideline/regulatory reversals**, then use MIMIC-IV as the *grounding corpus* (as you already planned), and optionally validate the reversal's real-world adoption using published EHR utilization studies (e.g., rosiglitazone prescribing dropped 70% within 2 years of the 2007 FDA warning — this is already one of your pilot concepts, C0289313).

## Direction A: Documented, non-LLM-invented ground truth for temporal/reversal forgetting

### 1. UMLS's own history mechanism (already in your stack, but weaker than it looks)

UMLS ships explicit diff files between releases:
- `MRCUI.RRF` — tracks the fate of every CUI from 1991 to present (merged, split, retired).
- `MRCONSO_HISTORY.txt` — all atoms/concepts/codes dropped, since 2004AA.
- `MRREL_HISTORY.txt` — all relations dropped from MRREL, since 2004AA.
- `MRSAB_HISTORY.txt` — every version of every updated source vocabulary.

**Caveat (confirms Dr. Vindo's concern):** these files record *that* something changed, not *why*. The dominant cause of UMLS-level change is terminology curation (duplicate CUI merges when NLM discovers two CUIs name the same concept, source vocabulary version updates) rather than "this clinical fact became false." Diffing MRREL/MRCONSO directly would produce a forget set that's mostly vocabulary noise, exactly as he suspected.

### 2. SNOMED CT reason-coded inactivation — the actual fix for his concern

SNOMED CT (already one of your UMLS source vocabularies) inactivates concepts through **Component Inactivation Reference Sets**, and every inactivation carries an explicit **reason code**, including (per SNOMED International's release format spec):
- `DUPLICATE` / `AMBIGUOUS` — housekeeping, not clinically meaningful
- `ERRONEOUS` — the concept was simply wrong (closest thing to a labeled "mistake")
- `OUTDATED` — the concept was correct once, superseded by newer understanding (this is your RGU/temporal-directionality target, *pre-labeled*)
- `MOVED_ELSEWHERE` / `LIMITED` — scope changes, not reversals

This gives you something UMLS's raw history files don't: **a machine-readable filter that separates "mistake," "legitimate evolution," and "pure housekeeping" before you ever touch an LLM.** Concretely: filter SNOMED CT historical association reference sets to `reason = OUTDATED` for RGU-scenario forget targets, and `reason = ERRONEOUS` for IFE-scenario forget targets. Each inactivated concept also links to its replacement concept via the historical association, which gives you a real, ontology-sourced (A_old → A_new) pair for free — no oracle needed to invent the pairing.

Practical note: this needs a UMLS Metathesaurus license-holder pull of SNOMED CT RF2 release files (Component History Association Reference Set), which you already have access to via your NLM UTS account.

### 3. RxNorm — usable but thinner reason-coding than SNOMED

RxNorm's `RXNCUICHANGES.RRF` / `RXNATOMARCHIVE.RRF` and the `/historystatus` API endpoint track deprecated/obsolete drug concepts and NDC codes, with a status field (Active/Obsolete) but not the rich reason taxonomy SNOMED has. Useful for confirming drug-concept deprecation dates to cross-check against FDA action dates (below), less useful as a primary reason-labeled source on its own.

### 4. PrimeKG-CL — a very recent, directly adjacent existing benchmark (worth checking before you build)

Found a 2026 benchmark, **PrimeKG-CL** ("A Continual Graph Learning Benchmark on Evolving Biomedical Knowledge Graphs," Radwan, Li et al., arXiv:2605.10529), built on Harvard's PrimeKG (129K nodes, 4M+ edges across 20 integrated biomedical resources — DrugBank, DisGeNET, Reactome, UMLS, SIDER, etc.). It snapshots PrimeKG at two real time points (June 2021 vs. July 2023) and reports **889K removed edges, 5.83M added edges, 7.21M persistent edges** across the two snapshots from nine asynchronously-updating source databases. It's openly released (HuggingFace + GitHub).

Two implications for you:
- **Opportunity**: this could hand you real, non-synthetic (edge existed → edge removed) ground truth across a much larger concept space than you could hand-curate, and it's already reason-agnostic-diffed for you at the graph level — you'd still need to apply your own filter for "clinically meaningful reversal" vs. "database housekeeping," but the raw diff computation is already done.
- **Risk**: it occupies adjacent territory (biomedical KG temporal evolution) to your Failure Mode 1/2 framing. You should read it closely before finalizing scope, both to differentiate BioUnlearn from it explicitly (it's continual *learning*, not *unlearning* — different problem) and to make sure you're not duplicating its data engineering.

### 5. Real, peer-reviewed catalogs of actual medical reversals (strongest option for RGU)

This is probably the single best answer to "concretized, documented ground truth that isn't LLM-invented":

- **Prasad, Cifu et al., "A Decade of Reversal: An Analysis of 146 Contradicted Medical Practices," Mayo Clinic Proceedings, 2013** — 146 named clinical practices that were standard of care, then formally contradicted by RCT evidence, each with the contradicting trial citation.
- **Herrera-Perez et al., "A comprehensive review of randomized clinical trials in three medical journals reveals 396 medical reversals," eLife, 2019** — 396 more, from JAMA/Lancet/NEJM 2003–2017, same structure (practice, contradicting trial, journal, year).
- Together: ~500+ real, dated, citation-backed (Q: was practice X standard? A_old: yes, recommended → A_new: no, contradicted by trial Y in year Z) instances, peer-reviewed and already published — i.e., ground truth that exists independent of anything you construct.

This directly replaces the weakest link in the current plan (Claude-3.5-Sonnet inventing RGU triples) with real literature. It also gives you a built-in, citable defense against the "silver labels are unreliable" reviewer objection, since these instances were validated by domain experts years before your paper existed.

## Direction B: "Go to EHR or guidelines" — what this most plausibly means, and what's available

I don't think raw **EHR audit-log mining** (clinician click/workflow behavior) is the right read of this — that literature (EHR audit trail logs, adherence-drift detection) is about *provider behavior*, not documented factual ground truth, and doesn't give you a clean forget/retain fact pair. The stronger reading is: **anchor forget targets in official guideline/regulatory change records**, several of which are much better-documented than UMLS:

- **FDA Drug Safety-related Labeling Changes (SrLC) database** — official, structured, since Jan 2016, tracks exactly which label section changed (Boxed Warning, Contraindications, Warnings and Precautions, Drug Interactions, Use in Specific Populations) and when, for every approved drug. Combine with **Drugs@FDA / DailyMed historical label PDFs** (both hold prior label versions) to get literal before/after label text — real (A_old, A_new) pairs with exact dates, zero LLM involvement needed for the ground truth itself (LLM only needed to phrase it as a clinical Q&A instance).
- **USPSTF recommendation history** — no single clean bulk-downloadable archive exists, but each recommendation topic page retains links to prior recommendation statements with grade and date; would need light scraping/curation per topic, not automatable at scale without manual verification.
- **Cochrane systematic review updates** — conclusion changes are rare (only ~4-9% of updates flip conclusion) but when they do, they're exceptionally well documented with an explicit "What's New" changelog. Small in count but very high quality; good for a "gold" tier subset.
- **NICE guideline surveillance reports** (UK NICE) — every guideline surveillance decision is published with explicit reasoning for whether evidence triggered an update. Useful if you want geographic/regulatory diversity beyond US sources, lower priority given your MIMIC/USPSTF-centric US design.
- **Choosing Wisely** (ABIM Foundation, 700+ specialty-society recommendations) — better fit for IFE (persistently overused/low-value practices) than RGU (it's about ongoing overuse, not a practice that used-to-be-recommended-then-wasn't), but well documented and citation-backed.

### A genuinely EHR-native idea worth proposing back to Dr. Vindo

Given his likely informatics angle, the strongest "go to EHR" idea isn't mining audit logs — it's using **MIMIC-IV's real timestamps as a natural experiment**. You already know exact guideline/regulatory change dates (e.g., rosiglitazone FDA warning: May 21, 2007 — already published EHR-utilization studies show a documented 70% prescribing drop within two years). You could bucket MIMIC-IV notes/orders by admission date relative to a known change-point date and use the *actual documented clinical practice shift in the data itself* as empirical confirmation that a concept is a genuine, adopted reversal — not just a paper claim. This turns "was this guideline change real and adopted" into an empirically checkable question against your own corpus, which is a stronger and more EHR-native contribution than static QA-pair construction. Notably, rosiglitazone (`C0289313`) is already in your pilot CLMI prescreen set — good sign this idea generalizes to concepts you've already selected.

## Recommended synthesis (what I'd bring back to Dr. Vindo)

1. Replace/augment the RGU scenario's ground truth: use the Prasad (2013) + Herrera-Perez (2019) reversal catalogs (~500 instances) as the primary, citation-backed forget set, supplemented by FDA SrLC label-change pairs for pharmacological reversals — both real, dated, peer-reviewed/regulatory sources, not LLM-invented.
2. Use SNOMED CT's reason-coded inactivation reference sets (`OUTDATED` vs. `ERRONEOUS` vs. `DUPLICATE`) as the mechanism for cleanly separating genuine clinical reversals from vocabulary housekeeping — this directly answers his "most changes aren't mistakes" concern with an existing, machine-readable label rather than manual judgment.
3. Add an EHR-native validation layer: confirm a subset of forget targets against published real-world adoption-decline studies (like the rosiglitazone literature), or directly against MIMIC-IV's own temporal distribution, as evidence the "reversal" was real and clinically adopted, not just committee text.
4. Do a close read of PrimeKG-CL before finalizing scope — both as a potential engineering shortcut (its edge-diff computation across 9 databases is already done and public) and as related work you need to explicitly differentiate from (continual learning vs. unlearning).

This set of changes would also directly strengthen the Nature Communications fit noted in the companion critique (`critique_nature_communications_fit.md`): real, physician-validated, peer-reviewed reversal catalogs and official FDA records are exactly the kind of external, independently-verified ground truth an NC reviewer would trust over an LLM-oracle-authored benchmark.

## Addendum: WHO and other international/national guideline bodies — ground truth AND a biomedical-specific methodological hook

Follow-up question: what about WHO and comparable guideline bodies (not just US-centric USPSTF/FDA)? Researched this against the project's restated dual goal — a strong benchmark, plus a methodological novelty that's specific to the biomedical paradigm (not portable to TOFU/WMDP-style general-domain unlearning).

### What's actually out there, ranked by how well-documented the "why it changed" is

1. **WHO Model List of Essential Medicines (EML)** — the standout. The Expert Committee on Selection and Use of Essential Medicines meets every two years, and for every addition, amendment, *and rejected/removed* medicine, WHO publishes a full prose rationale in the WHO Technical Report Series plus a per-meeting Executive Summary (efficacy/safety evidence, comparative cost, public-health relevance). This is richer than SNOMED's single-word reason codes — it's an actual expert-committee argument for the change, dated, versioned, and freely available. Best candidate for a "why" field that's more informative than anything else surveyed so far.
2. **WHO living guidelines** (e.g., COVID-19 therapeutics) — formally track recommendation changes against a pre-approved Guideline Review Committee protocol, explicitly tied to GRADE evidence updates (often from living network meta-analyses). Excellent documentation depth, but disease-scope is narrow (built for fast-moving single-disease evidence, mostly COVID so far) — less useful as a broad-coverage source, more useful as a small high-quality subset.
3. **ESC / ACC / AHA cardiology guidelines** — every recommendation carries a **Class of Recommendation (I/IIa/IIb/III)** and **Level of Evidence (A/B/C)**, and there's already a published academic literature auditing how these grades shift release-to-release (e.g., systematic reviews tracking guideline rigor 2008-2024). Good domain concentration (cardiology has many well-known downgrades — anti-arrhythmics, hormone therapy, certain device indications) and the grading is machine-parseable per-recommendation, not just per-document.
4. **ACIP (CDC vaccine recommendations)** — real, concrete, dated reversals with public meeting minutes and MMWR rationale (e.g., MMRV vaccine preference pulled in 2008 after a febrile-seizure safety signal, reinstated 2009, restricted again for under-4s in 2025). Good ground truth quality, but **flag a real risk**: vaccine recommendations are currently a politically contentious area (ACIP composition and process were themselves disputed in 2025). Building benchmark instances here risks the paper reading as taking a side in an active political controversy rather than a neutral evaluation of model behavior — worth being deliberate about which ACIP examples (if any) are old/settled enough (pre-2020) to be uncontroversial.

### The methodological novelty this unlocks

Across WHO, NICE, Cochrane, and (in adapted form) ACC/AHA/ESC, guideline recommendations carry a **standardized evidence-certainty / strength-of-recommendation grade** (GRADE: High/Moderate/Low/Very-low certainty × Strong/Conditional recommendation; ACC/AHA/ESC: Class I-III × Level A-C). This is a structural property that is *unique to the clinical-guideline domain* — TOFU's fictitious facts, WMDP's hazard facts, and MUSE's memorized passages have no analogous "how confident was the source in this fact" metadata. That asymmetry is exactly the kind of biomedical-paradigm-specific hook that would make a methodological contribution non-portable-by-construction (i.e., a genuine domain contribution, not a generic method tested on medical data):

- **Evidence-Weighted Directional Gradient Projection**: scale the forget/retain gradient balance in DGP by the certainty delta between the old and new recommendation. A Strong/High-certainty recommendation overturned by another Strong/High-certainty trial should be erased with more confidence (less hedging toward retaining residual old-answer probability) than a Conditional/Low-certainty nuance update. Currently DGP treats every (A_old, A_new) pair identically regardless of how epistemically solid either side is — this is a real gap the guideline metadata can fill.
- **Evidence-Weighted OGFR boundary construction**: currently OGFR weights UMLS neighbors purely by hop-distance and ICD priority (`w(c,f) = α/(dist+1) + (1-α)×ICD_priority(c)`). Neighbors connected to the forget concept via a Strong/High-certainty guideline link arguably deserve more retain-protection than ones connected via a Conditional/Low-certainty link — you could add evidence-certainty as a third weighting term.
- **New metric — Evidence-Weighted Erasure Fidelity (EWEF)**: DEF currently treats all RGU instances equally (`FA_old × Acc_new`). An evidence-weighted variant would upweight instances where both the retraction and the replacement are high-certainty (the clearest, least ambiguous reversals) and downweight or separately report instances built on conditional/low-certainty evidence — directly addressing a reviewer question like "how do you know the 'new' answer isn't itself going to be reversed next year?"

This is the strongest answer I can give to "novelty specific to biomedical paradigm": it isn't a new unlearning algorithm dressed up with medical data, it's a method that structurally depends on something only clinical guidelines have (certified, standardized, per-fact evidence-strength grading) — which is a much harder claim for a reviewer to wave off as "this would work on TOFU too."

### Recommendation

Prioritize **WHO EML** (richest documented rationale, broad medicine coverage, cleanly bienniel-versioned) and **ESC/AHA/ACC** (best existing academic cross-version tracking, concentrated domain good for a pilot) as the primary new ground-truth sources, treat **WHO living guidelines** as a small high-confidence bonus subset, and be deliberate/conservative about ACIP given its current political sensitivity. Layer the GRADE-style certainty metadata from all of these on top as the substrate for the evidence-weighted methodological extension above.

## Sources consulted

- [Metathesaurus - UMLS Reference Manual - NCBI Bookshelf](https://www.ncbi.nlm.nih.gov/books/NBK9684/)
- [UMLS Release Notes (NLM Technical Bulletin, various years)](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/notes.html)
- [Ending Medical Reversal (Prasad & Cifu) - Wikipedia](https://en.wikipedia.org/wiki/Ending_Medical_Reversal)
- Prasad V. et al., "A Decade of Reversal: An Analysis of 146 Contradicted Medical Practices," Mayo Clinic Proceedings, 2013
- [Meta-Research: A comprehensive review of randomized clinical trials in three medical journals reveals 396 medical reversals | eLife](https://elifesciences.org/articles/45183)
- [USPSTF Grade Definitions](https://www.uspreventiveservicestaskforce.org/uspstf/about-uspstf/methods-and-processes/grade-definitions)
- [USPSTF A & B Recommendations as of January 2025 (CHLPI)](https://chlpi.org/wp-content/uploads/2025/01/USPSTF-A-B-Recommendations-V3.pdf)
- [Drug Safety-related Labeling Changes (SrLC) Database Overview | FDA](https://www.fda.gov/drugs/drug-safety-and-availability/drug-safety-related-labeling-changes-srlc-database-overview-updates-safety-information-fda-approved)
- [FDALabel: Full-Text Search of Drug Product Labeling | FDA](https://www.fda.gov/science-research/bioinformatics-tools/fdalabel-full-text-search-drug-product-labeling)
- [Retraction of Systematic Reviews and Clinical Practice Guidelines - Peer Review Congress](https://peerreviewcongress.org/abstract/retraction-of-systematic-reviews-and-clinical-practice-guidelines/)
- [Investing in updating: how do conclusions change when Cochrane systematic reviews are updated?](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1274326/)
- [NICE: Guidance surveillance process manual](https://www.nice.org.uk/process/pmg36/chapter/guidance-surveillance-2)
- [Analyzing SNOMED CT's Historical Data: Pitfalls and Possibilities - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5333315/)
- [3.2.6.3.2. Representing Historical Associations - SNOMED Reference Sets Practical Guide](https://confluence.ihtsdotools.org/display/DOCRFSPG/3.2.6.3.2.+Representing+Historical+Associations)
- [Component Inactivation Reference Sets - SNOMED Release File Specification](https://docs.snomed.org/snomed-ct-specifications/snomed-ct-release-file-specification/reference-set-release-file-specification/5.2-reference-set-types/5.2.1-content-reference-sets/5.2.1.3-attribute-value-reference-set/5.2.1.3-attribute-value-reference-set)
- [Approaches to Supporting the Analysis of Historical Medication Datasets with RxNorm - PubMed](https://pubmed.ncbi.nlm.nih.gov/26958241/)
- [RxNorm Technical Documentation - NLM](https://www.nlm.nih.gov/research/umls/rxnorm/docs/techdoc.html)
- PrimeKG-CL: A Continual Graph Learning Benchmark on Evolving Biomedical Knowledge Graphs, arXiv:2605.10529
- [Building a Knowledge Graph to Enable Precision Medicine (PrimeKG) | Scientific Data / Nature](https://www.nature.com/articles/s41597-023-01960-3)
- [GitHub - mims-harvard/PrimeKG](https://github.com/mims-harvard/PrimeKG)
- [Responding to an FDA Warning — Geographic Variation in the Use of Rosiglitazone | NEJM](https://www.nejm.org/doi/full/10.1056/NEJMp1011042)
- [Choosing Wisely: An International Campaign to Combat Overuse - Commonwealth Fund](https://www.commonwealthfund.org/blog/2017/choosing-wisely-international-campaign-combat-overuse)
- [Expert Committee on Selection and Use of Essential Medicines | WHO](https://www.who.int/groups/expert-committee-on-selection-and-use-of-essential-medicines)
- [Essential medicines | WHO fact sheet](https://www.who.int/news-room/fact-sheets/detail/essential-medicines)
- [Frontiers: Medicines not recommended for inclusion in the WHO essential medicines list - a retrospective observational study](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1517020/full)
- [A living WHO guideline on drugs for covid-19 - PubMed](https://pubmed.ncbi.nlm.nih.gov/32887691/)
- [Methods for living guidelines: early guidance based on practical experience - Journal of Clinical Epidemiology](https://www.jclinepi.com/article/S0895-4356(22)00344-4/fulltext)
- [GRADE approach - Wikipedia](https://en.wikipedia.org/wiki/GRADE_approach)
- [The GRADE Evidence to Decision (EtD) framework - PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5975536/)
- [Levels of Evidence Supporting ACC/AHA and ESC Guidelines, 2008-2018 - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6439920/)
- [Rigour of development of ESC, ACC and AHA guidelines over a 12-year period (2013-2024) - PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12445687/)
- [ACIP Recommendations | CDC](https://www.cdc.gov/acip/vaccine-recommendations/index.html)
- [Advisory Committee on Immunization Practices - Wikipedia](https://en.wikipedia.org/wiki/Advisory_Committee_on_Immunization_Practices)
