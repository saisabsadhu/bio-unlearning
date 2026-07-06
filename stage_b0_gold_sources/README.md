# Stage B0: Gold-tier source harvesting

Per MASTER_RESEARCH_PLAN.md Section 5 (Stage B0) and Section 4.3 (source catalog). This is the harvesting stage for Provenance Quality Score (PQS) 3 "Gold" instances -- real, dated, citation-backed reversals independent of any LLM.

## Status (2026-07-07)

- `data/gold_sources/herrera_perez_2019_seed.json`: 20 of 396 reversals from Herrera-Perez et al. 2019 (eLife), extracted from the article's main-text Table 2 (the representative-examples table). **This is a partial seed, not the full 396** -- the complete list lives in the paper's "Supplementary file 2" (a .docx, not directly machine-readable via fetch tools). Getting the full list requires downloading that supplementary docx directly and parsing it (same approach used for the BioUnlearn proposal docx via pandoc).
- One entry (Sertraline/mirtazapine for Alzheimer's depression) has an unverified citation and was downgraded to PQS 1 pending manual check -- **do not treat any entry in this file as final until independently verified against the primary source**, since it was extracted via an automated fetch-and-summarize tool, not read character-by-character from the PDF.
- Prasad et al. 2013 (Mayo Clinic Proceedings, 146 reversals) not yet harvested -- likely paywalled, needs manual access or institutional login.
- FDA openFDA API (`api.fda.gov/drug/label.json`) confirmed reachable and queryable (260K+ label records), but only exposes *current* labels, not historical versions -- computing real before/after label diffs requires DailyMed's history API or Drugs@FDA supplement-level PDFs, not yet built.
- WHO EML Technical Reports, ESC/AHA/ACC version comparisons, Cochrane conclusion-flip reviews: not yet started.

## Next steps (not yet done)

1. Download Herrera-Perez 2019's Supplementary file 2 docx directly and parse with pandoc (same method as the proposal docx) to get the remaining ~376 reversals.
2. Get Prasad 2013's 146 reversals (may need Mayo Clinic Proceedings access).
3. Build a DailyMed history-API puller for boxed-warning/contraindication diffs on a triaged drug list (start with drugs already appearing in the pilot data: rosiglitazone, rofecoxib, etc.)
4. WHO EML Technical Report Series PDFs -- identify the specific biennial reports covering known removals and extract rationale text.
