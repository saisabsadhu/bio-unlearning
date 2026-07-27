"""
Extracts SNOMED CT inactivated concepts with a machine-readable reason
code, cross-referenced with their historical-association replacement
concept -- real, structured (A_old -> A_new) pairs straight from the
authoritative RF2 release files, zero LLM involvement in the ground truth
itself. Per MASTER_RESEARCH_PLAN.md Section 4.3: reason=OUTDATED maps to
RGU-style instances, reason=ERRONEOUS maps to IFE-style instances.

Requires the SNOMED CT US Edition RF2 Snapshot files (downloaded via the
UMLS Terminology Services download API, see documentation -- not
committed to the repo, too large; re-download with
stage_a_umls/snomed_download.py if needed).

Usage:
    python stage_a_umls/snomed_reason_extraction.py
"""

import argparse
import csv
import json
import os
import sys

csv.field_size_limit(sys.maxsize)

REASON_CODES = {
    "900000000000483008": ("OUTDATED", "RGU"),
    "900000000000485001": ("ERRONEOUS", "IFE"),
}

ASSOCIATION_REFSETS = {
    "900000000000526001": "REPLACED_BY",
    "900000000000527005": "SAME_AS",
    "900000000000523009": "POSSIBLY_EQUIVALENT_TO",
    "900000000000528000": "WAS_A",
    "900000000000530003": "ALTERNATIVE",
}

# FSN semantic tags that indicate clinically-actionable concepts worth
# keeping (drugs, procedures, findings/disorders) -- filters out purely
# administrative/qualifier-value/staging-scale noise
RELEVANT_TAGS = ["(product)", "(substance)", "(procedure)", "(disorder)",
                  "(finding)", "(regime/therapy)", "(clinical drug)"]


def load_descriptions(path):
    """conceptId -> FSN term (active, type=FSN only)."""
    fsn_by_concept = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            _id, _et, active, _mod, concept_id, _lang, type_id, term, _case = row
            if active == "1" and type_id == "900000000000003001":  # FSN
                fsn_by_concept[concept_id] = term
    return fsn_by_concept


def load_attribute_values(path):
    """referencedComponentId -> reason code (only our 2 target reasons, active rows)."""
    reasons = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            _id, _et, active, _mod, _refset, ref_component, value_id = row
            if active == "1" and value_id in REASON_CODES:
                reasons[ref_component] = REASON_CODES[value_id]
    return reasons


def load_associations(path):
    """referencedComponentId -> list of (assoc_type, target_component_id), active rows."""
    assoc = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            _id, _et, active, _mod, refset, ref_component, target = row
            if active == "1" and refset in ASSOCIATION_REFSETS:
                assoc.setdefault(ref_component, []).append((ASSOCIATION_REFSETS[refset], target))
    return assoc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf2-dir", required=True, help="path to the extracted Snapshot dir")
    parser.add_argument("--out", default="data/gold_sources/snomed_reason_coded.json")
    args = parser.parse_args()

    desc_path = None
    attr_path = None
    assoc_path = None
    for root, _, files in os.walk(args.rf2_dir):
        for fn in files:
            if fn.startswith("sct2_Description_Snapshot"):
                desc_path = os.path.join(root, fn)
            elif fn.startswith("der2_cRefset_AttributeValueSnapshot"):
                attr_path = os.path.join(root, fn)
            elif fn.startswith("der2_cRefset_AssociationSnapshot"):
                assoc_path = os.path.join(root, fn)
    assert desc_path and attr_path and assoc_path, "missing one or more RF2 files"

    print("Loading descriptions (FSNs)...")
    fsn_by_concept = load_descriptions(desc_path)
    print(f"  {len(fsn_by_concept)} active FSNs")

    print("Loading inactivation reasons...")
    reasons = load_attribute_values(attr_path)
    print(f"  {len(reasons)} concepts with OUTDATED/ERRONEOUS reason")

    print("Loading historical associations...")
    assoc = load_associations(assoc_path)
    print(f"  {len(assoc)} concepts with an association")

    results = []
    for concept_id, (reason, scenario) in reasons.items():
        old_fsn = fsn_by_concept.get(concept_id)
        if not old_fsn:
            continue
        if not any(tag in old_fsn for tag in RELEVANT_TAGS):
            continue
        assocs = assoc.get(concept_id, [])
        if not assocs:
            continue  # need a real replacement pair
        assoc_type, target_id = assocs[0]
        new_fsn = fsn_by_concept.get(target_id)
        if not new_fsn:
            continue
        results.append({
            "old_concept_id": concept_id,
            "old_fsn": old_fsn,
            "reason": reason,
            "scenario": scenario,
            "association_type": assoc_type,
            "new_concept_id": target_id,
            "new_fsn": new_fsn,
            "ground_truth_source": "SNOMED CT US Edition RF2, Component Inactivation + Historical Association reference sets",
            "ground_truth_tier": "silver_snomed_reason_coded",
        })

    print(f"\n{len(results)} clinically-relevant reason-coded pairs found")
    by_scenario = {}
    for r in results:
        by_scenario.setdefault(r["scenario"], 0)
        by_scenario[r["scenario"]] += 1
    print("By scenario:", by_scenario)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
