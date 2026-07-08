"""
Stage A — Merge UMLS graph + RxNorm enrichment into one unified JSON.
This is the final Stage A output consumed by Stage B instance generation.

Output format per concept:
{
  "cui":         "C0004057",
  "name":        "aspirin",
  "scenario":    "RGU",
  "note":        "...",
  "rxcui":       "1191",
  "fda_label":   {...},
  "neighbors": {
      "<id>": {
          "name":     "...",
          "relation": "...",
          "source":   "umls" | "rxnorm",
          "hop":      1 | 2,
          "weight":   0.0–1.0
      }, ...
  }
}
"""

import json
import os
import glob

GRAPH_DIR = "data/concept_graphs"
OUT_DIR   = "data/concept_graphs/merged"

def load_json(path):
    with open(path) as f:
        return json.load(f)

def merge(umls_path, enriched_path):
    umls     = load_json(umls_path)
    enriched = load_json(enriched_path)

    merged = {
        "cui":         umls["cui"],
        "name":        umls["name"],
        "search_term": umls.get("search_term", ""),
        "scenario":    umls.get("scenario", enriched.get("scenario", "")),
        "note":        umls.get("note", ""),
        "rxcui":       enriched.get("rxcui", None),
        "fda_label":   enriched.get("fda_label", {}),
        "drug_classes":enriched.get("drug_classes", []),
        "neighbors":   {}
    }

    # Add UMLS neighbors first (tag source)
    for cui, info in umls.get("neighbors", {}).items():
        merged["neighbors"][cui] = dict(info, source="umls")

    # Add RxNorm neighbors (skip duplicates)
    for cls_id, info in enriched.get("neighbors", {}).items():
        if cls_id not in merged["neighbors"]:
            merged["neighbors"][cls_id] = dict(info, source="rxnorm")

    return merged

def save_merged(data):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{data['cui']}_merged.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path

# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":

    # Find all pairs
    umls_files     = glob.glob(f"{GRAPH_DIR}/C*.json")
    umls_files     = [f for f in umls_files if "_enriched" not in f
                      and "_merged" not in f]

    print("BioUnlearn — Stage A: Merging UMLS + RxNorm graphs")
    print(f"Found {len(umls_files)} UMLS graphs\n")

    results = []
    for umls_path in sorted(umls_files):
        cui          = os.path.basename(umls_path).replace(".json","")
        enriched_path = os.path.join(GRAPH_DIR, f"{cui}_enriched.json")

        if not os.path.exists(enriched_path):
            print(f"  [SKIP] No enriched file for {cui}")
            continue

        merged = merge(umls_path, enriched_path)
        path   = save_merged(merged)
        n      = len(merged["neighbors"])
        print(f"  {cui}  {merged['name'][:35]:<35}  "
              f"neighbors={n}  scenario={merged['scenario']}")
        print(f"    → {path}")
        results.append(merged)

    print(f"\n{'='*65}")
    print(f"STAGE A FINAL — Merged Graph Summary")
    print(f"{'='*65}")
    print(f"{'CUI':<12} {'Name':<30} {'N':<5} {'Scenario':<8} {'FDA?'}")
    print(f"{'-'*65}")
    for m in results:
        has_fda = "yes" if m["fda_label"].get("indications") else "no"
        print(f"{m['cui']:<12} {m['name'][:29]:<30} "
              f"{len(m['neighbors']):<5} {m['scenario']:<8} {has_fda}")
    print(f"{'='*65}")
    print(f"\nMerged JSONs → {OUT_DIR}/")
    print(f"Stage A complete. These files are the input to Stage B.")
