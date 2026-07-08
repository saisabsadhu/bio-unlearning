"""
Stage A — UMLS Graph Traversal
Uses additionalRelationLabel to get clinical neighbors (not drug products).
Produces per-concept JSON: {cui, name, neighbors, weights}
"""

import requests
import json
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    UMLS_API_KEY, UMLS_BASE, UMLS_LOGIN,
    UMLS_SERVICE, HOP_DEPTH, ALPHA, MAX_RETAIN
)

# ── Clinical relation labels we actually want ─────────────────────────
# These come from additionalRelationLabel, not relationLabel
CLINICAL_RELATIONS = {
    "contraindicated_with_disease",
    "has_mechanism_of_action",
    "has_physiologic_effect",
    "has_structural_class",
    "may_treat",
    "may_prevent",
    "has_contraindication",
    "disease_has_associated_gene",
    "has_causative_agent",
    "has_associated_morphology",
    "isa",                        # CHD — concept hierarchy (keep for drug classes)
    "inverse_isa",                # PAR — broader concept
    "has_parent",
}

# Noise labels to explicitly skip regardless
SKIP_RELATIONS = {
    "basis_of_strength_substance_of",
    "active_ingredient_of",
    "active_moiety_of",
    "contained_in",
    "ingredient_of",
    "part_of",
    "concept_in_subset",
    "has_tradename",
    "precise_active_ingredient_of",
    "direct_substance_of",
}

# ── Auth ──────────────────────────────────────────────────────────────

def get_tgt():
    r = requests.post(UMLS_LOGIN, data={"apikey": UMLS_API_KEY})
    import re
    loc = r.headers.get("location") or re.search(
        r'action="(https://[^"]+)"', r.text).group(1)
    return loc

def get_st(tgt_url):
    r = requests.post(tgt_url, data={"service": UMLS_SERVICE})
    return r.text.strip()

# ── Core API calls ────────────────────────────────────────────────────

def search_cui(term, tgt_url):
    st = get_st(tgt_url)
    r = requests.get(
        f"{UMLS_BASE}/search/current",
        params={"string": term, "ticket": st,
                "returnIdType": "concept", "pageSize": 1}
    )
    r.raise_for_status()
    results = r.json()["result"]["results"]
    if not results or results[0]["ui"] == "NONE":
        return None, None
    return results[0]["ui"], results[0]["name"]

def get_clinical_relations(cui, tgt_url):
    """
    Fetch relations for a CUI and keep only clinically meaningful ones
    by filtering on additionalRelationLabel.
    Returns list of {cui, name, relation, add_relation}
    """
    st = get_st(tgt_url)
    r = requests.get(
        f"{UMLS_BASE}/content/current/CUI/{cui}/relations",
        params={"ticket": st, "pageSize": 100}
    )
    if r.status_code == 404:
        return []
    r.raise_for_status()

    results   = r.json().get("result", [])
    relations = []

    for rel in results:
        add_label = rel.get("additionalRelationLabel", "")
        rel_label = rel.get("relationLabel", "")
        rel_name  = rel.get("relatedIdName", "")
        rel_cui   = rel.get("relatedId", "")

        # Extract CUI from URL if needed
        if "/" in rel_cui:
            rel_cui = rel_cui.rstrip("/").split("/")[-1]

        # Skip non-CUI entries
        if not rel_cui.startswith("C"):
            continue

        # Skip explicit noise
        if add_label in SKIP_RELATIONS:
            continue

        # Keep if add_label is clinically meaningful
        if add_label in CLINICAL_RELATIONS:
            relations.append({
                "cui":          rel_cui,
                "name":         rel_name,
                "rel_label":    rel_label,
                "add_label":    add_label,
            })

    return relations

def has_icd_code(cui, tgt_url):
    st = get_st(tgt_url)
    r = requests.get(
        f"{UMLS_BASE}/content/current/CUI/{cui}/atoms",
        params={"ticket": st, "sabs": "ICD10CM", "pageSize": 1}
    )
    try:
        return len(r.json().get("result", [])) > 0
    except Exception:
        return False

# ── Weight formula ────────────────────────────────────────────────────

def retention_weight(dist, icd_priority):
    return round(ALPHA / (dist + 1) + (1 - ALPHA) * icd_priority, 4)

# ── Main traversal ────────────────────────────────────────────────────

def build_concept_graph(concept_term, tgt_url, hop_depth=HOP_DEPTH):
    print(f"\n{'='*60}")
    print(f"  Concept : {concept_term}")
    print(f"{'='*60}")

    cui, name = search_cui(concept_term, tgt_url)
    if not cui:
        print(f"  [SKIP] Could not resolve: {concept_term}")
        return None

    print(f"  CUI     : {cui}")
    print(f"  Name    : {name}")

    graph = {
        "cui":         cui,
        "name":        name,
        "search_term": concept_term,
        "neighbors":   {}
    }

    frontier = {cui}
    visited  = {cui}

    for hop in range(1, hop_depth + 1):
        next_frontier = set()
        print(f"\n  Hop {hop} — exploring {len(frontier)} concept(s)...")

        for parent_cui in frontier:
            time.sleep(0.2)
            try:
                rels = get_clinical_relations(parent_cui, tgt_url)
            except Exception as e:
                print(f"    [WARN] {parent_cui}: {e}")
                continue

            for rel in rels:
                child_cui = rel["cui"]
                if child_cui in visited:
                    continue
                visited.add(child_cui)
                next_frontier.add(child_cui)

                time.sleep(0.1)
                icd          = has_icd_code(child_cui, tgt_url)
                icd_priority = 1.0 if icd else 0.5
                w            = retention_weight(hop, icd_priority)

                graph["neighbors"][child_cui] = {
                    "name":         rel["name"],
                    "relation":     rel["add_label"],
                    "hop":          hop,
                    "icd_priority": icd_priority,
                    "weight":       w,
                }
                print(f"    + {child_cui}  [{rel['add_label']}]  {rel['name'][:45]}")

        print(f"  → {len(next_frontier)} new clinical neighbors at hop {hop}")
        frontier = next_frontier
        if not frontier:
            break

    # Summary
    total    = len(graph["neighbors"])
    sorted_n = sorted(graph["neighbors"].items(),
                      key=lambda x: x[1]["weight"], reverse=True)
    print(f"\n  Total clinical neighbors : {total}")
    print(f"  Top 5 by weight:")
    for c, info in sorted_n[:5]:
        print(f"    {c}  {info['name'][:45]:<45}  "
              f"w={info['weight']}  [{info['relation']}]  hop={info['hop']}")
    return graph

# ── Save ──────────────────────────────────────────────────────────────

def save_graph(graph, out_dir="data/concept_graphs"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{graph['cui']}.json")
    with open(path, "w") as f:
        json.dump(graph, f, indent=2)
    print(f"\n  Saved → {path}")
    return path

# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":

    PILOT_CONCEPTS = [
        ("Aspirin",                     "RGU", "USPSTF 2022 CVD prevention reversal"),
        ("Hormone Replacement Therapy", "RGU", "WHI trial reversal"),
        ("Rosiglitazone",               "IFE", "cardiovascular safety withdrawal"),
        ("Vioxx",                       "IFE", "rofecoxib recall"),
        ("Breast cancer screening",     "PAC", "age/frequency guideline change"),
    ]

    print("BioUnlearn — Stage A: UMLS Clinical Graph Traversal")
    print("Pilot: 5 concepts across 3 scenarios\n")

    tgt = get_tgt()
    print("TGT obtained.\n")

    results = []
    for term, scenario, note in PILOT_CONCEPTS:
        graph = build_concept_graph(term, tgt)
        if graph:
            graph["scenario"] = scenario
            graph["note"]     = note
            path = save_graph(graph)
            results.append((term, graph["cui"],
                            len(graph["neighbors"]), scenario))
        time.sleep(0.5)

    print("\n" + "="*65)
    print("STAGE A COMPLETE — Clinical Graph Summary")
    print("="*65)
    print(f"{'Concept':<35} {'CUI':<12} {'Neighbors':<10} {'Scenario'}")
    print("-"*65)
    for term, cui, n, scenario in results:
        print(f"{term:<35} {cui:<12} {n:<10} {scenario}")
    print("="*65)
    print("JSON files → data/concept_graphs/")
