"""
Stage A (supplement) — RxNorm + OpenFDA clinical relations
"""

import requests
import json
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RXNORM_BASE  = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_BASE = "https://api.fda.gov/drug/label.json"

# ── RxNorm ────────────────────────────────────────────────────────────

def get_rxcui(drug_name):
    r = requests.get(f"{RXNORM_BASE}/rxcui.json",
                     params={"name": drug_name, "allsrc": 0})
    r.raise_for_status()
    cuis = r.json().get("idGroup", {}).get("rxnormId", [])
    return cuis[0] if cuis else None

def get_drug_classes(rxcui):
    """
    Query each relaSource separately — RxNorm does not accept
    comma-separated relaSources in one call.
    """
    classes  = []
    sources  = ["ATC", "MESH", "FDASPL", "EPC", "MOA", "PE"]
    for src in sources:
        try:
            r = requests.get(
                f"{RXNORM_BASE}/rxclass/class/byRxcui.json",
                params={"rxcui": rxcui, "relaSource": src}
            )
            if r.status_code != 200:
                continue
            for entry in (r.json()
                          .get("rxclassDrugInfoList", {})
                          .get("rxclassDrugInfo", [])):
                cls = entry.get("rxclassMinConceptItem", {})
                classes.append({
                    "class_id":   cls.get("classId", ""),
                    "class_name": cls.get("className", ""),
                    "class_type": cls.get("classType", ""),
                    "rela":       entry.get("rela", ""),
                    "source":     src,
                })
            time.sleep(0.15)
        except Exception as e:
            print(f"    [WARN] {src}: {e}")
            continue
    return classes

def get_fda_label(drug_name):
    try:
        r = requests.get(OPENFDA_BASE,
                         params={"search": f'openfda.generic_name:"{drug_name}"',
                                 "limit": 1})
        if r.status_code != 200:
            return {}
        results = r.json().get("results", [])
        if not results:
            return {}
        label = results[0]
        return {
            "indications":       label.get("indications_and_usage",   [""])[0][:500],
            "contraindications": label.get("contraindications",        [""])[0][:500],
            "warnings":          label.get("warnings",                 [""])[0][:300],
            "boxed_warning":     label.get("boxed_warning",            [""])[0][:300],
        }
    except Exception as e:
        print(f"    [WARN] FDA label: {e}")
        return {}

# ── Combined builder ──────────────────────────────────────────────────

def build_clinical_neighbors(drug_name, umls_cui):
    print(f"\n{'='*60}")
    print(f"  Drug    : {drug_name}")
    print(f"  UMLS CUI: {umls_cui}")
    print(f"{'='*60}")

    enrichment = {
        "umls_cui":     umls_cui,
        "drug_name":    drug_name,
        "rxcui":        None,
        "drug_classes": [],
        "fda_label":    {},
        "neighbors":    {}
    }

    # Step 1 — RxCUI
    rxcui = get_rxcui(drug_name)
    if not rxcui:
        print(f"  [WARN] No RxCUI for '{drug_name}'")
        return enrichment
    enrichment["rxcui"] = rxcui
    print(f"  RxCUI   : {rxcui}")
    time.sleep(0.2)

    # Step 2 — Drug classes (all sources)
    print(f"  Fetching drug classes (ATC/MESH/FDASPL/EPC/MOA/PE)...")
    classes = get_drug_classes(rxcui)
    enrichment["drug_classes"] = classes
    print(f"  Found {len(classes)} class entries:")
    for c in classes[:8]:
        print(f"    [{c['source']}|{c['class_type']}] "
              f"{c['class_name']}  rela={c['rela']}")
    time.sleep(0.2)

    # Step 3 — FDA label
    print(f"  Fetching FDA label...")
    fda = get_fda_label(drug_name)
    enrichment["fda_label"] = fda
    if fda.get("indications"):
        print(f"  Indications  : {fda['indications'][:120]}...")
    if fda.get("contraindications"):
        print(f"  Contraindic. : {fda['contraindications'][:120]}...")
    if fda.get("boxed_warning"):
        print(f"  Boxed warning: {fda['boxed_warning'][:120]}...")

    # Step 4 — Convert to neighbor format
    for cls in classes:
        class_id = cls["class_id"]
        if class_id and class_id not in enrichment["neighbors"]:
            enrichment["neighbors"][class_id] = {
                "name":       cls["class_name"],
                "relation":   cls["rela"],
                "class_type": cls["class_type"],
                "source":     cls["source"],
                "hop":        1,
                "weight":     0.6 if cls["source"] in ("MOA","EPC") else 0.5,
            }

    print(f"\n  Total neighbors: {len(enrichment['neighbors'])}")
    return enrichment

def save_enrichment(data, out_dir="data/concept_graphs"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{data['umls_cui']}_enriched.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")
    return path

# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":

    PILOT_DRUGS = [
        ("Aspirin",            "C0004057", "RGU"),
        ("Rosiglitazone",      "C0289313", "IFE"),
        ("Rofecoxib",          "C0876768", "IFE"),
        ("Medroxyprogesterone","C0282402", "RGU"),
    ]

    print("BioUnlearn — Stage A: RxNorm + OpenFDA Enrichment")
    print("Clinical neighbors via drug class, MOA, EPC\n")

    results = []
    for drug, cui, scenario in PILOT_DRUGS:
        data = build_clinical_neighbors(drug, cui)
        data["scenario"] = scenario
        save_enrichment(data)
        results.append((drug, cui, data["rxcui"],
                        len(data["neighbors"]), scenario))
        time.sleep(0.5)

    print("\n" + "="*70)
    print("RxNorm Enrichment — Summary")
    print("="*70)
    print(f"{'Drug':<25} {'UMLS CUI':<12} {'RxCUI':<10} "
          f"{'Neighbors':<10} {'Scenario'}")
    print("-"*70)
    for drug, cui, rxcui, n, scenario in results:
        print(f"{drug:<25} {cui:<12} {str(rxcui):<10} {n:<10} {scenario}")
    print("="*70)
