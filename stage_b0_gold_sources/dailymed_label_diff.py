"""
Stage B0 gold-source harvester: real, dated, zero-LLM-involvement drug label
diffs via DailyMed's SPL version-history API.

This is the "DailyMed history-API puller" flagged as not-yet-built in
stage_b0_gold_sources/README.md's next-steps -- directly answers Dr. Vindo's
original concern (documentation/groundtruth_sources_dr_vindo_directions.md)
by pulling literal before/after label text from FDA-regulated official
documents, not an LLM-invented or LLM-summarized reversal.

How it works:
1. /spls.json?drug_name=X -> find a SETID with multiple SPL versions.
2. /spls/{SETID}/history.json -> list all version numbers + publish dates.
3. getFile.cfm?type=zip&setid=X&version=N -> download that version's full SPL
   (ZIP containing the XML label text).
4. Extract a named section (by LOINC-derived displayName, e.g. "BOXED WARNING")
   from two versions and diff them.

Usage:
    python stage_b0_gold_sources/dailymed_label_diff.py --drug Avandamet --section "BOXED WARNING"
"""

import argparse
import os
import re
import xml.etree.ElementTree as ET
import zipfile

import requests

RAW_DIR = "stage_b0_gold_sources/raw/dailymed"
BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"


def find_setid(drug_name):
    r = requests.get(f"{BASE}/spls.json", params={"drug_name": drug_name, "pagesize": 50})
    r.raise_for_status()
    candidates = r.json().get("data", [])
    # prefer the SETID with the most versions (spl_version is the latest version number)
    candidates.sort(key=lambda c: c.get("spl_version", 0), reverse=True)
    return candidates


def get_history(setid):
    r = requests.get(f"{BASE}/spls/{setid}/history.json")
    r.raise_for_status()
    return r.json()["data"]["history"]


def download_version(setid, version, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, f"v{version}.zip")
    if not os.path.exists(zip_path):
        r = requests.get(
            "https://dailymed.nlm.nih.gov/dailymed/getFile.cfm",
            params={"type": "zip", "setid": setid, "version": version},
        )
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)
    extract_dir = os.path.join(out_dir, f"v{version}")
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    for fn in os.listdir(extract_dir):
        if fn.endswith(".xml"):
            return os.path.join(extract_dir, fn)
    return None


def extract_section_by_code(xml_path, code_substring):
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()
    content_clean = re.sub(r'xmlns="[^"]+"', "", content, count=1)
    root = ET.fromstring(content_clean)
    for section in root.iter("section"):
        code_el = section.find("code")
        code_name = code_el.get("displayName") if code_el is not None else ""
        if code_substring.lower() in (code_name or "").lower():
            text_el = section.find("text")
            text = " ".join(text_el.itertext()).strip() if text_el is not None else ""
            return re.sub(r"\s+", " ", text)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", required=True)
    parser.add_argument("--section", default="BOXED WARNING")
    args = parser.parse_args()

    candidates = find_setid(args.drug)
    if not candidates:
        print(f"No SETID found for '{args.drug}'")
        return
    setid = candidates[0]["setid"]
    print(f"Using SETID {setid} ({candidates[0]['title'][:80]})")

    history = get_history(setid)
    versions = sorted(h["spl_version"] for h in history)
    if len(versions) < 2:
        print(f"Only {len(versions)} version(s) available -- no diff possible.")
        return
    print(f"Versions available: {versions}")

    out_dir = os.path.join(RAW_DIR, args.drug.lower().replace(" ", "_"))
    early_xml = download_version(setid, versions[0], out_dir)
    late_xml = download_version(setid, versions[-1], out_dir)

    early_date = next(h["published_date"] for h in history if h["spl_version"] == versions[0])
    late_date = next(h["published_date"] for h in history if h["spl_version"] == versions[-1])

    early_text = extract_section_by_code(early_xml, args.section)
    late_text = extract_section_by_code(late_xml, args.section)

    print(f"\n=== {args.section} -- version {versions[0]} ({early_date}) ===")
    print(early_text[:1200] if early_text else "NOT FOUND")
    print(f"\n=== {args.section} -- version {versions[-1]} ({late_date}) ===")
    print(late_text[:1200] if late_text else "NOT FOUND")


if __name__ == "__main__":
    main()
