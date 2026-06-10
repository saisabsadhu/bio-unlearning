import requests

# ── Config ────────────────────────────────────────────────────────────
API_KEY = "b92da1dd-2ffc-4484-8a05-46440e91b58b"
BASE    = "https://uts-ws.nlm.nih.gov/rest"

# ── Auth ──────────────────────────────────────────────────────────────
def get_tgt():
    r = requests.post(
        "https://utslogin.nlm.nih.gov/cas/v1/api-key",
        data={"apikey": API_KEY}
    )
    return r.headers["location"]

def get_st(tgt_url):
    r = requests.post(tgt_url, data={"service": "http://umlsks.nlm.nih.gov"})
    return r.text.strip()

# ── API calls ─────────────────────────────────────────────────────────
def search_cui(term, tgt_url):
    st = get_st(tgt_url)
    r = requests.get(
        f"{BASE}/search/current",
        params={"string": term, "ticket": st, "returnIdType": "concept"}
    )
    results = r.json()["result"]["results"]
    return results[0] if results else None

def get_relations(cui, tgt_url):
    st = get_st(tgt_url)
    r = requests.get(
        f"{BASE}/content/current/CUI/{cui}/relations",
        params={"ticket": st}
    )
    return r.json().get("result", [])

# ── Test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Getting TGT...")
    tgt = get_tgt()
    print(f"TGT obtained.")

    print("\nSearching for 'Aspirin'...")
    result = search_cui("Aspirin", tgt)
    if result:
        print(f"  CUI  : {result['ui']}")
        print(f"  Name : {result['name']}")
        CUI = result['ui']
    else:
        print("  No result — check API key")
        exit(1)

    print(f"\nFetching relations for {CUI}...")
    relations = get_relations(CUI, tgt)
    print(f"  Found {len(relations)} relations")
    for rel in relations[:5]:
        print(f"  → {rel.get('relatedIdName','?')}  [{rel.get('relationLabel','?')}]")

    print("\n✓ UMLS API working.")