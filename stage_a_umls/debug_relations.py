import requests, sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import UMLS_API_KEY, UMLS_BASE, UMLS_LOGIN, UMLS_SERVICE

def get_tgt():
    r = requests.post(UMLS_LOGIN, data={"apikey": UMLS_API_KEY})
    import re
    loc = r.headers.get("location") or re.search(r'action="(https://[^"]+)"', r.text).group(1)
    return loc

def get_st(tgt):
    return requests.post(tgt, data={"service": UMLS_SERVICE}).text.strip()

def search_cui(term, tgt, sabs=None):
    params = {"string": term, "ticket": get_st(tgt), "returnIdType": "concept", "pageSize": 5}
    if sabs:
        params["sabs"] = sabs
    r = requests.get(f"{UMLS_BASE}/search/current", params=params)
    results = r.json()["result"]["results"]
    print(f"\n  Search results for '{term}' (sabs={sabs}):")
    for res in results[:5]:
        print(f"    {res['ui']}  {res['name']}")
    return results[0]["ui"] if results and results[0]["ui"] != "NONE" else None

def get_all_relations(cui, tgt):
    st = get_st(tgt)
    r = requests.get(f"{UMLS_BASE}/content/current/CUI/{cui}/relations",
                     params={"ticket": st, "pageSize": 50})
    results = r.json().get("result", [])
    print(f"\n  ALL relation labels for {cui}:")
    labels = set()
    for rel in results:
        label = rel.get("relationLabel","")
        name  = rel.get("relatedIdName","")
        labels.add(label)
        print(f"    [{label}]  {name}")
    print(f"\n  Unique labels: {labels}")

tgt = get_tgt()

# Check search with and without sabs filter
for term in ["Aspirin", "Rosiglitazone", "Vioxx"]:
    search_cui(term, tgt, sabs=None)
    time.sleep(0.3)
    search_cui(term, tgt, sabs="MSH,SNOMEDCT_US")
    time.sleep(0.3)

# Check what relations Aspirin actually has
print("\n" + "="*55)
print("RELATIONS for C0004057 (Aspirin):")
get_all_relations("C0004057", tgt)
time.sleep(0.3)
print("\nRELATIONS for C0289313 (Rosiglitazone):")
get_all_relations("C0289313", tgt)
