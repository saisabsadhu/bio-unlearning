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

tgt = get_tgt()

# Try 1 — SNOMED relations for Aspirin
print("=== SNOMED relations for C0004057 (Aspirin) ===")
st = get_st(tgt)
r = requests.get(f"{UMLS_BASE}/content/current/CUI/C0004057/relations",
                 params={"ticket": st, "sabs": "SNOMEDCT_US", "pageSize": 50})
print(f"Status: {r.status_code}")
results = r.json().get("result", [])
print(f"Count: {len(results)}")
for rel in results[:10]:
    print(f"  [{rel.get('relationLabel','')}]  {rel.get('relatedIdName','')}  |  {rel.get('additionalRelationLabel','')}")

time.sleep(0.5)

# Try 2 — cross-references via atoms (gets SNOMED concept ID)
print("\n=== SNOMED atom for C0004057 ===")
st = get_st(tgt)
r = requests.get(f"{UMLS_BASE}/content/current/CUI/C0004057/atoms",
                 params={"ticket": st, "sabs": "SNOMEDCT_US", "pageSize": 5})
print(f"Status: {r.status_code}")
results = r.json().get("result", [])
for a in results[:5]:
    print(f"  {a.get('ui','')}  {a.get('name','')}  code={a.get('code','')}")

time.sleep(0.5)

# Try 3 — broader relations including additionalRelationLabel
print("\n=== ALL relations C0289313 (Rosiglitazone) with additionalRelationLabel ===")
st = get_st(tgt)
r = requests.get(f"{UMLS_BASE}/content/current/CUI/C0289313/relations",
                 params={"ticket": st, "pageSize": 50})
results = r.json().get("result", [])
for rel in results:
    add = rel.get("additionalRelationLabel", "")
    name = rel.get("relatedIdName", "")
    label = rel.get("relationLabel", "")
    if add:
        print(f"  [{label}|{add}]  {name}")

time.sleep(0.5)

# Try 4 — search broader for "aspirin cardiovascular"
print("\n=== Searching 'aspirin cardiovascular prevention' ===")
st = get_st(tgt)
r = requests.get(f"{UMLS_BASE}/search/current",
                 params={"string": "aspirin cardiovascular prevention",
                         "ticket": st, "returnIdType": "concept", "pageSize": 5})
for res in r.json()["result"]["results"][:5]:
    print(f"  {res['ui']}  {res['name']}")
