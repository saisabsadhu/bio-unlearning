# ── BioUnlearn Master Config ──────────────────────────────────────────

UMLS_API_KEY   = "PASTE_YOUR_UMLS_API_KEY_HERE"
ANTHROPIC_KEY  = "PASTE_YOUR_ANTHROPIC_KEY_HERE"   # needed in Stage B

UMLS_BASE      = "https://uts-ws.nlm.nih.gov/rest"
UMLS_LOGIN     = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
UMLS_SERVICE   = "http://umlsks.nlm.nih.gov"

HOP_DEPTH      = 2       # k=2 default
ALPHA          = 0.6     # retention weight formula
MAX_RETAIN     = 25      # max OGFR instances per forget concept

# Clinically significant relation types to keep during graph traversal
KEEP_RELATIONS = {
    "may_treat", "has_contraindication", "ingredient_of",
    "has_mechanism", "part_of", "isa", "associated_with",
    "contraindicated_with", "PAR", "CHD", "RB", "RN"
}

# Scenario sizes (targets)
PAC_FORGET  = 800
PAC_RETAIN  = 600
RGU_FORGET  = 700
RGU_RETAIN  = 700
IFE_FORGET  = 1300
IFE_RETAIN  = 300
