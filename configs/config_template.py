# Copy this to config.py and fill in your keys
UMLS_API_KEY     = "YOUR_UMLS_KEY_HERE"
OPENROUTER_KEY   = "YOUR_OPENROUTER_KEY_HERE"
OPENROUTER_BASE  = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemini-2.5-flash"
SITE_URL         = "https://github.com/saisabsadhu/bio-unlearning"
SITE_NAME        = "BioUnlearn"
UMLS_BASE        = "https://uts-ws.nlm.nih.gov/rest"
UMLS_LOGIN       = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
UMLS_SERVICE     = "http://umlsks.nlm.nih.gov"
HOP_DEPTH        = 2
ALPHA            = 0.6
MAX_RETAIN       = 25
KEEP_RELATIONS   = {
    "may_treat", "has_contraindication", "ingredient_of",
    "has_mechanism", "part_of", "isa", "associated_with",
    "contraindicated_with", "PAR", "CHD", "RB", "RN"
}
PAC_FORGET  = 800
PAC_RETAIN  = 600
RGU_FORGET  = 700
RGU_RETAIN  = 700
IFE_FORGET  = 1300
IFE_RETAIN  = 300
