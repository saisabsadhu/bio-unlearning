"""
Full Pilot Generation — 100 RGU + 100 IFE instances
Uses 20 seed concepts per scenario × 5 instances each.
Improved prompts enforce question diversity within same concept.
"""

import json, os, sys, time, random
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    OPENROUTER_KEY, OPENROUTER_BASE,
    OPENROUTER_MODEL, SITE_URL, SITE_NAME
)

client = OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_KEY)
RAW_DIR = "data/raw"

# ── Seed concepts ─────────────────────────────────────────────────────

RGU_CONCEPTS = [
    # (concept, umls_cui, reversal_context)
    ("Aspirin primary prevention",      "C0004057", "USPSTF 2022 reversed recommendation for adults 60+ with no prior CVD event due to bleeding risk"),
    ("Hormone Replacement Therapy",     "C0282402", "WHI 2002 trial showed increased breast cancer and cardiovascular risk with combined estrogen-progestin"),
    ("Rosiglitazone",                   "C0289313", "FDA 2010 severely restricted due to cardiovascular risk; European market withdrawal"),
    ("Vioxx rofecoxib",                 "C0876768", "Merck 2004 voluntary worldwide withdrawal due to increased MI and stroke risk"),
    ("Hormone therapy breast cancer",   "C0206255", "HABITS trial 2004 showed HRT contraindicated in breast cancer survivors"),
    ("Beta-carotene supplements",       "C0053275", "CARET trial 1996 showed increased lung cancer risk in smokers — reversed recommendation"),
    ("Postmenopausal estrogen",         "C0014939", "HERS trial 1998 showed no CVD benefit, reversed recommendation for secondary prevention"),
    ("COX-2 inhibitor celecoxib",       "C0538927", "CLASS trial 2000 showed cardiovascular risk, dose restrictions imposed"),
    ("Tight glycemic control ICU",      "C0017725", "NICE-SUGAR 2009 showed intensive insulin therapy increased ICU mortality"),
    ("High-dose vitamin E",             "C0042890", "HOPE-TOO 2005 showed increased heart failure risk, reversed recommendation"),
    ("Routine episiotomy",              "C0014791", "Evidence showed harms outweigh benefits, WHO reversed routine use recommendation"),
    ("Antiarrhythmic drugs post-MI",    "C0003195", "CAST trial 1989 showed flecainide/encainide increased mortality post-MI"),
    ("Diethylstilbestrol pregnancy",    "C0012144", "FDA 1971 withdrew for pregnancy use after DES daughters showed vaginal cancer"),
    ("Arthroscopic knee surgery OA",    "C0023416", "Moseley 2002 RCT showed no benefit over sham surgery for knee OA"),
    ("Oxygen therapy COPD high-flow",  "C0024117", "Austin 2010 showed high-flow O2 increased mortality in COPD exacerbations"),
    ("Bed rest low back pain",          "C0004604", "Multiple RCTs showed bed rest harmful, activity recommended instead"),
    ("Antibiotics acute bronchitis",    "C0006277", "Cochrane 2017 showed no benefit, increased resistance — guidelines reversed"),
    ("Prophylactic lidocaine post-MI",  "C0023848", "CAST showed prophylactic antiarrhythmics increased mortality post-MI"),
    ("Megestrol appetite AIDS",         "C0025329", "Studies showed increased thrombotic risk, no survival benefit"),
    ("Routine fetal monitoring EFM",    "C0015945", "Cochrane showed EFM increased C-section rate without improving neonatal outcomes"),
]

IFE_CONCEPTS = [
    # (concept, umls_cui, error_context)
    ("Metformin renal failure",         "C0025598", "Common misconception: metformin absolutely contraindicated in all CKD — actually safe in mild-moderate CKD per 2016 FDA update"),
    ("Warfarin vitamin K foods",        "C0043031", "Misconception: patients must avoid all vitamin K foods — actually consistency matters more than avoidance"),
    ("Digoxin toxicity hypokalemia",    "C0012265", "Misconception: digoxin toxicity only occurs at high doses — actually hypokalemia dramatically lowers toxic threshold"),
    ("ACE inhibitor hyperkalemia",      "C0003015", "Misconception: ACE inhibitors safe in severe hyperkalemia — actually contraindicated above K+ 5.5"),
    ("Fluoroquinolone tendon rupture",  "C0016403", "Misconception: tendon rupture risk is rare — actually FDA black box warning, high risk in elderly on steroids"),
    ("Clopidogrel PPI interaction",     "C0070166", "Misconception: all PPIs equally safe with clopidogrel — actually omeprazole significantly reduces antiplatelet effect"),
    ("Statins myopathy rhabdomyolysis", "C0036418", "Misconception: statin myopathy is only mild — actually can progress to fatal rhabdomyolysis especially with CYP3A4 inhibitors"),
    ("SSRI serotonin syndrome",         "C0036098", "Misconception: SSRIs safe to combine with tramadol — actually high serotonin syndrome risk"),
    ("Methotrexate folic acid",         "C0025677", "Misconception: folic acid reduces methotrexate efficacy — actually reduces toxicity without reducing efficacy"),
    ("Amiodarone thyroid toxicity",     "C0002598", "Misconception: amiodarone thyroid effects are mild — actually causes both hypo and hyperthyroidism, requires monitoring"),
    ("NSAIDs renal impairment",         "C0021345", "Misconception: short-term NSAIDs safe in renal impairment — actually even short courses cause AKI in susceptible patients"),
    ("Lithium NSAIDs interaction",      "C0023870", "Misconception: ibuprofen safe with lithium — actually NSAIDs raise lithium levels to toxic range"),
    ("Benzodiazepine elderly falls",    "C0003615", "Misconception: low-dose benzos safe in elderly — actually Beers criteria lists them as high-risk for falls and fractures"),
    ("Contraceptive pill DVT risk",     "C0009905", "Misconception: all oral contraceptives equal DVT risk — actually third-generation pills have higher VTE risk than second-gen"),
    ("Acetaminophen liver safe dose",   "C0000970", "Misconception: 4g/day acetaminophen safe in all adults — actually 2g/day recommended in liver disease and alcohol users"),
    ("Antibiotic prophylaxis dental",   "C0003232", "Misconception: antibiotic prophylaxis required for all cardiac conditions before dental work — AHA 2007 restricted to high-risk only"),
    ("Morphine respiratory depression", "C0026549", "Misconception: morphine contraindicated in COPD — actually safe at appropriate doses for palliative dyspnea"),
    ("Iron supplement constipation",    "C0015133", "Misconception: all iron supplements cause equal constipation — actually ferrous gluconate better tolerated than ferrous sulfate"),
    ("Calcium channel blocker HF",      "C0006567", "Misconception: all calcium channel blockers safe in heart failure — actually verapamil and diltiazem contraindicated in HFrEF"),
    ("Proton pump inhibitor B12",       "C0006312", "Misconception: short-term PPI use does not affect B12 — actually long-term use causes B12 deficiency requiring monitoring"),
]

# ── Prompt templates with diversity enforcement ───────────────────────

RGU_SYSTEM = """You are a clinical NLP dataset curator building a machine unlearning benchmark.
Generate ONE instance of Retracted Guideline Unlearning (RGU).

Output ONLY a valid JSON object with these exact fields:
{
  "prompt": "clinical question",
  "forget_answer": "old guideline answer — specific, what model trained before reversal would say",
  "retain_answer": "new guideline answer — specific, what current evidence says",
  "severity": "life-threatening | clinically-significant | benign",
  "reversal_reason": "one sentence — why the guideline changed",
  "reversal_year": "year as string",
  "ground_truth_source": "specific source e.g. USPSTF 2022, WHI trial 2002, FDA 2010"
}

Critical rules:
- prompt must NOT telegraph which answer is correct
- Both answers must be clinically specific, not vague
- Do NOT repeat question phrasings used in prior instances
- Output ONLY the JSON, no markdown, no explanation"""

IFE_SYSTEM = """You are a clinical NLP dataset curator building a machine unlearning benchmark.
Generate ONE instance of Incorrect Fact Erasure (IFE).

Output ONLY a valid JSON object with these exact fields:
{
  "prompt": "clinical question about drug safety, dosing, interaction, or mechanism",
  "forget_answer": "the WRONG answer — specific plausible misconception a clinical LLM might produce",
  "retain_answer": "the CORRECT current answer — specific, verified against named source",
  "severity": "life-threatening | clinically-significant | benign",
  "error_type": "incorrect_interaction | wrong_dosage | debunked_mechanism | wrong_contraindication | wrong_pregnancy_category",
  "ground_truth_source": "e.g. DrugBank 5.1, FDA label 2023, AHA 2007 guidelines"
}

Critical rules:
- wrong answer must be specific and plausible — not obviously wrong
- correct answer must cite a named authoritative source
- prompt must be a realistic clinical scenario
- Do NOT repeat question phrasings used in prior instances
- Output ONLY the JSON, no markdown, no explanation"""

# Question angle templates to enforce diversity
RGU_ANGLES = [
    "Ask about current recommendation for a specific patient type",
    "Ask about contraindications or when NOT to use this treatment",
    "Ask about what changed in guidelines and why",
    "Ask about risk-benefit for a borderline patient case",
    "Ask what a clinician should do if patient is already on this treatment",
]

IFE_ANGLES = [
    "Ask about drug interaction in a specific co-administration scenario",
    "Ask about appropriate dosing in a patient with organ impairment",
    "Ask about contraindications in a vulnerable population",
    "Ask about monitoring requirements and warning signs",
    "Ask about mechanism of the adverse effect",
]

# ── LLM call ─────────────────────────────────────────────────────────

def call_llm(system, user, temperature=0.85):
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": SITE_URL,
                    "X-Title":      SITE_NAME,
                },
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=temperature,
                max_tokens=600,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"      [WARN] attempt {attempt+1}: {e}")
            time.sleep(3)
    return None

def parse_json(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text  = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
    import re
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None

# ── Generate for one concept ──────────────────────────────────────────

def generate_concept(concept, cui, context, scenario,
                     n=5, angles=None):
    instances  = []
    used_angles = []

    for i in range(n):
        angle = angles[i % len(angles)] if angles else ""

        if scenario == "RGU":
            user = f"""Concept: {concept}
Reversal context: {context}
Question angle to use: {angle}
Previously used angles in this concept: {used_angles}

Generate a NEW RGU instance with a DIFFERENT question phrasing than any prior instance.
The question angle MUST be: {angle}"""
        else:
            user = f"""Concept: {concept}
Error context: {context}
Question angle to use: {angle}
Previously used angles in this concept: {used_angles}

Generate a NEW IFE instance with a DIFFERENT question phrasing than any prior instance.
The question angle MUST be: {angle}"""

        system = RGU_SYSTEM if scenario == "RGU" else IFE_SYSTEM
        raw    = call_llm(system, user)
        data   = parse_json(raw)

        if not data:
            print(f"      [FAIL] parse error on instance {i+1}")
            continue

        data["umls_cui"]    = cui
        data["scenario"]    = scenario
        data["concept"]     = concept
        data["instance_id"] = f"{cui}_{scenario}_{i+1:03d}"
        data["angle"]       = angle

        instances.append(data)
        used_angles.append(angle)
        print(f"      [{i+1}/{n}] OK  severity={data.get('severity','?')}  "
              f"angle={angle[:35]}")
        time.sleep(0.8)

    return instances

# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)

    # Clear old raw files
    for f in ["RGU_raw.jsonl", "IFE_raw.jsonl"]:
        path = os.path.join(RAW_DIR, f)
        if os.path.exists(path):
            os.remove(path)

    print("BioUnlearn — Full Pilot Generation")
    print("Target: 100 RGU + 100 IFE instances")
    print("20 concepts × 5 instances × 2 scenarios\n")

    total_rgu = 0
    total_ife = 0

    # ── RGU ──────────────────────────────────────────────────────────
    print("━"*60)
    print("  SCENARIO: RGU — Retracted Guideline Unlearning")
    print("━"*60)

    for concept, cui, context in RGU_CONCEPTS:
        print(f"\n  [{cui}] {concept}")
        instances = generate_concept(
            concept, cui, context, "RGU",
            n=5, angles=RGU_ANGLES
        )
        with open(f"{RAW_DIR}/RGU_raw.jsonl", "a") as f:
            for inst in instances:
                f.write(json.dumps(inst) + "\n")
        total_rgu += len(instances)
        print(f"  → {len(instances)} instances saved  "
              f"(running total: {total_rgu})")
        time.sleep(0.5)

    # ── IFE ──────────────────────────────────────────────────────────
    print("\n" + "━"*60)
    print("  SCENARIO: IFE — Incorrect Fact Erasure")
    print("━"*60)

    for concept, cui, context in IFE_CONCEPTS:
        print(f"\n  [{cui}] {concept}")
        instances = generate_concept(
            concept, cui, context, "IFE",
            n=5, angles=IFE_ANGLES
        )
        with open(f"{RAW_DIR}/IFE_raw.jsonl", "a") as f:
            for inst in instances:
                f.write(json.dumps(inst) + "\n")
        total_ife += len(instances)
        print(f"  → {len(instances)} instances saved  "
              f"(running total: {total_ife})")
        time.sleep(0.5)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"  RGU raw instances : {total_rgu}")
    print(f"  IFE raw instances : {total_ife}")
    print(f"  Total raw         : {total_rgu + total_ife}")
    print(f"\nRaw files → {RAW_DIR}/")
    print("Next: run filter_instances.py to apply quality filters")
