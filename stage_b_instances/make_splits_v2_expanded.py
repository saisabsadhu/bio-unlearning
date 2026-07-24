"""
Stage D (v2) -- builds an EXPANDED dataset (original 190-instance pilot +
352 new Herrera-Perez-derived RGU instances) as a SEPARATE, clearly-labeled
set. Does NOT touch data/splits/*.jsonl.

Why separate: every GA/NPO/RMU/RMU/OGDA/RAG-suppression baseline result
computed so far in this project is tied to the exact original 190-instance
(58 RGU / 56 IFE train) set for comparability. Silently regenerating those
files would invalidate every existing comparison. The expanded set below is
for future benchmark-scale work and is versioned into data/splits_v2/.

Herrera-Perez instances have no umls_cui (they aren't concept-graph-linked)
-- each is independently generated from its own real citation, so each gets
its own pseudo-concept key for split purposes (no leakage risk the way
multi-instance-per-concept UMLS data has).
"""

import json
import os
import random

random.seed(42)

SPLITS_DIR = "data/splits"
SPLITS_V2_DIR = "data/splits_v2"
HERRERA_PASSED = "data/filtered/RGU_herrera_perez_passed.jsonl"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def make_splits(instances, train=0.60, val=0.20):
    by_concept = {}
    for i, inst in enumerate(instances):
        key = inst.get("umls_cui") or f"hp_{inst.get('instance_id', i)}"
        by_concept.setdefault(key, []).append(inst)

    keys = list(by_concept.keys())
    random.shuffle(keys)
    n = len(keys)
    n_train = max(1, int(n * train))
    n_val = max(1, int(n * val))

    train_inst = [i for k in keys[:n_train] for i in by_concept[k]]
    val_inst = [i for k in keys[n_train : n_train + n_val] for i in by_concept[k]]
    test_inst = [i for k in keys[n_train + n_val :] for i in by_concept[k]]
    return train_inst, val_inst, test_inst


def main():
    os.makedirs(SPLITS_V2_DIR, exist_ok=True)

    # Original pilot RGU instances (recombine existing splits back to one pool)
    original_rgu = (
        load_jsonl(f"{SPLITS_DIR}/RGU_train.jsonl")
        + load_jsonl(f"{SPLITS_DIR}/RGU_val.jsonl")
        + load_jsonl(f"{SPLITS_DIR}/RGU_test.jsonl")
    )
    herrera_rgu = load_jsonl(HERRERA_PASSED)
    for inst in herrera_rgu:
        # normalize to match original pilot's string-typed reversal_year --
        # mixed str/int/None in the same jsonl column breaks pyarrow's loader
        inst["reversal_year"] = str(inst["reversal_year"]) if inst.get("reversal_year") is not None else ""

    print(f"Original pilot RGU instances: {len(original_rgu)}")
    print(f"New Herrera-Perez RGU instances: {len(herrera_rgu)}")

    combined_rgu = original_rgu + herrera_rgu
    print(f"Combined RGU pool: {len(combined_rgu)}")

    train, val, test = make_splits(combined_rgu)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = f"{SPLITS_V2_DIR}/RGU_{name}.jsonl"
        with open(path, "w") as f:
            for inst in split:
                f.write(json.dumps(inst) + "\n")
        print(f"  {name}: {len(split)} -> {path}")

    # IFE unchanged for now (Herrera-Perez entries are all RGU-shaped practice reversals)
    for name in ["train", "val", "test"]:
        src = f"{SPLITS_DIR}/IFE_{name}.jsonl"
        dst = f"{SPLITS_V2_DIR}/IFE_{name}.jsonl"
        with open(src) as f_in, open(dst, "w") as f_out:
            f_out.write(f_in.read())
        print(f"  IFE {name}: copied unchanged -> {dst}")

    print(f"\nTotal RGU: {len(combined_rgu)}  |  Total IFE: {sum(1 for _ in open(f'{SPLITS_V2_DIR}/IFE_train.jsonl')) + sum(1 for _ in open(f'{SPLITS_V2_DIR}/IFE_val.jsonl')) + sum(1 for _ in open(f'{SPLITS_V2_DIR}/IFE_test.jsonl'))}")
    print(f"Grand total dataset size: {len(combined_rgu) + sum(1 for _ in open(f'{SPLITS_V2_DIR}/IFE_train.jsonl')) + sum(1 for _ in open(f'{SPLITS_V2_DIR}/IFE_val.jsonl')) + sum(1 for _ in open(f'{SPLITS_V2_DIR}/IFE_test.jsonl'))}")
    print(f"\nNOTE: data/splits/ (original pilot, 190 instances) is UNCHANGED.")
    print(f"All existing baseline results (GA/NPO/RMU/OGDA/RAG-suppression) remain comparable to each other,")
    print(f"since they were all computed against data/splits/, not data/splits_v2/.")


if __name__ == "__main__":
    main()
