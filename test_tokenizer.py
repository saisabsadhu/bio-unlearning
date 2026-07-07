from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
print("pad_token:", tok.pad_token)
print("eos_token:", tok.eos_token)
print("padding_side:", tok.padding_side)

import json
with open("data/wmdp/wmdp-corpora/cyber-forget-corpus.jsonl") as f:
    line = json.loads(f.readline())
print("keys:", line.keys())
text = line.get("text", str(line))
ids = tok(text)["input_ids"]
print("num tokens:", len(ids))
print("first 20 ids:", ids[:20])
