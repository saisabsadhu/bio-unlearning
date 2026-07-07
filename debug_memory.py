import torch
from transformers import AutoModelForCausalLM

print(torch.cuda.mem_get_info())

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
).cuda()

torch.cuda.synchronize()

print(torch.cuda.memory_summary())

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct"
)

inputs = tokenizer(
    "Hello world!",
    return_tensors="pt"
).to("cuda")

outputs = model(
    **inputs,
    labels=inputs["input_ids"]
)

print("Forward OK")

outputs.loss.backward()

print("Backward OK")
