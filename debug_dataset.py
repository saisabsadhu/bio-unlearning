import sys
sys.path.insert(0, "src")

from omegaconf import OmegaConf
from data import get_data
from model import get_model

cfg = OmegaConf.load("configs/train.yaml")
exp = OmegaConf.load("configs/experiment/unlearn/wmdp/llama31_npo_cyber.yaml")

cfg.merge_with(exp)
cfg.mode = "unlearn"

model, tokenizer = get_model(cfg.model)

data = get_data(
    cfg.data,
    mode="unlearn",
    tokenizer=tokenizer,
    template_args=cfg.model.template_args,
)

sample = data["train"][0]

print(sample.keys())
print(sample["forget"].keys())
print(sample["retain"].keys())

