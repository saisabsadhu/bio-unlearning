import copy
import logging
import os

import torch
from trainer.utils import compute_kl_divergence
from trainer.unlearn.base import UnlearnTrainer

logger = logging.getLogger("trainer")


class GradDiff(UnlearnTrainer):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.ref_model = None
        if retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def _prepare_ref_model(self, model):
        # This shared reference-model copy is the dominant memory cost for
        # KL-based methods (NPO/RMU/GradDiff-KL/UNDIAL/WGA/SatImp/DPO) on a
        # single, often shared, GPU -- a full-precision deepcopy roughly
        # doubles the base model's memory. The reference model is
        # frozen/eval-only (no gradients ever flow into it), so it's a good
        # candidate for 8-bit quantization: reload it fresh from the same
        # checkpoint in int8 instead of deepcopy-ing the live bf16/fp16
        # weights, unless BIOUNLEARN_REF_MODEL_FULL_PRECISION=1 is set (e.g.
        # for exact-precision ablation comparisons).
        use_8bit = os.environ.get("BIOUNLEARN_REF_MODEL_FULL_PRECISION", "0") != "1"
        if use_8bit and not self.is_deepspeed_enabled:
            try:
                from transformers import AutoModelForCausalLM, BitsAndBytesConfig

                name_or_path = model.config._name_or_path
                # Match the base model's quantization when QLoRA is active (see
                # src/model/__init__.py) -- 4-bit for both trainable base and
                # reference squeezes out the most memory for larger (8B+) models.
                use_4bit = os.environ.get("BIOUNLEARN_QLORA", "0") == "1"
                bits = "4-bit" if use_4bit else "8-bit"
                logger.info(
                    f"Loading reference model '{name_or_path}' in {bits} "
                    "(set BIOUNLEARN_REF_MODEL_FULL_PRECISION=1 to disable)."
                )
                quant_config = (
                    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
                    if use_4bit
                    else BitsAndBytesConfig(load_in_8bit=True)
                )
                ref_model = AutoModelForCausalLM.from_pretrained(
                    name_or_path,
                    quantization_config=quant_config,
                    device_map={"": self.accelerator.device},
                )
                ref_model.eval()
                return ref_model
            except Exception as e:
                logger.warning(
                    f"8-bit reference model load failed ({e}); falling back "
                    "to full-precision deepcopy."
                )

        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        return retain_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        forget_outputs = model(**forget_inputs)
        forget_loss = -forget_outputs.loss

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        return (loss, forget_outputs) if return_outputs else loss
