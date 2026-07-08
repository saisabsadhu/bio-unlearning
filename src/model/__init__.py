from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, open_dict
from typing import Dict, Any
import os
import torch
import logging
from model.probe import ProbedLlamaForCausalLM

hf_home = os.getenv("HF_HOME", default=None)

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Any] = {}


def _register_model(model_class):
    MODEL_REGISTRY[model_class.__name__] = model_class


def get_dtype(model_args):
    with open_dict(model_args):
        torch_dtype = model_args.pop("torch_dtype", None)
    if model_args.get("attn_implementation", None) == "flash_attention_2":
        # This check handles https://github.com/Dao-AILab/flash-attention/blob/7153673c1a3c7753c38e4c10ef2c98a02be5f778/flash_attn/flash_attn_triton.py#L820
        # If you want to run at other precisions consider running "training or inference using
        # Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):`
        # decorator" or using an attn_implementation compatible with the precision in the model
        # config.
        assert torch_dtype in ["float16", "bfloat16"], ValueError(
            f"Invalid torch_dtype '{torch_dtype}' for the requested attention "
            f"implementation: 'flash_attention_2'. Supported types are 'float16' "
            f"and 'bfloat16'."
        )
    if torch_dtype == "float16":
        return torch.float16
    elif torch_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def get_model(model_cfg: DictConfig):
    assert model_cfg is not None and model_cfg.model_args is not None, ValueError(
        "Model config not found or model_args absent in configs/model."
    )
    model_args = model_cfg.model_args
    tokenizer_args = model_cfg.tokenizer_args
    torch_dtype = get_dtype(model_args)
    model_handler = model_cfg.get("model_handler", "AutoModelForCausalLM")
    model_cls = MODEL_REGISTRY[model_handler]
    with open_dict(model_args):
        model_path = model_args.pop("pretrained_model_name_or_path", None)
    try:
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            **model_args,
            cache_dir=hf_home,
        )
    except Exception as e:
        logger.warning(f"Model {model_path} requested with {model_cfg.model_args}")
        raise ValueError(
            f"Error {e} while fetching model using {model_handler}.from_pretrained()."
        )

    # Opt-in LoRA wrapping: on a shared GPU, full fine-tuning of a 7B model plus a
    # KL-based method's frozen reference-model copy (see trainer/unlearn/grad_diff.py)
    # can exceed available memory regardless of the reference model's own precision --
    # the trainable model's own weights+gradients+optimizer state are the dominant
    # cost, and quantizing only the reference model doesn't address that. LoRA cuts
    # trainable-parameter memory (gradients + optimizer state) to a small fraction of
    # the full model, at the cost of being a lower-rank approximation of full
    # fine-tuning. Off by default (full fine-tuning is closer to what the unlearning
    # literature reports); set BIOUNLEARN_USE_LORA=1 to enable.
    if os.environ.get("BIOUNLEARN_USE_LORA", "0") == "1":
        from peft import LoraConfig, get_peft_model

        lora_r = int(os.environ.get("BIOUNLEARN_LORA_R", "16"))
        lora_alpha = int(os.environ.get("BIOUNLEARN_LORA_ALPHA", "32"))
        logger.info(f"Wrapping model with LoRA (r={lora_r}, alpha={lora_alpha}).")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        # peft's default LoRA adapter dtype doesn't always match the base model's
        # torch_dtype (bf16 here) -- a dtype mismatch between the base model's
        # bf16 params/gradients and fp16(-ish) adapter params surfaces as
        # "RuntimeError: Found dtype Half but expected BFloat16" during backward,
        # only with certain trainers (e.g. RMU, which accumulates gradients across
        # multiple cached activations). Force uniform dtype explicitly.
        for name, param in model.named_parameters():
            if param.requires_grad and param.dtype != torch_dtype:
                param.data = param.data.to(torch_dtype)
        model.print_trainable_parameters()
        # Required with gradient_checkpointing=True: the embedding layer's output
        # otherwise has requires_grad=False (nothing upstream of the LoRA adapters is
        # trainable), which breaks checkpointing's recomputed backward pass entirely
        # ("element 0 of tensors does not require grad and does not have a grad_fn").
        # This registers a hook forcing the embedding output to require grad.
        model.enable_input_require_grads()

    tokenizer = get_tokenizer(tokenizer_args)
    return model, tokenizer


def _add_or_replace_eos_token(tokenizer, eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.info("New tokens have been added, make sure `resize_vocab` is True.")


def get_tokenizer(tokenizer_cfg: DictConfig):
    try:
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)
    except Exception as e:
        error_message = (
            f"{'--' * 40}\n"
            f"Error {e} fetching tokenizer using AutoTokenizer.\n"
            f"Tokenizer requested from path: {tokenizer_cfg.get('pretrained_model_name_or_path', None)}\n"
            f"Full tokenizer config: {tokenizer_cfg}\n"
            f"{'--' * 40}"
        )
        raise RuntimeError(error_message)

    if tokenizer.eos_token_id is None:
        logger.info("replacing eos_token with <|endoftext|>")
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Setting pad_token as eos token: {}".format(tokenizer.pad_token))

    return tokenizer


# register models
_register_model(AutoModelForCausalLM)
_register_model(ProbedLlamaForCausalLM)
