"""Borrowed implementation from https://github.com/centerforaisafety/wmdp/blob/main/rmu/unlearn.py"""
import re
import torch
import deepspeed
from trainer.unlearn.grad_diff import GradDiff


class RMU(GradDiff):
    def __init__(
        self,
        module_regex=r"model\.layers\.(14|16|18|20)",
        trainable_params_regex=[".*"],
        steering_coeff=20,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)
        self.trainable_params_regex = trainable_params_regex
        self.module_regex = module_regex
        self.model_modules = self._get_matching_modules(self.model, self.module_regex)
        self.ref_modules = self._get_matching_modules(self.ref_model, self.module_regex)
        self.steering_coeff = steering_coeff
        self.control_vec = None

    def create_optimizer(self):
        self._freeze_all_params(self.model, False)
        self._set_trainable_params(self.model, self.trainable_params_regex, True)
        trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
        print(f"[RMU] create_optimizer: {trainable} trainable params")
        super().create_optimizer()

    def _get_matching_modules(self, model, module_regex):
        if isinstance(model, deepspeed.DeepSpeedEngine):
            model = model.module
        matched = {
            name: module
            for name, module in model.named_modules()
            if re.fullmatch(module_regex, name)
        }
        if not matched:
            raise ValueError(f"No module matched with regex: {module_regex}")
        print(f"[RMU] Matched modules: {list(matched.keys())}")
        return list(matched.values())

    def _freeze_all_params(self, model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def _set_trainable_params(self, model, trainable_params_regex, requires_grad=True):
        matched_count = 0
        for name, param in model.named_parameters():
            if any(re.fullmatch(pattern, name) for pattern in trainable_params_regex):
                param.requires_grad = requires_grad
                matched_count += 1
        print(f"[RMU] Set requires_grad={requires_grad} on {matched_count} parameters")

    def forward_with_cache_all_modules(self, model, inputs, modules, no_grad=False):
        """Single forward pass extracting activations from ALL target modules at once."""
        caches = {id(m): None for m in modules}

        def make_hook(m, detach):
            def hook(mod, input, output):
                out = output[0] if isinstance(output, tuple) else output
                caches[id(mod)] = out.detach() if detach else out
            return hook

        handles = [
            m.register_forward_hook(make_hook(m, detach=no_grad))
            for m in modules
        ]
        with torch.set_grad_enabled(not no_grad):
            outputs = model(**inputs)
        for h in handles:
            h.remove()

        activations = [caches[id(m)] for m in modules]
        return activations, outputs

    def get_control_vector(self, dim):
        if self.control_vec is None:
            random_vector = torch.rand(1, 1, dim)
            self.control_vec = (
                random_vector / torch.norm(random_vector) * self.steering_coeff
            )
        return self.control_vec

    def compute_activation_loss(self, activation1, activation2, mask):
        squared_diff = torch.nn.functional.mse_loss(
            activation1, activation2, reduction="none"
        )
        expanded_mask = mask.unsqueeze(-1).expand_as(squared_diff)
        squared_diff_sum = (squared_diff * expanded_mask).mean(dim=2).sum(dim=1)
        num_tokens = mask.sum(dim=-1, keepdim=True)
        return (squared_diff_sum / num_tokens).mean()

    def compute_retain_loss(self, model, retain_inputs):
        print(f"[RETAIN] inputs keys: {list(retain_inputs.keys())}, input_ids shape: {retain_inputs['input_ids'].shape}")
        if self.retain_loss_type == "EMBED_DIFF":
            model_acts, _ = self.forward_with_cache_all_modules(
                model, retain_inputs, self.model_modules, no_grad=False
            )
            ref_acts, _ = self.forward_with_cache_all_modules(
                self.ref_model, retain_inputs, self.ref_modules, no_grad=True
            )
            mask = retain_inputs["labels"] != -100
            # Start accumulation from first real loss — NOT from torch.tensor(0.0)
            retain_loss = None
            for model_act, ref_act in zip(model_acts, ref_acts):
                l = self.compute_activation_loss(
                    model_act, ref_act.to(model_act.device), mask
                )
                retain_loss = l if retain_loss is None else retain_loss + l
            return retain_loss / len(self.model_modules)
        else:
            return super().compute_retain_loss(model, retain_inputs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        print(f"[DEBUG] retain_loss_type={self.retain_loss_type}, alpha={self.alpha}, gamma={self.gamma}")
        forget_inputs = {
            "input_ids": inputs["forget"]["input_ids"],
            "attention_mask": inputs["forget"]["attention_mask"],
            "labels": inputs["forget"]["labels"],
        }
        retain_inputs = {
            "input_ids": inputs["retain"]["input_ids"],
            "attention_mask": inputs["retain"]["attention_mask"],
            "labels": inputs["retain"]["labels"],
        }

        model_acts, first_outputs = self.forward_with_cache_all_modules(
            model, forget_inputs, self.model_modules, no_grad=False
        )

        # Accumulate forget loss without torch.tensor(0.0) seed
        forget_loss = None
        for model_act in model_acts:
            control_vec = self.get_control_vector(model_act.shape[-1])
            control_vec = control_vec.to(
                dtype=model_act.dtype, device=model_act.device
            ).expand_as(model_act)
            mask = forget_inputs["labels"] != -100
            l = self.compute_activation_loss(model_act, control_vec, mask)
            forget_loss = l if forget_loss is None else forget_loss + l
        forget_loss = forget_loss / len(self.model_modules)

        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        print(f"[LOSS] forget={forget_loss.item():.4f} retain={retain_loss.item():.6f} total={loss.item():.4f} grad_fn={loss.grad_fn}")

        return (loss, first_outputs) if return_outputs else loss
