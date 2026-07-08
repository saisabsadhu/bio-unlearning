from trainer.unlearn.base import UnlearnTrainer


class SurgicalGradAscent(UnlearnTrainer):
    def __init__(self, target_layers=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Default layers for (Llama-8B)
        if target_layers is None:
            self.target_layers = list(range(10, 21))
        else:
            self.target_layers = target_layers

        unfrozen_params = 0

        # Unfreeze only selected MLP down_proj layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for layer_idx in self.target_layers:
                layer = self.model.model.layers[layer_idx]
                for param in layer.mlp.down_proj.parameters():
                    param.requires_grad = True
                    unfrozen_params += param.numel()

        print("\n==== Surgical Gradient Ascent ====")
        print("Target layers:", self.target_layers)
        print(f"Trainable parameters: {unfrozen_params:,}")
        print("==================================\n")

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        forget_inputs = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        outputs = model(**forget_inputs)

        # Same objective as ordinary GradAscent
        loss = -outputs.loss

        return (loss, outputs) if return_outputs else loss
