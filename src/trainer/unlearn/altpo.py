from trainer.utils import compute_dpo_loss
from trainer.unlearn.grad_diff import GradDiff

class AltPO(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]["forget"]
        
        forget_loss, forget_outputs = compute_dpo_loss(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )

        forget_sub_inputs = inputs["forget"]["forget_sub"]
        forget_sub_inputs = {
            "input_ids": forget_sub_inputs["input_ids"],
            "attention_mask": forget_sub_inputs["attention_mask"],
            "labels": forget_sub_inputs["labels"],
        }
        forget_sub_loss = self.compute_retain_loss(model=model, retain_inputs=forget_sub_inputs)

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.gamma * forget_sub_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
