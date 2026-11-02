from trainer.utils import compute_dpo_loss_with_outputs
from trainer.unlearn.grad_diff import GradDiff
from torch.nn import CrossEntropyLoss


class TNPO(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):

        forget_inputs = inputs["forget"]

        forget_loss, outputs = compute_dpo_loss_with_outputs(
            model=model,
            ref_model=self.ref_model,
            win_inputs=None,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )

        _, forget_outputs, _, forget_ref_outputs = outputs

        shift_labels = forget_inputs["labels"][..., 1:].contiguous()
        shift_logits = forget_outputs.logits[..., :-1, :].contiguous()
        shift_logits_ref = forget_ref_outputs.logits[..., :-1, :].contiguous()
        
        ce = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        ce_ref = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits_ref.view(-1, shift_logits_ref.size(-1)), shift_labels.view(-1))
        probs, probs_ref = (-ce).exp().detach(), (-ce_ref).exp().detach()
        npo_weights = 2 * probs ** self.beta / (probs ** self.beta + probs_ref ** self.beta + 1e-5)
        forget_loss = -(npo_weights * ce)[shift_labels.view(-1)!=-100].mean()

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss# + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss