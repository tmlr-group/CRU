from trainer.unlearn.base import UnlearnTrainer
from torch.nn import CrossEntropyLoss


class WeightedGradAscent(UnlearnTrainer):
    def __init__(self, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        outputs = model(**forget_inputs)

        labels = forget_inputs["labels"].to(outputs.logits.device)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        weights = (- lm_loss).exp().detach() ** self.alpha

        loss = -(weights * lm_loss)[shift_labels.view(-1)!=-100].mean()
        return (loss, outputs) if return_outputs else loss
