from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import compute_kl_divergence
from torch.nn import CrossEntropyLoss

class SatImp(UnlearnTrainer):
    def __init__(self, beta=0.3, beta_param=0.3, hyper_param=0.3, alpha=1.0, gamma=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.beta_param = beta_param
        self.hyper_param = hyper_param
        self.alpha = alpha
        self.gamma = gamma
        self.retain_loss_type = "NLL"

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

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.hyper_param != 'None':
            self.beta = self.beta_param
        else:
            self.beta = 5
        
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        outputs = model(**forget_inputs)

        # retain_inputs = inputs["retain"]
        # retain_inputs = {
        #     "input_ids": retain_inputs["input_ids"],
        #     "attention_mask": retain_inputs["attention_mask"],
        #     "labels": retain_inputs["labels"],
        # }
        # retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        labels = forget_inputs["labels"].to(outputs.logits.device)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = CrossEntropyLoss(ignore_index= -100, reduction = 'none')(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        weight_part = (- lm_loss).exp().detach() ** self.hyper_param
        weight_ce = (1-(- lm_loss).exp().detach()) ** self.beta
        loss = -((weight_ce * weight_part) * lm_loss)[shift_labels.view(-1)!=-100].mean()

        loss = self.gamma * loss# + self.alpha * retain_loss
        return (loss, outputs) if return_outputs else loss
