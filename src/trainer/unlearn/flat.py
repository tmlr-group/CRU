import torch
from trainer.unlearn.grad_diff import GradDiff
from torch import nn


class CrossEntropyLossStable(nn.Module):
    def __init__(self, reduction='mean', eps=1e-5):
        super(CrossEntropyLossStable, self).__init__()
        self._name = "Stable Cross Entropy Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def forward(self, outputs, labels):
        return self._nllloss( torch.log( self._softmax(outputs) + self._eps ), labels )
    

class ProbLossStable(nn.Module):
    def __init__(self, reduction='none', eps=1e-5):
        super(ProbLossStable, self).__init__()
        self._name = "Prob Loss"
        self._eps = eps
        self._softmax = nn.Softmax(dim=-1)
        # self._softmax = nn.LogSoftmax(dim=-1)
        # self._nllloss = nn.NLLLoss(reduction='none')
        self._nllloss = nn.NLLLoss(reduction='none', ignore_index=-100)

    def forward(self, outputs, labels):
        return self._nllloss( self._softmax(outputs), labels )
        
def get_contrastive_loss(prob_sum_unlearn, prob_sum_good, div='Total-Variation'):
    
    if div == 'KL':
        def activation(x): return -torch.mean(x)
        
        def conjugate(x): return -torch.mean(torch.exp(x - 1.))

    elif div == 'Reverse-KL':
        def activation(x): return -torch.mean(-torch.exp(x))
        
        def conjugate(x): return -torch.mean(-1. - x)  # remove log

    elif div == 'Jeffrey':
        def activation(x): return -torch.mean(x)
        
        def conjugate(x): return -torch.mean(x + torch.mul(x, x) / 4. + torch.mul(torch.mul(x, x), x) / 16.)

    elif div == 'Squared-Hellinger':
        def activation(x): return -torch.mean(1. - torch.exp(x))
        
        def conjugate(x): return -torch.mean((1. - torch.exp(x)) / (torch.exp(x)))

    elif div == 'Pearson':
        def activation(x): return -torch.mean(x)
        
        def conjugate(x): return -torch.mean(torch.mul(x, x) / 4. + x)

    elif div == 'Neyman':
        def activation(x): return -torch.mean(1. - torch.exp(x))

        def conjugate(x): return -torch.mean(2. - 2. * torch.sqrt(1. - x))

    elif div == 'Jenson-Shannon':
        def activation(x): return -torch.mean(- torch.log(1. + torch.exp(-x))) - torch.log(torch.tensor(2.))

        def conjugate(x): return -torch.mean(x + torch.log(1. + torch.exp(-x))) + torch.log(torch.tensor(2.))

    elif div == 'Total-Variation':
        def activation(x): return -torch.mean(torch.tanh(x) / 2.)
    
        def conjugate(x): return -torch.mean(torch.tanh(x) / 2.)
        
    else:
        raise NotImplementedError("[-] Not Implemented f-divergence %s" % div)

    prob_reg = -prob_sum_good
    loss_regular = activation(prob_reg)
    prob_peer = -prob_sum_unlearn
    loss_peer = conjugate(prob_peer)
    
    loss = loss_regular - loss_peer
    return loss


class FLAT(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        # self.cl_div = 'KL'
        self.cl_div = 'Total-Variation'
        # self.cl_div = 'Pearson'
        # self.cl_div = 'Jenson-Shannon'
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]["original"]
        idk_inputs = inputs["forget"]["alternate"]

        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        idk_inputs = {
            "input_ids": idk_inputs["input_ids"],
            "attention_mask": idk_inputs["attention_mask"],
            "labels": idk_inputs["labels"],
        }

        forget_outputs = model(**forget_inputs)
        idk_outputs = model(**idk_inputs)
            
        losses_unlearn = []
        losses_good = []

        shift_logits = forget_outputs.logits[:, :-1, :]
        shift_labels_unlearn = forget_inputs["labels"][:, 1:]
        shift_logits_good = idk_outputs.logits[:,:-1,:]
        shift_labels_good = idk_inputs["labels"][:,1:]
            
        criterion_prob = ProbLossStable()
       
        for bid in range(forget_inputs["input_ids"].shape[0]):
            loss_unlearn = criterion_prob(shift_logits[bid], shift_labels_unlearn[bid])
            loss_good = criterion_prob(shift_logits_good[bid], shift_labels_good[bid])
            losses_unlearn.append(loss_unlearn)
            losses_good.append(loss_good)

        loss_sum_unlearn = torch.stack(losses_unlearn).mean()
        loss_sum_good = torch.stack(losses_good).mean()
            
        forget_loss = get_contrastive_loss(loss_sum_unlearn, loss_sum_good, self.cl_div)
        print(self.gamma)
        loss = self.gamma * forget_loss
        return (loss, forget_outputs) if return_outputs else loss
