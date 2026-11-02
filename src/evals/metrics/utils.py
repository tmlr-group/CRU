from typing import List
from tqdm import tqdm
from rouge_score import rouge_scorer
from collections import defaultdict
from omegaconf import OmegaConf
import numpy as np
import scipy as sc
from torch import nn
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, PreTrainedTokenizer
from data.utils import IGNORE_INDEX
import warnings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os


def dict_transpose(evals):
    """Transpose a nested dictionary structure to group statistics by item indices."""
    # evals looks like {iidx0: {idx453: {prob: 0.1, loss: 1}},
    #                   iidx1: {idx453: {prob: 0.2, loss: 2}}}
    # multiple answers indexed by intra_item_idx, then item_idx
    # invert the dict, put outermost iidx deepest inside
    # after dict transpose looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}
    all_iidxs = list(evals.keys())
    all_idxs = list(evals[all_iidxs[0]].keys())
    all_stat_names = list(evals[all_iidxs[0]][all_idxs[0]].keys())
    evals = {
        idx: {
            stat: [evals[iidx][idx][stat] for iidx in all_iidxs]
            for stat in all_stat_names
        }
        for idx in all_idxs
    }
    return evals


def aggregate_to_1D(x):
    return np.mean(x, axis=tuple(range(1, x.ndim)))


def get_forget_quality(model_tr, reference_tr):
    test_res = sc.stats.ks_2samp(1 / (model_tr + 1e-10), 1 / (reference_tr + 1e-10))
    return {"agg_value": test_res.pvalue}


def run_batchwise_evals(model, dataloader, batch_eval_fn, batch_eval_fn_args, eval_msg):
    """Run batch-wise evaluations on a dataset using a specified evaluation function. Handles
    multi-answer datasets by organizing evaluations by answer indices and aggregating results."""
    evals = defaultdict(dict)
    for batch in tqdm(dataloader, desc=eval_msg, total=len(dataloader)):
        # if data arrives in normal format we convert the batch to multiple answer-style
        # like in tofu_perturbed by adding a fake intra_item_index
        if "input_ids" in batch:
            batch = {"0": batch}
        # Assume batch like {"0": {"input_ids": [[]]..., "index": [453, 454..]},
        #                    "1": {"input_ids": [[]]..., "index": [453, 454..]}..}
        assert isinstance(next(iter(batch.values())), dict) and "input_ids" in next(
            iter(batch.values())
        )
        for intra_item_idx, mini_batch in batch.items():
            data_indices = (
                mini_batch.pop("index").cpu().numpy().tolist()
            )  # data item indices
            batch_evals = batch_eval_fn(
                model=model, batch=mini_batch, **batch_eval_fn_args
            )
            indexwise_batch_evals = dict(zip(data_indices, batch_evals))
            assert not (
                evals[intra_item_idx].keys() & indexwise_batch_evals.keys()
            ), "Data indices repeated while iterating dataloader"
            evals[intra_item_idx] |= indexwise_batch_evals
    # evals looks like {iidx0: {idx453: {prob: 0.1, loss: 1}},
    #                   iidx1: {idx453: {prob: 0.2, loss: 2}}}
    if len(evals) == 1:  # normal single answer dataset, no need for list
        evals = next(iter(evals.values()))
    else:
        # for each index return a dict with all intra_item_idx values in list
        # after dict transpose looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}}
        evals = dict_transpose(evals)
    print("Evaluated", len(evals), "examples")
    return evals


def evaluate_probability(model, batch):
    """Evaluate model probabilities and average token-level loss for a given batch."""
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    labels = batch["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    # agg loss across tokens
    losses = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    num_token_gt = (batch["labels"] != IGNORE_INDEX).sum(-1)
    avg_losses = losses / num_token_gt
    normalized_probs = torch.exp(-avg_losses)

    avg_losses = avg_losses.cpu().numpy().tolist()
    normalized_probs = normalized_probs.cpu().numpy().tolist()
    return [
        {"prob": prob, "avg_loss": avg_loss}
        for prob, avg_loss in zip(normalized_probs, avg_losses)
    ]


def tokenwise_logprobs(model, batch, grad=False, return_labels=False):
    """
    Compute token-wise next token prediction logprobs for all labeled tokens for each sample in a batch.
    `grad` decides whether gradients are turned on
    Returns
    log_probs_batch (List[Tensor]): Tensors of size seq_len where seq_len is length of labeled tokens
    labels_batch (List[Tensor]): List of tensors of length N. Returned only if return_labels is True
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}

    model.train(mode=grad)
    with torch.set_grad_enabled(grad):
        output = model(**batch)

    logits = output.logits
    bsz, seq_len, V = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1, :]
    # ^ we don't predict next token for last token, bsz x seq_len-1 x V
    next_tokens = batch["input_ids"][:, 1:].unsqueeze(-1)  # bsz x seq_len-1 x 1
    target_log_probs = torch.gather(log_probs, dim=2, index=next_tokens).squeeze(-1)
    log_probs_batch = []
    labels_batch = []
    for i in range(bsz):
        labels = batch["labels"][i][:-1]
        # only focus on tokens which have loss on them (i.e. used in labels)
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        num_actual_tokens = actual_indices.numel()
        if num_actual_tokens == 0:
            log_probs_batch.append(torch.tensor([0.0], device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            warnings.warn(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        log_probs_batch.append(target_log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])

    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch


def tokenwise_vocab_logprobs(model, batch, grad=False, return_labels=False):
    """Get vocabulary-wise log probabilities for each token in the sequence.

    Returns:
        log_probs_batch (List[Tensor]): List of tensors of shape (N, V) containing log probabilities
        for each sequence, where N is the length of labeled tokens and V is vocab size.
        labels_batch (List[Tensor]): List of tensors of length N. Returned only if return_labels is True
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    model.train(mode=grad)
    with torch.set_grad_enabled(grad):
        output = model(**batch)

    logits = output.logits
    bsz, seq_len, V = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[
        :, :-1, :
    ]  # Don't predict for last token

    # Process each sequence in batch separately
    log_probs_batch = []
    labels_batch = []
    for i in range(bsz):
        labels = batch["labels"][i][:-1]
        # Only include positions that have labels
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0]
        if len(actual_indices) == 0:
            log_probs_batch.append(torch.zeros(1, V, device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            warnings.warn(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        # Return full distribution for each position: shape (N, V)
        log_probs_batch.append(log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])

    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch


class MultiTokenEOSCriteria(StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence. Stopping Criteria forked
    and modified from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/27924d77953491f66a038a09892807065e469358/lm_eval/models/utils.py#L208)"""

    def __init__(
        self,
        sequence: str,
        tokenizer: PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> StoppingCriteriaList:
    return StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )


def eval_text_similarity(model, tokenizer, batch, generation_args):
    """Evaluate text similarity between model-generated outputs and ground truth using ROUGE scores."""

    def eval_rouge_recall_batch(gen_outputs, ground_truths):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        evals = []
        for gen, gt in zip(gen_outputs, ground_truths):
            rouge_scores = scorer.score(gt, gen)
            evals.append(
                {
                    "rouge1_recall": rouge_scores["rouge1"].recall,
                    "rougeL_f1": rouge_scores["rougeL"].fmeasure,
                    "rougeL_recall": rouge_scores["rougeL"].recall,
                }
            )
        return evals

    batch = {k: v.to(model.device) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    input_texts = tokenizer.batch_decode(
        input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    tokens = [label[label != IGNORE_INDEX] for label in labels]
    full_texts = tokenizer.batch_decode(
        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    ground_truths = [
        full_text.replace(input_text, "").strip()
        for input_text, full_text in zip(input_texts, full_texts)
    ]

    attention_mask = batch["attention_mask"]

    # convert to a simple dict from DictConfig
    generation_args = OmegaConf.to_container(generation_args, resolve=True)
    stopwords = generation_args.pop("stopwords", None)
    if stopwords is not None:
        assert isinstance(stopwords, list)
        sc = stop_sequences_criteria(
            tokenizer, stopwords, input_ids.shape[1], input_ids.shape[0]
        )
        generation_args["stopping_criteria"] = sc
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **generation_args,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_texts = tokenizer.batch_decode(
        output[:, input_ids.shape[-1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # cut off at stopwords
    if stopwords is None:
        stopwords = []
    stopwords = [tokenizer.decode([tokenizer.eos_token_id])] + stopwords
    for i in range(len(gen_texts)):
        raw_text = gen_texts[i]
        for word in stopwords:
            if word and word in raw_text:
                raw_text = raw_text.split(word)[0]
        raw_text = raw_text.strip()
        gen_texts[i] = raw_text

    scores = eval_rouge_recall_batch(gen_texts, ground_truths)
    scores = [
        {
            **rouge_evals,
            "input": input_text,
            "ground_truth": ground_truth,
            "generation": gen_text,
        }
        for rouge_evals, input_text, ground_truth, gen_text in zip(
            scores, input_texts, ground_truths, gen_texts
        )
    ]
    return scores


def extract_target_texts_from_processed_data(tokenizer, batch):
    """Extract and detokenize text from activated positions in the batch."""
    labels = batch["labels"]
    labels = [elem[elem != -100] for elem in labels]
    texts = [
        tokenizer.decode(elem.tolist(), skip_special_tokens=True) for elem in labels
    ]
    return texts


def get_hidden_states(model, dataloader, device):
    model.eval()
    all_hidden_states = None
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Batches", total=len(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch, output_hidden_states=True)
            # pooled = [layer.mean(dim=1).cpu().numpy() for layer in output.hidden_states]
            # pooled = [(layer[:, 0, :] + 1e-12).cpu().numpy() for layer in output.hidden_states]
            # pooled = [(layer[:, -1, :] + 1e-12).cpu().numpy() for layer in output.hidden_states]
            pooled = [layer.max(dim=1).cpu().numpy() for layer in output.hidden_states]
            if all_hidden_states is None:
                all_hidden_states = [p for p in pooled]
            else:
                for i in range(len(pooled)):
                    all_hidden_states[i] = np.concatenate([all_hidden_states[i], pooled[i]], axis=0)
    return all_hidden_states


def run_tsne_and_save(hidden_states_forget, hidden_states_retain, out_dir, model_name):
    os.makedirs(out_dir, exist_ok=True)
    num_layers = len(hidden_states_forget)
    for i in tqdm(range(num_layers), desc="Processing Layers", total=num_layers):
        X = np.concatenate([hidden_states_forget[i], hidden_states_retain[i]], axis=0)
        y = np.array([0]*len(hidden_states_forget[i]) + [1]*len(hidden_states_retain[i]))
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        plt.figure(figsize=(6,6))
        plt.scatter(X_tsne[y==0,0], X_tsne[y==0,1], c='red', label='forget', s=5)
        plt.scatter(X_tsne[y==1,0], X_tsne[y==1,1], c='blue', label='retain', s=5)
        plt.legend()
        plt.title(f"{model_name} Layer {i+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_name}_layer_{i+1}.png"))
        plt.close()

def get_batch_hidden_states(model, batch, device):
    model.eval()
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs, output_hidden_states=True)
        pooled = [(layer[:, -1, :] + 1e-12).cpu() for layer in outputs.hidden_states]
    return pooled

def replace_layer(model, other_model, layer_idx):
    model.model.layers[layer_idx] = other_model.model.layers[layer_idx].to(model.device)
    return model

def compute_layerwise_cos_sim(original_model, modified_model, dataloader, device, num_layers):
    cos_sim_per_layer = [0.0 for _ in range(num_layers)]
    total_samples = 0

    original_model.eval()
    modified_model.eval()
    with torch.no_grad():
        original_layers = [layer for layer in original_model.model.layers]

        for batch in tqdm(dataloader, desc="Processing Batches", total=len(dataloader)):
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            outputs = original_model(**inputs, output_hidden_states=True)
            original_hiddens = [layer[:, -1, :].cpu() for layer in outputs.hidden_states]

            batch_size = original_hiddens[0].shape[0]
            total_samples += batch_size

            for i in range(num_layers):
                original_model.model.layers[i] = modified_model.model.layers[i].to(device)

                outputs_mod = original_model(**inputs, output_hidden_states=True)
                modified_hiddens = [layer[:, -1, :].cpu() for layer in outputs_mod.hidden_states]

                original_model.model.layers[i] = original_layers[i].to(device)

                cos_sim = torch.nn.functional.cosine_similarity(
                    original_hiddens[i+1], modified_hiddens[i+1], dim=1
                )
                cos_sim_per_layer[i] += cos_sim.sum().item()

    for i in range(num_layers):
        cos_sim_per_layer[i] /= total_samples
    return cos_sim_per_layer

def compute_layerwise_cos_sim_with_window(original_model, modified_model, dataloader, device, num_layers, window_size=4, step_size=1):
    cos_sim_per_layer = [0.0 for _ in range(num_layers)]
    count_per_layer = [0 for _ in range(num_layers)]

    original_model.eval()
    modified_model.eval()

    with torch.no_grad():
        original_layers = [layer for layer in original_model.model.layers]

        for batch in tqdm(dataloader, desc="Processing Batches", total=len(dataloader)):
            inputs = {k: v.to(device) for k, v in batch.items()}

            outputs = original_model(**inputs, output_hidden_states=True)
            original_hiddens = [layer[:, -1, :].cpu() for layer in outputs.hidden_states]

            batch_size = original_hiddens[0].shape[0]

            for start in range(0, num_layers - window_size + 1, step_size):
                end = start + window_size
                window_indices = list(range(start, end))

                original_window_layers = [original_model.model.layers[i] for i in window_indices]

                for i in window_indices:
                    original_model.model.layers[i] = modified_model.model.layers[i].to(device)

                outputs_mod = original_model(**inputs, output_hidden_states=True)
                modified_hiddens = [layer[:, -1, :].cpu() for layer in outputs_mod.hidden_states]

                for i, layer in zip(window_indices, original_window_layers):
                    original_model.model.layers[i] = layer.to(device)

                representative_model_idx = start + window_size - 1
                representative_hidden_idx = start + window_size

                cos_sim = torch.nn.functional.cosine_similarity(
                    original_hiddens[representative_hidden_idx],
                    modified_hiddens[representative_hidden_idx],
                    dim=1
                )

                cos_sim_per_layer[representative_model_idx] += cos_sim.sum().item()
                count_per_layer[representative_model_idx] += batch_size

    for i in range(num_layers):
        if count_per_layer[i] > 0:
            cos_sim_per_layer[i] /= count_per_layer[i]
        else:
            cos_sim_per_layer[i] = 0.0

    return cos_sim_per_layer

def plot_cos_sim_curve(cos_sim_per_layer, model_name, window_size=4, step_size=1):
    plt.figure(figsize=(10, 10))

    num_layers = len(cos_sim_per_layer)
    window_labels = []
    for start in range(0, num_layers - window_size + 1, step_size):
        end = start + window_size
        window_labels.append(f"{start}-{end-1}")

    window_sim = []
    for start in range(0, num_layers - window_size + 1, step_size):
        end = start + window_size
        window_indices = list(range(start, end))
        last_layer = window_indices[-1]
        window_sim.append(cos_sim_per_layer[last_layer])
    
    plt.plot(range(len(window_sim)), window_sim, marker='o', color='blue', linestyle='-', linewidth=2)
    
    plt.xticks(ticks=range(len(window_labels)), labels=window_labels, rotation=45, fontsize=12)

    plt.xlabel("Window Range (Layer Start - End)", fontsize=14)
    plt.ylabel("Average Cosine Similarity", fontsize=14)
    plt.title(f"{model_name} - Layer-wise Cosine Similarity (Window Size: {window_size})", fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"hidden/{model_name}_cos_sim_window_size_{window_size}_step_size_{step_size}.pdf", dpi=300, bbox_inches='tight')
    plt.close()