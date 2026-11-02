from pathlib import Path as _CRUPath
import sys as _cru_sys
_cru_root = _CRUPath(__file__).resolve().parents[2]
if str(_cru_root / "src") not in _cru_sys.path:
    _cru_sys.path.insert(0, str(_cru_root / "src"))
from cru_paths import resolve_input_path

import torch
import datasets
import numpy as np
from typing import List, Dict, Any, Union

IGNORE_INDEX = -100  # TODO put in common constants


def load_hf_dataset(path, **kwargs):
    import os
    split_root = os.environ.get("CRU_EVAL_DATA_ROOT")
    heldout_names = {"forget04", "forget01", "forget04_perturbed", "forget01_perturbed",
                     "holdout04", "holdout01", "retain_perturbed_320", "retain_perturbed_80"}
    if path == "locuslab/TOFU" and split_root and kwargs.get("name") in heldout_names:
        local_file = _CRUPath(split_root) / f"{kwargs['name']}.json"
        if not local_file.is_file():
            raise FileNotFoundError(f"Missing held-out split: {local_file}; run prepare_tofu_data.py")
        local_kwargs = dict(kwargs)
        local_kwargs.pop("name")
        return datasets.load_dataset("json", data_files=str(local_file), **local_kwargs)
    local_root = os.environ.get("CRU_TOFU_DATA_ROOT")
    if path == "locuslab/TOFU" and local_root:
        local_file = _CRUPath(local_root) / f"{kwargs.get('name')}.json"
        if not local_file.is_file():
            raise FileNotFoundError(f"Missing staged TOFU split: {local_file}")
        local_kwargs = dict(kwargs)
        local_kwargs.pop("name", None)
        local_kwargs["data_files"] = str(local_file)
        return datasets.load_dataset("json", **local_kwargs)
    path = resolve_input_path(path)
    if "data_files" in kwargs:
        def resolve_files(value):
            if isinstance(value, dict):
                return {key: resolve_files(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [resolve_files(item) for item in value]
            return resolve_input_path(value)
        kwargs["data_files"] = resolve_files(kwargs["data_files"])
    if kwargs.get('name') not in ['forget04', 'forget04_perturbed', 'holdout04', 'retain_perturbed_80', 'retain_perturbed_320', 'knowmem80', 'privleak80', 'verbmem80', 'knowmem20', 'privleak20', 'verbmem20', 'knowmem80_books', 'privleak80_books', 'verbmem80_books', 'knowmem20_books', 'privleak20_books', 'verbmem20_books', 'forget01_t', 'forget02_t', 'forget03_t', 'forget04_t', 'forget01_v', 'forget02_v', 'forget03_v', 'forget04_v', 'forget01_t_perturbed', 'forget02_t_perturbed', 'forget03_t_perturbed', 'forget04_t_perturbed', 'forget01_v_perturbed', 'forget02_v_perturbed', 'forget03_v_perturbed', 'forget04_v_perturbed', 'holdout01_t', 'holdout02_t', 'holdout03_t', 'holdout04_t', 'holdout01_v', 'holdout02_v', 'holdout03_v', 'holdout04_v', 'retain01_t_perturbed', 'retain02_t_perturbed', 'retain03_t_perturbed', 'retain04_t_perturbed', 'retain01_v_perturbed', 'retain02_v_perturbed', 'retain03_v_perturbed', 'retain04_v_perturbed', 'forget05_alt', 'forget_blk0_v', 'forget_blk0_t', 'forget_blk0_v_perturbed', 'forget_blk0_t_perturbed', 'holdout_blk0_v', 'holdout_blk0_t', 'forget_blk1_v', 'forget_blk1_t', 'forget_blk1_v_perturbed', 'forget_blk1_t_perturbed', 'holdout_blk1_v', 'holdout_blk1_t', 'forget_blk2_v', 'forget_blk2_t', 'forget_blk2_v_perturbed', 'forget_blk2_t_perturbed', 'holdout_blk2_v', 'holdout_blk2_t', 'forget_blk3_v', 'forget_blk3_t', 'forget_blk3_v_perturbed', 'forget_blk3_t_perturbed', 'holdout_blk3_v', 'holdout_blk3_t']:
        dataset = datasets.load_dataset(path, **kwargs)
    else:
        dataset = datasets.load_from_disk(resolve_input_path(f"datasets/{kwargs['name']}/{kwargs['split']}"))
    return dataset


def preprocess_chat_instance(
    tokenizer,
    template_config: Dict[str, Any],
    prompt_msgs: Union[List[str], str],
    response_msgs: Union[List[str], str],
    max_length: int,
    predict_with_generate: bool = False,
) -> Dict[str, torch.Tensor]:
    """Preprocesses a chat instance for training or generation.
    When in training, both the returned `input_ids` and `labels` cover the entire conversation.
    `input_ids` has no padding, and `labels` assign `IGNORE_INDEX` to tokens where loss is not computed (i.e. all tokens except the final response message).
    When in generation, `input_ids` are returned only up to the last user prompt, excluding the assistant's response. The `labels` returned are the same as during training.
    `attention_mask` is always 1 over the full `input_ids` token sequence.

    `prompt_msgs` and `response_msgs` are lists where, except for the last pair, all
    corresponding pairs are in-context examples. When they are a string and not
    a list, there are no in-context examples.

    Args:
        tokenizer: Tokenizer to apply on text
        template_config (Dict[str, Any]): Configuration for the chat template (comes from model-specific config).
        prompt_msgs (Union[List[str], str]): List of prompt messages or a single prompt message string.
        response_msgs (Union[List[str], str]): List of response messages or a single response message string.
        max_length (int): Maximum sequence length after tokenization.
        predict_with_generate (bool, optional): Whether to prepare inputs for generation.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'labels', and 'attention_mask' tensors for model input.
    """
    assert len(prompt_msgs) == len(response_msgs)
    if isinstance(prompt_msgs, str):
        assert isinstance(response_msgs, str)
        prompt_msgs, response_msgs = [prompt_msgs], [response_msgs]

    if template_config["apply_chat_template"]:
        chat = []
        system_prompt = template_config.get("system_prompt", None)
        if system_prompt:
            chat += [{"role": "system", "content": system_prompt}]
        for prompt, response in zip(prompt_msgs, response_msgs):
            chat += [{"role": "user", "content": prompt}]
            chat += [{"role": "assistant", "content": response}]
        chat_ids = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=False
        )
        # all except last response are in-context examples
        wrapped_prompt = tokenizer.apply_chat_template(
            chat[:-1], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.apply_chat_template(
            chat[:-1], tokenize=True, add_generation_prompt=True
        )
    else:
        wrapped_prompt = ""
        system_prompt_with_special_tokens = template_config.get(
            "system_prompt_with_special_tokens", None
        )
        if system_prompt_with_special_tokens:
            wrapped_prompt += system_prompt_with_special_tokens
        # add in-context examples
        n_few_shot = len(prompt_msgs) - 1
        for i in range(n_few_shot):
            fs_prompt, fs_response = prompt_msgs[i], response_msgs[i]
            wrapped_prompt += (
                template_config["user_start_tag"]
                + fs_prompt
                + template_config["user_end_tag"]
                + template_config["asst_start_tag"]
                + fs_response
                + template_config["asst_end_tag"]
            )

        # add actual example
        final_prompt, final_response = prompt_msgs[-1], response_msgs[-1]
        wrapped_prompt += (
            template_config["user_start_tag"]
            + final_prompt
            + template_config["user_end_tag"]
            + template_config["asst_start_tag"]
        )
        chat_ids = tokenizer(
            wrapped_prompt + final_response,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )["input_ids"]

        prompt_ids = tokenizer(
            wrapped_prompt,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )["input_ids"]

    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids += [tokenizer.eos_token_id]

    len_matched = len(prompt_ids)

    item = {}
    if predict_with_generate:
        item["input_ids"] = prompt_ids
        labels = chat_ids  # contains the entire conversation
    else:
        item["input_ids"] = chat_ids
        labels = [IGNORE_INDEX] * len_matched + chat_ids[len_matched:]
    item["labels"] = labels
    item["attention_mask"] = [1] * len(item["input_ids"])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item


def preprocess_pretraining_instance(
    tokenizer,
    prefix: str,
    text_content: str,
    max_length: int,
    predict_with_generate: bool = False,
    insert_space: bool = False,
) -> Dict[str, torch.Tensor]:
    """Preprocesses a pretraining instance for training or generation.
    When in training, both the returned `input_ids` and `labels` are over the entire token sequence. `input_ids` has no padding, `labels` assigns `IGNORE_INDEX` to ignore all tokens that we don't compute loss over (i.e. the the 0th index token, all prefix tokens)
    When in generation, `input_ids` are returned only until the prefix portion. The `labels` returned are the same as during training.
    `attention_mask` is always 1 over the full input token sequence.
    Args:
        tokenizer: Tokenizer to apply on text
        prefix (str): The prefix string to prepend to the content.
        text_content (str): The main text content (following the prefix) to be tokenized.
        max_length (int): Maximum text content length after tokenization.
        predict_with_generate (bool, optional): Whether to prepare inputs for generation.
        insert_space (bool, optional): Whether to insert a space between prefix and content.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'labels', and 'attention_mask' tensors for model input.
    """
    full_seq_ids = tokenizer(
        prefix + (" " if insert_space else "") + text_content, add_special_tokens=True
    )["input_ids"]
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    prefix_len = len(prefix_ids)
    full_seq_ids = full_seq_ids[: prefix_len + max_length]  # manual truncation

    len_matched = prefix_len
    if len_matched == 0:  # never give loss on index 0, when prefix is empty
        len_matched = 1
    labels = [IGNORE_INDEX] * len_matched + full_seq_ids[len_matched:]
    item = {}
    if predict_with_generate:
        item["input_ids"] = prefix_ids
    else:
        item["input_ids"] = full_seq_ids
    item["labels"] = labels
    item["attention_mask"] = [1] * len(item["input_ids"])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item


def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column("index", indexing)
    return dataset
