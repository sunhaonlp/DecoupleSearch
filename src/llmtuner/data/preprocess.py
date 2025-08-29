import copy
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Tuple

from ..extras.constants import IGNORE_INDEX
from ..extras.logging import get_logger
from .utils import Role, preprocess_messages
import json
import torch

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer

    from ..hparams import DataArguments
    from .template import Template


logger = get_logger(__name__)


def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer", data_args: "DataArguments"
) -> Dict[str, List[List[int]]]:
    # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
    text_examples = [messages[0]["content"] + tokenizer.eos_token for messages in examples["prompt"]]
    if not data_args.packing:
        return tokenizer(text_examples, add_special_tokens=False, max_length=data_args.cutoff_len)

    tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
    concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    block_size = data_args.cutoff_len
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    total_length = (total_length // block_size) * block_size
    # split by chunks of cutoff_len
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    if data_args.template == "gemma":
        for i in range(len(result["input_ids"])):
            result["input_ids"][i][0] = tokenizer.bos_token_id

    return result

from copy import deepcopy

def preprocess_value_dataset_hierarchical(example, tokenizer, template, data_args):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], 'Q_thought': [], 'Q_search': []}
    all_messages = example['messages']
    thought_id, search_id = tokenizer.encode('<thought>')[0], tokenizer.encode('<search>')[0]
    for messages in all_messages:
        question_message = [messages[1]]+  [{"role": 'assistant', 'content': ""}]
        input_ids, _ = template.encode_oneturn(tokenizer, question_message, template.default_system, None)

        labels = deepcopy(input_ids)

        Q_thought = [IGNORE_INDEX] * len(input_ids)
        Q_search = [IGNORE_INDEX] * len(input_ids)
        for index, message in enumerate(messages[2:]):
            sub_message = [{"role": 'user', 'content': ""}] + [{"role": message['role'].replace('function', 'user'), 'content': message['content'].strip()}]
            _, sub_response_ids = template.encode_oneturn(tokenizer, sub_message, template.default_system, None)

            if index % 2 == 0:
                sub_response_ids += tokenizer.encode(template.format_separator.apply()[0])

            input_ids += sub_response_ids
            if message['q_value_search'] != None:
                Q_thought += [IGNORE_INDEX] * (len(sub_response_ids))
                value_index = sub_response_ids.index(search_id)
                Q_search += [IGNORE_INDEX] * value_index + [float(message['q_value_search'])] + [IGNORE_INDEX] * (len(sub_response_ids)-value_index-1)
                labels += [IGNORE_INDEX] * len(sub_response_ids)
            elif message['q_value_thought'] != None:
                Q_search += [IGNORE_INDEX] * (len(sub_response_ids))
                value_index = sub_response_ids.index(thought_id)
                Q_thought += [IGNORE_INDEX] * value_index + [float(message['q_value_thought'])] + [IGNORE_INDEX] * (len(sub_response_ids)-value_index-1)
                labels += sub_response_ids

            if len(input_ids) > data_args.cutoff_len:
                break

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["Q_thought"].append(Q_thought)
        model_inputs["Q_search"].append(Q_search)

        if messages[-1]['q_value_thought'] < 0.7:
            model_inputs["labels"].append([IGNORE_INDEX] * len(labels))
        else:
            model_inputs["labels"].append(labels)

        if len(labels) != len(input_ids):
            print('find!')

    return model_inputs


import numpy as np

def calculate_all_percentiles(data):
    # 将列表转换为 numpy 数组以方便计算
    sorted_data = np.sort(data)

    results = {}
    for percentile in range(101):  # 从0到100的百分位
        index = int(np.ceil((percentile / 100) * len(sorted_data))) - 1
        index = max(index, 0)  # 保证索引不小于0
        results[percentile] = sorted_data[index]

    return results

def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    input_ids, labels = [], []
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            continue

        messages = examples["prompt"][i] + examples["response"][i]
        for source_ids, target_ids in template.encode_multiturn(
            tokenizer, messages, examples["system"][i], examples["tools"][i]
        ):
            if data_args.train_on_prompt:
                source_mask = source_ids
            elif len(input_ids) != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    total_length = len(input_ids)
    block_size = data_args.cutoff_len
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    total_length = (total_length // block_size) * block_size
    # split by chunks of cutoff_len
    for i in range(0, total_length, block_size):
        if not all(label == IGNORE_INDEX for label in labels[i : i + block_size]):
            model_inputs["input_ids"].append(input_ids[i : i + block_size])
            model_inputs["attention_mask"].append([1] * block_size)
            model_inputs["labels"].append(labels[i : i + block_size])

    return model_inputs


def preprocess_unsupervised_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X` and labels with format `Y <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1:
            continue

        if len(examples["response"][i]) == 1:
            messages = examples["prompt"][i] + examples["response"][i]
        else:
            messages = examples["prompt"][i] + [{"role": Role.ASSISTANT.value, "content": ""}]

        input_ids, labels = template.encode_oneturn(
            tokenizer,
            messages,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )

        if template.efficient_eos:
            labels += [tokenizer.eos_token_id]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    return model_inputs


def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) < 2:
            continue

        chosen_messages = examples["prompt"][i] + [examples["response"][i][0]]
        rejected_messages = examples["prompt"][i] + [examples["response"][i][1]]
        prompt_ids, chosen_ids = template.encode_oneturn(
            tokenizer,
            chosen_messages,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )
        _, rejected_ids = template.encode_oneturn(
            tokenizer,
            rejected_messages,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )

        if template.efficient_eos:
            chosen_ids += [tokenizer.eos_token_id]
            rejected_ids += [tokenizer.eos_token_id]

        model_inputs["prompt_ids"].append(prompt_ids)
        model_inputs["chosen_ids"].append(chosen_ids)
        model_inputs["rejected_ids"].append(rejected_ids)

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print(
        "labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        )
    )


def print_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("prompt_ids:\n{}".format(example["prompt_ids"]))
    print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"], skip_special_tokens=False)))
    print("chosen_ids:\n{}".format(example["chosen_ids"]))
    print("chosen:\n{}".format(tokenizer.decode(example["chosen_ids"], skip_special_tokens=False)))
    print("rejected_ids:\n{}".format(example["rejected_ids"]))
    print("rejected:\n{}".format(tokenizer.decode(example["rejected_ids"], skip_special_tokens=False)))


def print_unsupervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))


def get_preprocess_and_print_func(
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo"],
) -> Tuple[Callable, Callable]:
    if stage == "pt":
        preprocess_func = partial(preprocess_pretrain_dataset, tokenizer=tokenizer, data_args=data_args)
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)
    elif stage == "sft" and not training_args.predict_with_generate:
        preprocess_func = partial(
            preprocess_value_dataset_hierarchical, tokenizer=tokenizer, template=template, data_args=data_args
        )

        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)

    elif stage == "rm":
        preprocess_func = partial(
            preprocess_pairwise_dataset, tokenizer=tokenizer, template=template, data_args=data_args
        )
        print_function = partial(print_pairwise_dataset_example, tokenizer=tokenizer)
    else:
        preprocess_func = partial(
            preprocess_unsupervised_dataset, tokenizer=tokenizer, template=template, data_args=data_args
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)

    return preprocess_func, print_function