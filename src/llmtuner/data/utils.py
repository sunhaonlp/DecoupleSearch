import hashlib
from enum import Enum, unique
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from datasets import concatenate_datasets, interleave_datasets

from ..extras.logging import get_logger

import math
from qwen_agent.utils.utils import (
    format_as_multimodal_message, has_chinese_messages,
    build_text_completion_prompt, format_as_text_message, merge_generate_cfgs
)
from qwen_agent.settings import DEFAULT_MAX_INPUT_TOKENS
from qwen_agent.llm.schema import (
    ASSISTANT, FUNCTION, USER, ContentItem, Message,
    ROLE, CONTENT, SYSTEM, DEFAULT_SYSTEM_MESSAGE
)
from qwen_agent.llm.function_calling import validate_num_fncall_results
from qwen_agent.llm.fncall_prompts.qwen_fncall_prompt import QwenFnCallPrompt
from qwen_agent.llm import get_chat_model
from typing import List, Literal, Optional, Dict


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments

    from llmtuner.hparams import DataArguments


logger = get_logger(__name__)


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


def checksum(data_files: List[str], file_sha1: Optional[str] = None) -> None:
    if file_sha1 is None:
        logger.warning("Checksum failed: missing SHA-1 hash value in dataset_info.json.")
        return

    if len(data_files) != 1:
        logger.warning("Checksum failed: too many files.")
        return

    with open(data_files[0], "rb") as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
        if sha1 != file_sha1:
            logger.warning("Checksum failed: mismatched SHA-1 hash value at {}.".format(data_files[0]))


def infer_max_len(source_len: int, target_len: int, max_len: int, reserved_label_len: int) -> Tuple[int, int]:
    max_target_len = int(max_len * (target_len / (source_len + target_len)))
    max_target_len = max(max_target_len, reserved_label_len)
    max_source_len = max_len - max_target_len
    return max_source_len, max_target_len



def _format_as_text_messages(messages: List[Message]) -> List[Message]:
    for msg in messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                assert item.type == 'text'
        else:
            assert isinstance(msg.content, str)
    messages = [format_as_text_message(msg, add_upload_info=False) for msg in messages]
    return messages

import os,copy,random



def post_process(output: str):
    qwen_agent_llm = get_chat_model({
        'model': 'qwen-max-latest',
        'model_server': 'dashscope',
        'api_key': os.getenv('DASHSCOPE_API_KEY', "sk-603e56d690a04327a1e926a6f4bbfd14"),
    })

    output_message = [Message(ASSISTANT, output)]
    fncall_mode = True
    generate_cfg = {
        "seed": 1,
        "stop": ["✿RESULT✿", "✿RETURN✿"]
    }

    processed_messages = qwen_agent_llm._postprocess_messages(output_message, fncall_mode=fncall_mode,
                                                              generate_cfg=generate_cfg)
    formatted_messages = _format_as_text_messages(messages=processed_messages)
    output_dict = qwen_agent_llm._convert_messages_to_target_type(formatted_messages, 'dict')

    return output_dict


def preprocess_messages(messages: List[Message],
                        extra_generate_cfg: Optional[Dict] = None) -> List[Message]:
    generate_cfg = {
        "seed": 1,
        "stop": ["✿RESULT✿", "✿RETURN✿"],
        "lang": "zh"
    }
    functions = [
        {
            "name_for_human": "web_search",
            "name_for_model": "web_search",
            "description": "Utilize the web search engine to retrieve relevant information based on multiple queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "The search query."
                        },
                        "description": "The list of search queries."
                    }
                },
                "required": [
                    "queries"
                ]
            }
        }
    ]
    qwen_agent_llm = get_chat_model({
        'model': 'qwen-max-latest',
        'model_server': 'dashscope',
        'api_key': os.getenv('DASHSCOPE_API_KEY', "sk-603e56d690a04327a1e926a6f4bbfd14"),
    })

    messages = copy.deepcopy(messages)
    generate_cfg = merge_generate_cfgs(base_generate_cfg=generate_cfg, new_generate_cfg=extra_generate_cfg)

    new_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            new_messages.append(Message(**msg))
        else:
            new_messages.append(msg)
    messages = new_messages

    if 'seed' not in generate_cfg:
        generate_cfg['seed'] = random.randint(0, 2 ** 30)

    lang: Literal['en', 'zh'] = generate_cfg.pop('lang', 'zh' if has_chinese_messages(messages) else 'en')

    if messages[0].role != SYSTEM:
        messages.insert(0, Message(role=SYSTEM, content=DEFAULT_SYSTEM_MESSAGE))

    fncall_mode = bool(functions) if 'function_choice' not in generate_cfg or generate_cfg[
        'function_choice'] != 'none' else False

    messages = qwen_agent_llm._preprocess_messages(messages, lang=lang, generate_cfg=generate_cfg, functions=functions)
    formatted_messages = [format_as_text_message(msg, add_upload_info=False) for msg in messages]
    text_prompt = build_text_completion_prompt(formatted_messages)

    return text_prompt




def merge_dataset(
    all_datasets: List[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning("The samples between different datasets will not be mixed in streaming mode.")
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning("We recommend using `mix_strategy=concat` in non-streaming mode.")
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=training_args.seed,
            stopping_strategy="first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted",
        )
    else:
        raise ValueError("Unknown mixing strategy.")


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"], data_args: "DataArguments", training_args: "Seq2SeqTrainingArguments"
) -> Dict[str, "Dataset"]:
    if training_args.do_train:
        if data_args.val_size > 1e-6:  # Split the dataset
            if data_args.streaming:
                val_set = dataset.take(int(data_args.val_size))
                train_set = dataset.skip(int(data_args.val_size))
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
                return {"train_dataset": train_set, "eval_dataset": val_set}
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                dataset = dataset.train_test_split(test_size=val_size, seed=training_args.seed)
                return {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            if data_args.streaming:
                dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
            return {"train_dataset": dataset}
    else:  # do_eval or do_predict
        return {"eval_dataset": dataset}
