from sympy.physics.units import current
from tqdm import trange, tqdm
import argparse
import wandb
from utils import *
import re
import numpy as np
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
import json
import argparse
import os
import copy
import numpy as np
import random
from tqdm import tqdm
from typing import List, Literal, Optional, Dict
from vllm import LLM, SamplingParams
from vllm.utils import Counter
from collections import Counter

def cut_and_normalize_strs(s):
    if s:
        s = s.strip().lower()
        s = s.split('\n')[0]
        s = s.split('.')[0]
        s = s.split(',')[0]
        if 'answer is' in s:
            s = s.split('answer is')[-1]
        if 'The answer is' in s:
            s = s.split('The answer is')[-1]
        # Cut off the first newline, period, or comma
        truncated_text = re.split(r'[\n.,]', s, 1)[0]

        # Remove punctuation
        no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

        # Remove article
        no_articles = re.sub(r'\b(an|the)\b',
                            '',
                            no_punctuation,
                            flags=re.IGNORECASE)

        # Remove duplicated blank spaces
        cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()
    else:
        cleaned_text = ''
    return cleaned_text

def remove_punctuation(text):
    # 创建一个所有标点符号的集合
    punctuations = string.punctuation

    # 使用 translate() 方法删除标点符号
    no_punct = text.translate(str.maketrans('', '', punctuations))

    return no_punct

def generation_finish(args, custom_model, tokenizer, messages, num_samples=5, max_length=2000, temperature=1.0):
    """
    Generate multiple sequences with associated value heads using high-temperature sampling.

    Args:
        custom_model (nn.Module): The custom model with a value head.
        tokenizer (Tokenizer): The tokenizer corresponding to the model.
        prompt (str): The input prompt to generate from.
        num_samples (int, optional): Number of sequences to generate. Defaults to 5.
        max_length (int, optional): Maximum length of generated sequences. Defaults to 50.
        temperature (float, optional): Sampling temperature. Higher values increase diversity. Defaults to 1.0.
        top_k (int, optional): Top-K filtering parameter. Defaults to 50.
        top_p (float, optional): Top-P (nucleus) filtering parameter. Defaults to 0.95.

    Returns:
        List[Tuple[str, List[float]]]: A list of tuples containing generated text and their corresponding value scores.
    """
    finish_message = [{'content': "✿THOUGHT✿: We now have the necessary information to answer the question. Let's analyze these information and obtain the final answer. ",
                      'role': 'assistant'}]
    input_ids = tokenizer.apply_chat_template(
        messages + finish_message,
        add_generation_prompt=True,
        return_tensors="pt"
    ).tolist()[0][:-2]
    input_ids = torch.tensor([input_ids]).to(f'cuda:{args.base_device}')

    # Use model.generate to obtain sequences
    generated_sequences = custom_model.model.generate(
        input_ids=input_ids,
        max_new_tokens=max_length,
        temperature=temperature,
        num_return_sequences=num_samples,
        do_sample=True  # Enable sampling to increase diversity
    )

    # Decode the generated sequences
    generated_texts = [tokenizer.decode(seq[input_ids.shape[1]:], skip_special_tokens=True) for seq in generated_sequences][0]

    return "✿THOUGHT✿: We now have the necessary information to answer the question. Let's analyze these information and obtain the final answer. " + generated_texts, None

def obtain_values(messages, tokenizer, value_ip, value_port, mode):
    """
    Generate multiple sequences with associated value heads using high-temperature sampling.

    Args:
        custom_model (nn.Module): The custom model with a value head.
        tokenizer (Tokenizer): The tokenizer corresponding to the model.
        prompt (str): The input prompt to generate from.
        num_samples (int, optional): Number of sequences to generate. Defaults to 5.
        max_length (int, optional): Maximum length of generated sequences. Defaults to 50.
        temperature (float, optional): Sampling temperature. Higher values increase diversity. Defaults to 1.0.
        top_k (int, optional): Top-K filtering parameter. Defaults to 50.
        top_p (float, optional): Top-P (nucleus) filtering parameter. Defaults to 0.95.

    Returns:
        List[Tuple[str, List[float]]]: A list of tuples containing generated text and their corresponding value scores.
    """
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    input_prompt = tokenizer.decode(input_ids[0])
    input_ids = tokenizer.encode(input_prompt)

    if mode == 'values_thought':
        thought_id = tokenizer.encode('<thought>')[0]
        for i in range(len(input_ids)):
            if input_ids[i] == thought_id:
                symbol_index = i
        input_ids = input_ids[:symbol_index+1]
    elif mode == 'values_search':
        search_id = tokenizer.encode('<search>')[0]
        for i in range(len(input_ids)):
            if input_ids[i] == search_id:
                symbol_index = i
        input_ids = input_ids[:symbol_index+1]

    data['input_ids'] = input_ids
    while(1):
        try:
            response = requests.post(f"http://{value_ip}:{value_port}/statistic", json=data)
            json_data = response.json()
            return json_data[mode][0]
        except Exception as e:
            time.sleep(1)
            pass

def remove_duplicate(generated_texts):
    generated_texts_new, values_list_new = [], []
    for i in range(len(generated_texts)):
        if generated_texts[i] not in generated_texts_new:
            generated_texts_new.append(generated_texts[i])
    return generated_texts_new, None

def parallel_search(args, queries):
    all_search_result = [['No Information Found', 1] for _ in range(len(queries))]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(retrieve_from_api, args, [query], index) for index, query in enumerate(queries)]
        for future in as_completed(futures):
            try:
                result, index, state = future.result()
                all_search_result[index] = [result, state]
            except Exception as e:
                continue
    return all_search_result

def sbs(args, eval_item, custom_model, tokenizer):
    eval_item["messages"] = [{'content': eval_item['question'], 'role': 'user'}]
    eval_item["value"] = [0]
    eval_item["state"] = False
    S_previous = [eval_item]
    finish_result = []
    record = []
    for layer_index in range(args.beam_depth):
        print(f'{layer_index}-th iteration')
        S_thought, S_search = [], []
        ## Expand thought
        for elem in S_previous:
            generated_texts = generation_sever_hierarchical(tokenizer, elem['messages'], args.policy_port, args.policy_ip,  args.expand_num_thought, args.max_length, args.temperature)
            for tmp in generated_texts:
                new_message = elem["messages"] + [{"role": "assistant", "content": tmp}]
                value = obtain_values(new_message, tokenizer, args.value_ip, args.value_port, 'values_thought')
                print(f'Thought:{value}')
                new_elem = copy.deepcopy(elem)
                new_elem['messages'] = new_message
                new_elem['value'] = [value]

                if 'search(' in tmp.lower():
                    S_thought.append(new_elem)
                else:
                    finish_result.append(new_elem)

        ## Rank thought
        # print(f'\nExpanded Thought: {len(S_thought)}')
        # print('\n\n'.join([elem['messages'][-1]['content'] for elem in S_thought]))

        S_thought.sort(key=lambda x: x['value'][-1], reverse=True)
        record.append(copy.deepcopy([(elem['messages'][-1], elem['value'][-1]) for elem in S_thought]))
        S_thought = S_thought[:args.beam_size_thought]

        ## Expand Search Query
        queries = []
        for i in range(len(S_thought)):
            generated_texts = generation_sever_hierarchical_complete(tokenizer, S_thought[i]["messages"], args.policy_port, args.policy_ip, args.expand_num_search, args.max_length, args.temperature, args.temperature_search, 'truncate')
            new_message =  copy.deepcopy(S_thought[i])
            for generated_text in generated_texts:
                new_message["messages"][-1]['content'] = truncate_at_last_search(new_message["messages"][-1]['content']) + generated_text.split(')')[0]+ ')'
                queries.append(generated_text.split(')')[0].strip('('))
                S_search.append(copy.deepcopy(new_message))
        all_search_result = parallel_search(args, queries)
        # print(f'\nExpanded Search: {len(S_search)}')

        ## Rank Search Result
        for i in range(len(S_search)):
            search_result, status = all_search_result[i][0], all_search_result[i][1]
            S_search[i]["messages"]  = S_search[i]["messages"] + [{"role": "user", "content": search_result + ' <search>'}]
            value = obtain_values(S_search[i]["messages"], tokenizer, args.value_ip, args.value_port, 'values_search')
            S_search[i]['value'] = [value]
            print(f'Search:{value}')
            S_search[i]['state'] = False

        S_search.sort(key=lambda x: x['value'][-1], reverse=True)
        record.append(copy.deepcopy([(elem['messages'][-2:], elem['value']) for elem in S_search]))

        S_previous = S_search[:args.beam_size_search]
        if len(S_previous) == 0:
            break

    if len(finish_result) == 0:
        finish_result = [{'value': [1], 'reasoning_process': [S_previous[0]['messages'][-3]]}]
    return finish_result, record

def ans_recall(gen, answer):
    from nltk.tokenize import word_tokenize

    def process_string(s):
        words = []
        for word in word_tokenize(str(s).replace('"', "")):
            if word not in ",'.<>!()":
                words.append(word.lower())
        return words

    def compute_acc_single(gold_toks, pred_toks):
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return float(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        return num_same / len(gold_toks)

    def compute_acc(golds, pred):
        golds_toks = [process_string(gold) for gold in golds if gold != ""]
        pred_toks = process_string(pred)
        try:
            return max(compute_acc_single(gold_toks, pred_toks) for gold_toks in golds_toks)
        except:
            return 0

    return sum(compute_acc(golds=golds, pred=gen) for golds in answer) / len(answer)

def find_most_common(lst):
    # 创建 Counter 对象
    count = Counter(lst)
    # 找到出现次数最多的元素及其频次
    most_common_element, frequency = count.most_common(1)[0]
    return most_common_element

def extract_value(state):
    import re
    pattern = r"###([\d.]+)###"
    match = re.search(pattern, state)
    if match:
        number = match.group(1)
    else:
        pattern = r"value of this step is (\d+(\.\d+)?)"
        match = re.search(pattern, state)
        if match:
            number = match.group(1)  # 提取匹配结果中的分数部分
        else:
            number = 0.5
    return number

def get_output(args, custom_model, dataset, test_index, model, tokenizer):
    eval_item = dataset[test_index]
    answer, question = eval_item['answer'], eval_item['question']

    finish_result, record = sbs(args, eval_item, custom_model, tokenizer)

    finish_result.sort(key=lambda x: x['value'][-1], reverse=True)

    record.append(copy.deepcopy([elem['messages'][-1] for elem in finish_result]))

    try:
        most_common = finish_result[0]['messages'][-1]['content'].split('answer is:')[1].strip('<thought>')
    except:
        most_common = finish_result[0]['messages'][-1]['content']

    return most_common, answer, finish_result[0], record

def main(args, dataset, model_path):
    dataset = dataset[:args.max_num_examples]
    parts = list(split_data(dataset, args.total_parts))
    dataset = parts[args.split_index]

    test_num = min(len(dataset), args.max_num_examples)
    dataset = dataset[:test_num]

    custom_model, model, tokenizer = None, None, None
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    all_result_dict = []
    all_em, all_f1 = 0, 0
    for test_index in trange(test_num):
        args.search_fail_times = 0
        args.search_trial_times = 0
        try:
            pred, answer, result_dict, record = get_output(args, custom_model, dataset, test_index, model, tokenizer)
            em_score = np.max([int(cut_and_normalize_strs(answer[index]) in cut_and_normalize_strs(pred)) for index in range(len(answer))])
            f1_score = np.max([f1_score_cal(cut_and_normalize_strs(pred), cut_and_normalize_strs(answer[index])) for index in range(len(answer))])

            all_em += float(em_score)
            all_f1 += float(f1_score)
            wandb.log({'Step': test_index, 'EM': all_em/(test_index+1), 'F1': all_f1/(test_index+1)})

            result_dict['pred'] = pred
            result_dict['GT'] = answer
            result_dict['EM'] = float(em_score)
            result_dict['F1'] = float(f1_score)
            result_dict['process'] = record
            all_result_dict.append(result_dict)

            with open(f'{args.model_path}/{args.unique_identifier}.json', 'w') as file:
                json.dump(all_result_dict, file, indent=4)

        except Exception as e:
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data', type=str)
    parser.add_argument("--data_file", default='test_data.jsonl', type=str)
    parser.add_argument("--method", default='ours', type=str, required=False, help=" ")
    parser.add_argument("--model_path", default='', type=str)
    parser.add_argument("--dataset", default='musique', type=str)
    parser.add_argument("--beam_size_thought", default=1, type=int)
    parser.add_argument("--beam_size_search", default=1, type=int)
    parser.add_argument("--beam_depth", default=5, type=int)
    parser.add_argument("--expand_num_thought", default=1, type=int)
    parser.add_argument("--expand_num_search", default=1, type=int)
    parser.add_argument("--llm", type=str)
    parser.add_argument("--unique_identifier", type=str)
    parser.add_argument("--retriever", default='', type=str)
    parser.add_argument("--retriever_ip", default='', type=str)
    parser.add_argument("--retriever_port", default=5000, type=int)
    parser.add_argument("--policy_ip", default='', type=str)
    parser.add_argument("--value_port", default=5000, type=int)
    parser.add_argument("--value_ip", default='', type=str)
    parser.add_argument("--policy_port", default=8001, type=int)
    parser.add_argument("--split_index", default=0, type=int)
    parser.add_argument("--total_parts", default=1, type=int)
    parser.add_argument("--base_device", default=0, type=int)
    parser.add_argument("--use_retrieve", default=0, type=int)
    parser.add_argument("--max_length", default=500, type=int)
    parser.add_argument("--temperature", default=0.01, type=float)
    parser.add_argument("--temperature_search", default=1.2, type=float)
    parser.add_argument("--top_k", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--consistency", default=0, type=int)
    parser.add_argument("--trial_time", default=1, type=int)
    parser.add_argument("--llm_api_times", default=0, type=int)
    parser.add_argument("--llm_output_tokens", default=0, type=int)
    parser.add_argument("--llm_input_tokens", default=0, type=int)
    parser.add_argument("--top_K_original", default=5, type=int)
    parser.add_argument("--max_num_examples", type=int, default=500, help="maximum number of examples to evaluate.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load finetune in 8bit method, which will reduce memory and speed up inference.")
    args = parser.parse_args()

    from datetime import datetime
    current_time = datetime.now().strftime("%Y%m%d%H")

    args.unique_identifier = (args.model_path.split('save/')[1].replace('/', '_') + '_' + args.dataset
                              + f'_bs_{args.beam_size_thought}_{args.beam_size_search}_bd_{args.beam_depth}_en_{args.expand_num_thought}_{args.expand_num_search}_{args.temperature}_{args.unique_identifier}')

    args.unique_identifier = args.unique_identifier.replace('__', '_')
    wandb.login(key='7a5de6a00077b290a03126cfd94564b0ed3a5c59')
    wandb.init(project='mcts', entity="sunhaopku")
    wandb.run.name = args.unique_identifier
    wandb.config.update(args, allow_val_change=True)

    all_data = []
    file_path = f'{args.data_dir}/{args.data_file}'
    if args.data_file.endswith('.json'):
        with open(file_path, 'r') as file:
            all_data = json.load(file)
    elif args.data_file.endswith('.jsonl'):
        with open(file_path, 'r') as file:
            for line in file:
                if not line.strip():
                    continue
                data = json.loads(line)
                all_data.append(data)

    filter_data = []
    for elem in all_data:
        if args.dataset == elem['source']:
            filter_data.append(elem)

    main(args, filter_data, args.model_path)

    wandb.finish()