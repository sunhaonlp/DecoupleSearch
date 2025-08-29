import pickle
import torch.nn as nn
import torch
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai
from typing import List, Literal, Optional, Dict
from collections import Counter
from openai import OpenAI
import regex, string


prompt_expand_initial = '''**You are a highly capable web agent. Your task is to engage in multi-step reasoning and propose plans to reach a final answer for the given question.**
For each step, please include the following elements:
1. **Thought:** Offer a comprehensive and detailed analysis. This section should cover:
   - The specific information needed to address the query effectively.
   - Identification of what needs to be searched or verified to move closer to a solution.
2. **Action:** Clearly specify the exact query in the format Search([List of Queries]) without extra content. Ensure the queries convey the same semantic information but are expressed differently to enhance the likelihood of finding the necessary information.

You only need to generate the first step and the output format is as follows:
- **Thought:** [Detailed analysis of the needed information]
- **Action:** Search([List of Queries])

Below is the original query. Please provide the plan for the first step:
{query}
'''

prompt_expand_subsequent = '''**You are a highly capable web agent. Your task is to engage in multi-step reasoning and propose plans to reach a final answer for the given question.**
For each step, please include the following elements:
**Thought:** Offer a comprehensive and detailed analysis. This section should cover:
    - An analysis of the specific information required to address the question effectively and the information currently available.
    - If the information is enough to answer the question, you should conduct deep analysis based on the information and then answer the question.
    - If the information is not enough to answer the question, you should analyze whether the current plan progresses well. 
        - If yes, predict the next action.
        - If no, reflect on why the progress is not good and then propose a new plan.

**Action:** Provide the next action. This section should cover:
   - If the information is enough to answer the question, you should output the final answer in format of Finish(put the answer here) without extra content. 
   - If the information is not enough to answer the question, you should clearly specify the exact query for the next search in the format Search([List of Queries]) without extra content. Ensure the queries convey the same semantic information but are expressed differently to enhance the likelihood of finding the necessary information.

For the question: {query}, here is the reasoning process so far:
{history}

**The Output Format:**
- **Thought:** [Detailed analysis of the needed information, existing information, identifies whether information is enough. If enough, conduct analysis to obtain the final answer, else, identify what still needs to be searched]
- **Action:** [Finish(put the answer here) or Search([List of Queries])]

Please provide the plan for the next step:
'''

prompt_expand_initial_one_query = '''**You are a highly capable web agent. Your task is to engage in multi-step reasoning and propose plans to reach a final answer for the given question.**
For each step, please include the following elements:
1. **Thought:** Offer a comprehensive and detailed analysis. This section should cover:
   - The specific information needed to address the query effectively.
   - Identification of what needs to be searched or verified to move closer to a solution.
2. **Action:** Clearly specify the exact query in the format Search([Query]) without extra content.

You only need to generate the first step and the output format is as follows:
- **Thought:** [Detailed analysis of the needed information]
- **Action:** Search([Query])

Below is the original query. Please provide the plan for the first step:
{query}
'''

prompt_expand_subsequent_one_query = '''**You are a highly capable web agent. Your task is to engage in multi-step reasoning and propose plans to reach a final answer for the given question.**
For each step, please include the following elements:
**Thought:** Offer a comprehensive and detailed analysis. This section should cover:
    - An analysis of the specific information required to address the question effectively and the information currently available.
    - If the information is enough to answer the question, you should conduct deep analysis based on the information and then answer the question.
    - If the information is not enough to answer the question, you should analyze whether the current plan progresses well. 
        - If yes, predict the next action.
        - If no, reflect on why the progress is not good and then propose a new plan.

**Action:** Provide the next action. This section should cover:
   - If the information is enough to answer the question, you should output the final answer in format of Finish(put the answer here) without extra content. 
   - If the information is not enough to answer the question, you should clearly specify the exact query for the next search in the format Search([Query]) without extra content.

For the question: {query}, here is the reasoning process so far:
{history}

**The Output Format:**
- **Thought:** [Detailed analysis of the needed information, existing information, identifies whether information is enough. If enough, conduct analysis to obtain the final answer, else, identify what still needs to be searched]
- **Action:** [Finish(put the answer here) or Search([Query])]

Please provide the plan for the next step:
'''

prompt_expand_initial_one_query_ = '''**You are a highly capable web agent. Your task is to engage in multi-step reasoning and propose plans to reach a final answer for the given question.**
For each step, please include the following elements:
1. ✿THOUGHT✿ Offer a comprehensive and detailed analysis. This section should cover:
   - The specific information needed to address the query effectively.
   - Identification of what needs to be searched or verified to move closer to a solution.
   - Put <thought> at the end of thought.
2. ✿Action✿ Clearly specify the exact query in the format Search(Query) without extra content.

You only need to generate the first step and the output format is as follows:
✿THOUGHT✿: [Detailed analysis of the needed information]<thought>
✿Action✿: Search(Query)

Below is the original query. Please provide the plan for the first step:
{query}
'''

prompt_expand_subsequent_one_query_ = '''**You are a highly capable web agent. Your task is to engage in multi-step reasoning and propose plans to reach a final answer for the given question.**

For the question: {query}, here is the reasoning process so far:
{history}

For each step, please include the following elements:
✿THOUGHT✿ Offer a comprehensive and detailed analysis. This section should cover:
    - An analysis of the specific information required to address the question effectively and the information currently available.
    - If the information is enough to answer the question, you should conduct deep analysis based on the information and then answer the question.
    - If you cannot find the information after several search trials, please analyze the information at head and obtain the final answer.
    - If the information is not enough to answer the question, you should analyze whether the current plan progresses well. 
        - If yes, predict the next action.
        - If no, reflect on why the progress is not good and then propose a new plan.
        - Put <thought> at the end of thought

✿Action✿ Provide the next action. This section should cover:
   - If the information is enough to answer the question, you should output the final answer in format of Finish(put the answer here) without extra content. 
   - If the information is not enough to answer the question, you should clearly specify the exact query for the next search in the format Search(Query) without extra content.

Please provide the plan for the next step:
**The Output Format:**
✿THOUGHT✿: [Detailed analysis of the needed information. If you cannot find the information after several search trials, please analyze the information at head and obtain the final answer.]<thought>
✿Action✿: [Finish(put the answer here) or Search(Query)]
Output:
'''

prompt_expand_query = '''**You are a highly capable web agent. Your task is to generate the search query based on the plan.**

For the question: {query}, here is the reasoning process so far:
{history}

Here is the the current reasoning step:
{current}

Please provide the search query for the next step:
**The Output Format:**
✿Action✿: Search(put the query here)
Output:
'''

prompt_evaluate_intermediate = '''**Task:**  Assess the effectiveness of the thought and the search result in the last reasoning step.
As an advanced web search agent, your role is to systematically evaluate the current step step.
For the question: {query}, here is the reasoning process so far:
{history}

As an expert in web search, your tasks are as follows:
1. Analyze the thought in the last step: Evaluate the thought and determine its effectiveness in reaching the final answer. Assign a score between -1 and 1, where -1 means the thought is useless and 1 means the thought is very effective.
2. Analyze the search result in the last step: Evaluate the search result and determine its effectiveness in reaching the final answer. Assign a score between -1 and 1, where -1 means the search result was ineffective, and 1 means the search results were highly useful.

You should output the following elements
**Analysis of the thought:**
- Analyze whether the thought from the last step were helpful in progressing toward the final answer.
- Assign a score between -1 and 1, where -1 means the step was ineffective, and 1 indicates high usefulness.
- You must conclude the analysis with the format of "the value of the thought is ###x###", where x represent the value and # is the identifier. Remember that you must output the value x with identifier ###.

**Analysis of the search result:**
- Analyze whether the search query and search results from the last step were helpful in progressing toward the final answer.
- Assign a score between -1 and 1, where -1 means the step was ineffective, and 1 indicates high usefulness.
- You must conclude the analysis with the format of "the value of the search result is ###x###", where x represent the value and # is the identifier. Remember that you must output the value x with identifier ###.

Please begin by analyzing the previous step:
**Analysis of the thought:**
'''

prompt_simulate = '''**Task:**  Assess the effectiveness of each reasoning step and suggest subsequent actions.
As an advanced web search agent, your role is to systematically evaluate each search step and guide the investigation process to answer specific questions.
For the question: {query}, here is the reasoning process so far:
{history}

As an expert in web search, your tasks are as follows:
1. Analyze the last step: Evaluate the reasoning and search results of the previous step to determine their effectiveness in reaching the final answer. Assign a score between 0 and 1, where 0 means the step was ineffective, and 1 means the results were highly useful.
2. Propose the next step: If the search process is progressing well, continue with the plan and propose the next step. If the results are unsatisfactory, identify the issue and suggest a new approach or search strategy.

You should output the following elements
**Analysis of the last step:**
- Analyze whether the reasoning and search results from the last step were helpful in progressing toward the final answer.
- Assign a score between 0 and 1, where 0 means the step was ineffective, and 1 indicates high usefulness.
- You must conclude the analysis with the format of "the value of this step is ###x###", where x represent the value and # is the identifier. Remember that you must output the value x with identifier ###.

**Propose The Next step**
**Thought:** Offer a comprehensive and detailed analysis. This section should cover:
    - An analysis of the specific information required to address the question effectively and the information currently available.
    - If the information is enough to answer the question, you should conduct deep analysis based on the information and then answer the question.
    - If the information is not enough to answer the question, you should analyze whether the current plan progresses well. 
        - If yes, predict the next action.
        - If no, reflect on why the progress is not good and then propose a new plan.

**Action:** Provide the next action. This section should cover:
   - If the information is enough to answer the question, you should output the final answer in format of Finish(put the answer here) without extra content. 
   - If the information is not enough to answer the question, you should clearly specify the exact query for the next search in the format Search([List of Queries]) without extra content. Ensure the queries convey the same semantic information but are expressed differently to enhance the likelihood of finding the necessary information.

**The Output Format:**
**Analysis of the last step:**
- Provide an analysis of the last completed step. You must conclude the analysis with the format of "the value of this step is ###x###", where x represent the value and # is the identifier. Remember that you must output the value x with identifier ###.

**Propose The Next step**
- **Thought:** [Detailed analysis of the needed information, existing information, identifies whether information is enough. If enough, conduct analysis to obtain the final answer, else, identify what still needs to be searched]
- **Action:** [Finish(put the answer here) or Search([List of Queries])]

Please begin by analyzing the previous step:
**Analysis of the last step:**
'''

prompt_evaluate_final = '''**Task: Evaluate the Predicted Answer Against the Correct Answer List**

You are tasked with grading the accuracy of a predicted answer compared to a list of correct answers. Assign a score of either -1 or 1, where:
- -1 means the predicted answer does not semantically match any of the answers in the correct answer list.
- 1 means the predicted answer semantically matches at least one answer in the correct answer list, even if the wording is different.

**Output Format:**
Just provide the score as either -1 or 1.

Given the following instance, directly output the score:

Query: {query}
Correct Answer List: {gt}
Predicted Answer: {pred}

The score is:
'''

prompt_observation = '''**Task:**  Assess the effectiveness of each reasoning step and suggest subsequent actions.
As an advanced web search agent, your role is to systematically evaluate each search step and guide the investigation process to answer specific questions.
For the question: {query}, here is the reasoning process so far:
{history}

As an expert in web search, your tasks are as follows:
1. Analyze the last step: Evaluate the reasoning and search results of the previous step to determine their effectiveness in reaching the final answer. Assign a score between 0 and 1, where 0 means the step was ineffective, and 1 means the results were highly useful.
2. Propose the next step: If the search process is progressing well, continue with the plan and propose the next step. If the results are unsatisfactory, identify the issue and suggest a new approach or search strategy.

You should output the following elements
**Analysis of the last step:**
- Analyze whether the reasoning and search results from the last step were helpful in progressing toward the final answer.
- Assign a score between 0 and 1, where 0 means the step was ineffective, and 1 indicates high usefulness.
- You must conclude the analysis with the format of "the value of this step is ###x###", where x represent the value and # is the identifier. Remember that you must output the value x with identifier ###.

**Propose The Next step**
**Thought:** Offer a comprehensive and detailed analysis. This section should cover:
    - An analysis of the specific information required to address the question effectively and the information currently available.
    - If the information is enough to answer the question, you should conduct deep analysis based on the information and then answer the question.
    - If the information is not enough to answer the question, you should analyze whether the current plan progresses well. 
        - If yes, predict the next action.
        - If no, reflect on why the progress is not good and then propose a new plan.

**Action:** Provide the next action. This section should cover:
   - If the information is enough to answer the question, you should output the final answer in format of Finish(put the answer here) without extra content. 
   - If the information is not enough to answer the question, you should clearly specify the exact query for the next search in the format Search([List of Queries]) without extra content. Ensure the queries convey the same semantic information but are expressed differently to enhance the likelihood of finding the necessary information.

**The Output Format:**
**Analysis of the last step:**
- Provide an analysis of the last completed step. You must conclude the analysis with the format of "the value of this step is ###x###", where x represent the value and # is the identifier. Remember that you must output the value x with identifier ###.

**Propose The Next step**
- **Thought:** [Detailed analysis of the needed information, existing information, identifies whether information is enough. If enough, conduct analysis to obtain the final answer, else, identify what still needs to be searched]
- **Action:** [Finish(put the answer here) or Search([List of Queries])]

Please begin by analyzing the previous step:
**Analysis of the last step:**
'''

prompt_is_final = '''**Task: Determine If Current Information Suffices to Answer the Query and Try to answer the question**

Given a query and the reasoning process, your task is to assess whether the information derived from the reasoning process is adequate to address the given query. If the information is sufficient, perform a detailed analysis based on the available information and provide an answer to the question.

Here is the query and the reasoning process:
Query: {query}

The reasoning process:
{reasoning_process}

**The Output Format:**
You should include the following elements in your response:
**Analysis of the Reasoning Process:**
- Analyze whether the information obtained from the reasoning process is enough for answer the question.
- If the information is sufficient, conduct deep analysis based on the information and then answer the question.

**Final Decision:**
- If the information is enough, output the final answer in format of Finish(put the answer here) without extra content.
- If the information is insufficient, output "Need more information" without extra content.

Now, first output the analysis of the reasoning process:
**Analysis of the Reasoning Process:**
'''

prompt_parse = '''**Task: Parse and Format the Model Output**

Your task is to take the unstructured output from the model and format it into a clear, structured format for easy parsing. The desired format involves outlining possible plans. Each plan should include a concise statement of the thought process and the corresponding search query.

Instructions:
1. Analyze the provided output.
2. Organize and format the output into the specified structure.

**Desired Output Format:**

- **Thought:** ...

- **Action:** Search(list of queries)

For example:
- **Thought:** The current search has confirmed that Lake Michigan is the primary source of drinking water ...

- **Action:** Search(["Is the band Chicago named after Lake Michigan", "adult contemporary music by the band Chicago"])

The regex expersion is r"(?:- )?\*\*Thought:\*\*\s*([\s\S]*?)\n\s*(?:- )?\*\*Action:\*\*\s*Search\((\[[\s\S]*?\])\)"
You must follow the output format and ensure the output can be parsed by the regex expersion.
Here is the output, please parse the output and output in the desired format:
{output}
**Desired Output Format:**:
'''


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

def f1_score_cal(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def retrieve_from_api(args, query, index=0, top_k=10):
    retriever_ip = [args.retriever_ip]
    retriever_port = [args.retriever_port]

    if len(query) > 1:
        return 'None', index, 1
    import random
    # 随机选择一个数字，从0到7（包括0和7）
    random_number = random.randint(0, len(retriever_port) - 1)
    for _ in range(50):
        try:
            # print(query)
            payload = {'query': query, 'top_k': 5}
            response = requests.post(f'http://{retriever_ip[random_number]}:{retriever_port[random_number]}/retrieve', json=payload)
            doc_texts = '\n'.join([f"Doc {i + 1}: {doc['text']}" for i, doc in enumerate(response.json())])

            return doc_texts, index, 0

        except Exception as e:
            time.sleep(1)
            print(e)
            continue

def truncate_at_last_search(input_string):
    # 找到子字符串 "Search" 最后一次出现的索引
    last_index = input_string.rfind("Search")
    if last_index == -1:
        last_index = input_string.rfind("search")

    # 如果找到了 "Search"，则截断字符串到这个位置
    if last_index != -1:
        truncated_string = input_string[:last_index + len("Search")]
    else:
        # 如果没有找到 "Search"，返回完整字符串
        truncated_string = input_string

    return truncated_string

def generation_sever_hierarchical_complete(tokenizer, messages, port, ip, num_samples=5, max_length=500, temperature=1.0, temperature_search=1.2, mode='Normal'):
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
    # messages = elem['messages']
    # functions = elem['functions']

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    if mode == 'Normal':
        input_prompt = tokenizer.decode(input_ids[0])
    else:
        input_prompt = tokenizer.decode(input_ids[0])
        input_prompt = truncate_at_last_search(input_prompt)

    client = openai.Client(base_url=f"http://{ip}:{port}/v1", api_key="token-abc123")
    # print(f"http://{ip}:8001/v1")
    for _ in range(40):
        try:
            completion = client.completions.create(model="",
                                                   prompt=input_prompt,
                                                   max_tokens=max_length,
                                                   temperature=temperature_search,
                                                   top_p=1,
                                                   stop=['✿RESULT✿', '✿RETURN✿'],
                                                   n=40)
            generated_texts = list(set([completion.choices[i].text for i in range(len(completion.choices))]))
            return most_different_strings(generated_texts, num_samples)
        except:
            print('parsing error, retrying!')
            pass

    return None

def get_reasoning_history(messages):
    turns = int((len(messages) - 1) / 2)
    history = ''
    for index in range(turns):
        history += f"## Step {index+1}:\n{messages[index*2+1]['content']}\n**Search Result:** {messages[index*2+2]['content']}\n\n"
    return history

def generation_sever_hierarchical(tokenizer, messages, port, ip, num_samples=5, max_length=500, temperature=1.0, mode='Normal'):
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
    # messages = elem['messages']
    # functions = elem['functions']

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    if mode == 'Normal':
        input_prompt = tokenizer.decode(input_ids[0])
    else:
        input_prompt = tokenizer.decode(input_ids[0])
        input_prompt = truncate_at_last_search(input_prompt)

    client = openai.Client(base_url=f"http://{ip}:{port}/v1", api_key="token-abc123")
    # print(f"http://{ip}:8001/v1")
    for _ in range(40):
        try:
            completion = client.completions.create(model="",
                                                   prompt=input_prompt,
                                                   temperature=temperature,
                                                   max_tokens=max_length,
                                                   stop=['✿RESULT✿', '✿RETURN✿'],
                                                   n=num_samples)
            generated_texts = [completion.choices[i].text for i in range(num_samples)]
            return generated_texts
        except:
            print('parsing error, retrying!')
            pass

    return None

def most_different_strings(strings: List[str], n: int) -> List[str]:
    if n >= len(strings):
        return strings

    def string_difference(s1, s2):
        # 使用集合方式计算字符串之间的差异
        return len(set(s1) ^ set(s2))

    # 初始化，选择第一个字符串
    chosen = [strings[0]]

    while len(chosen) < n:
        max_diff = -1
        next_string = None
        for candidate in strings:
            if candidate in chosen:
                continue
            # 计算候选字符串到已选字符串的最小差异
            min_diff_to_chosen = min(string_difference(candidate, selected) for selected in chosen)
            if min_diff_to_chosen > max_diff:
                max_diff = min_diff_to_chosen
                next_string = candidate
        chosen.append(next_string)

    return chosen

def call_llm(prompt, llm, llm_ip, llm_port, temperature=0.8, index=-1):
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{llm_ip}:{llm_port}/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    while(1):
        try:
            # Prepare the content list
            content = [{"type": "text", "text": prompt}]
            response = client.chat.completions.create(
                model='',
                max_tokens=5000,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": ""},
                    {
                        "role": "user",
                        "content": content
                    },
                ],
            )

            answer = response.choices[0].message.content
            print(answer)
            if index == -1:
                return answer
            else:
                return answer, index
        except Exception as e:
            import traceback
            traceback.print_exc()
            time.sleep(1.5)
            continue

def parallel_call_llm(prompt, llm, llm_ip, llm_port, num=1, temperature=0.8):
    all_generation = [0 for _ in range(num)]
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(call_llm, prompt, llm, llm_ip, llm_port, temperature, i) for i in range(num)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result, index = future.result()
                all_generation[index] = result
            except Exception as e:
                print(e)
    return all_generation

def split_data(data, n):
    k, m = divmod(len(data), n)
    return (data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def save_roots_to_file(roots, filename):
    with open(filename, 'wb') as f:
        pickle.dump(roots, f)

def load_roots_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def load_hf_lm_and_tokenizer(
        model_name_or_path,
        tokenizer_name_or_path=None,
        device_index=0,
        device_map=False,
        load_in_8bit=False,
        load_in_half=False,
        gptq_model=False,
        use_fast_tokenizer=False,
        padding_side="left",
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True
        )
        model = model_wrapper.model
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            load_in_8bit=True
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            if torch.cuda.is_available():
                model = model.to(f"cuda:{device_index}")
        if load_in_half:
            model = model.half()
    model.eval()
    return model, tokenizer


class CustomModel_hierarchical(torch.nn.Module):
    def __init__(self, model, value_head_thought_path, value_head_search_path):
        super(CustomModel_hierarchical, self).__init__()
        self.model = model
        self.value_head_thought = nn.Linear(model.config.hidden_size, 1)  # Simplified value head
        self.value_head_search = nn.Linear(model.config.hidden_size, 1)  # Simplified value head
        self.config = model.config  # Inherit base model's configuration
        self.load_value_head(value_head_thought_path, value_head_search_path)
        self.value_head_thought = self.value_head_thought.half()  # Convert to float16
        self.value_head_search = self.value_head_search.half()  # Convert to float16


    def load_value_head(self, value_head_thought_path, value_head_search_path):
        # Load state dictionary
        from safetensors.torch import load_file
        state_dict = load_file(value_head_thought_path, device='cuda:0')
        # Remove 'summary.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('v_head_thought.summary.'):
                new_key = k[len('v_head_thought.summary.'):]  # Remove prefix
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        self.value_head_thought.load_state_dict(new_state_dict)

        state_dict = load_file(value_head_search_path, device='cuda:0')
        # Remove 'summary.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('v_head_search.summary.'):
                new_key = k[len('v_head_search.summary.'):]  # Remove prefix
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        self.value_head_search.load_state_dict(new_state_dict)


    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,  # Use the passed return_dict parameter
            output_hidden_states=True,
            **kwargs  # Pass any additional parameters
        )
        logits = outputs.logits

        # Ensure output hidden states are used for value head
        hidden_states = outputs.hidden_states[-1]  # Last layer's hidden states

        values_thought = self.value_head_thought(hidden_states).squeeze(-1)
        values_thought = torch.tanh(values_thought)

        values_search = self.value_head_search(hidden_states).squeeze(-1)
        values_search = torch.tanh(values_search)

        if return_dict:
            # Return a dictionary ensuring it includes logits
            return {
                "logits": logits,
                "values_thought": values_thought,
                'values_search': values_search,
                "input_ids": input_ids
            }
        else:
            return logits, values_thought, values_search