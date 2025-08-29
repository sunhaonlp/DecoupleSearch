import re
import pandas as pd
import json
import jsonlines


def is_yes_no_question(prompt, answers):
    # 检查问题是否以常见的yes/no问题开头
    yes_no_starters = [
        "is", "are", "was", "were", "do", "does", "did", "have", "has", "had",
        "can", "could", "will", "would", "should", "may", "might"
    ]

    # 将问题转换为小写并分割成单词
    words = prompt.lower().split()

    # 检查第一个单词是否是yes/no问题的常见开头
    if words[0] in yes_no_starters:
        return True

    # 检查答案是否只包含"yes"或"no"
    if all(ans.lower() in ['yes', 'no'] for sublist in answers for ans in sublist):
        return True

    return False

df_train = pd.read_parquet('/mnt/workspace/sunhao/LLM/2WikiMultihopQA/train.parquet')

all_train_data = []
for i in range(10000):
    all_train_data.append({'prompt': df_train['question'][i], 'answers': [df_train['answer'][i]]})

print(len(all_train_data))



all_data = []
with open('/mnt/workspace/sunhao/LLM/MuSiQue/musique_ans_v1.0_train.jsonl', 'r') as file:
    for line in file:
        if not line.strip():
            continue
        data = json.loads(line)
        all_data.append(data)

for i in range(10000):
    all_train_data.append({'prompt': all_data[i]['question'], 'answers': [all_data[i]['answer']]})

print(len(all_train_data))



with open('/mnt/workspace/sunhao/LLM/hotpot/hotpot_train_v1.1.json', 'r') as file:
    all_data = json.load(file)

for i in range(10000):
    all_train_data.append({'prompt': all_data[i]['question'], 'answers': [all_data[i]['answer']]})

print(len(all_train_data))



all_data = []
with open('/mnt/nas-alinlp/sunhao/code/mcts-rag/data/labeling/nq-train.jsonl', 'r') as file:
    for line in file:
        if not line.strip():
            continue
        data = json.loads(line)
        all_data.append(data)

for i in range(10000):
    all_train_data.append({'prompt': all_data[i]['question'], 'answers': all_data[i]['answer']})

print(len(all_train_data))



filter_data = []
false_data = []
for elem in all_train_data:
    is_yes_no = is_yes_no_question(elem['prompt'], elem['answers'])
    if not is_yes_no:
        filter_data.append(elem)
    else:
        false_data.append(elem)


train_file = jsonlines.open(f'../data/mcts_labeling_data.jsonl', mode='w')

for elem in filter_data:
    train_file.write(elem)




## test_data
all_data = []
with open('/mnt/workspace/sunhao/code/mcts-rag/data/eval/eval_rag_v3.jsonl', 'r') as file:
    for line in file:
        if not line.strip():
            continue
        data = json.loads(line)
        all_data.append(data)

filter_data = []
for elem in all_data:
    for dataset in ['nq', 'hotpotqa', 'tqa', '2wikimultihopqa', 'bamboogle', 'musique']:
        if 'task' in elem and f'rag_qa_en/{dataset}' in elem['task']:
            filter_data.append({'question': elem['prompt'], 'answer': elem['answers'][0], 'source': dataset})
            break

source_n = {}
for elem in filter_data:
    if elem['source'] not in source_n:
        source_n[elem['source']] = 1
    else:
        source_n[elem['source']] += 1
print(source_n)
train_file = jsonlines.open(f'../data/test_data.jsonl', mode='w')

for elem in filter_data:
    train_file.write(elem)