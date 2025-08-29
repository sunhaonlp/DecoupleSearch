from tqdm import trange, tqdm
import pickle
import os
import glob
from collections import Counter
import json
import argparse

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

def load_roots_from_directory(directory_path):
    roots = []
    directory_path = 'save/' + directory_path
    # Use glob to find all files matching the pattern mcts_tree_*.pkl
    file_pattern = os.path.join(directory_path, 'mcts_tree_*.pkl')
    for filepath in glob.glob(file_pattern):
        while (1):
            try:
                with open(filepath, 'rb') as f:
                    roots.extend(pickle.load(f))
                break
            except:
                continue

    return roots

def extract_paths(root):
    def dfs(node, current_path):
        # 如果是叶子节点，添加当前路径到结果中
        if not node.children:
            paths.append(current_path + [node])
        else:
            # 对每个子节点递归
            for child in node.children:
                dfs(child, current_path + [node])

    paths = []
    dfs(root, [])
    return paths

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

def obtain_training_data(all_elem, max_paths_corr, max_paths_incorr, q_answer, threshold, different, filter_yes_no):
    labeled_query = []
    all_correct_path = []
    all_incorrect_path = []
    all_unfinishes_path = []
    all_reward = []
    for elem in tqdm(all_elem):
        all_paths = extract_paths(elem)
        flag = 0
        final_reward = []

        if filter_yes_no:
            if is_yes_no_question(all_paths[0][0].query, q_answer[all_paths[0][0].query]):
                continue

        if all_paths[0][0].query in labeled_query:
            continue
        else:
            labeled_query.append(all_paths[0][0].query)

        elem_correct_path = []
        elem_incorrect_path = []
        for path in all_paths:
            if path[-1].is_terminal:
                flag = 1
                final_reward.append(path[-1].reward_thought)
                if path[-1].reward_thought == 1:
                    elem_correct_path.append(path)
                else:
                    elem_incorrect_path.append(path)
            else:
                all_unfinishes_path.append(path)
        if flag == 1:
            all_reward.append(max(final_reward))

        if different:
            q_path = {}
            for path in elem_correct_path:
                if path[-2].query not in q_path:
                    q_path[path[-2].query] = [path]
                else:
                    q_path[path[-2].query].append(path)
            all_different_path = [q_path[q][0] for q in q_path]
            elem_correct_path = all_different_path[:max_paths_corr]

            ## filter incorrect
            q_path = {}
            for path in elem_incorrect_path:
                if str(path[-2].query) not in q_path:
                    q_path[str(path[-2].query)] = [path]
                else:
                    q_path[str(path[-2].query)].append(path)
            all_different_path = [q_path[q][0] for q in q_path]
            elem_incorrect_path = all_different_path[:max_paths_incorr]

        else:
            elem_correct_path = elem_correct_path[:max_paths_corr]
            elem_incorrect_path = elem_incorrect_path[:max_paths_incorr]

        all_correct_path.extend(elem_correct_path)
        all_incorrect_path.extend(elem_incorrect_path)

    print(len(all_correct_path))
    print(len(all_incorrect_path))

    all_training_elem = []

    for path in tqdm(all_correct_path):
        original_query = path[0].query

        original_answer = path[0].answer
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        },
            {
                "content": f"{original_query}",
                "role": "user"
            }]

        for elem in path[1:-1]:
            elem.query = eval(str(elem.query))
            elem = [{
                "role": "assistant",
                "content": f"✿THOUGHT✿: {elem.thought} <thought>\n✿Action✿: Search({elem.query[0]})",
                'q_value_thought': elem.q_value_thought
            },
                {
                    "role": "function",
                    "content": f"{elem.observation} <search>",
                    'q_value_search': elem.q_value_search
                }]
            messages.extend(elem)

        try:
            final_thought = path[-1].thought.split('**Thought:**')[1].split('**Action:**')[0].strip().strip('-').strip(
                '\n')
        except:
            continue

        if 'search query' in final_thought.lower():
            continue

        try:
            final_answer = path[-1].thought.split('**Thought:**')[1].split('inish(')[1].strip().strip('-').strip(
                '\n').strip(')')
        except:
            if type(original_answer) == type([]):
                original_answer = original_answer[0]
            else:
                final_answer = original_answer

        elem = [{"content": f"✿THOUGHT✿: {final_thought} The answer is: {final_answer} <thought>",
                 "response_role": "assistant",
                 "role": "assistant",
                 'q_value_thought': path[-1].q_value_thought
                 }]
        messages.extend(elem)

        training_elem = {"type": "chatml", 'messages': messages,
                         'ext': {'golden_answer': q_answer[original_query],
                                 'response_acc': True}}
        # print(path[-1].thought)
        pred = path[-1].thought.split('Finish(')[1][:-1]
        # print(pred)
        gt = q_answer[original_query]
        score = ans_recall(pred, gt)
        if score >= threshold:
            all_training_elem.append(training_elem)

    for path in tqdm(all_incorrect_path):
        original_query = path[0].query

        original_answer = path[0].answer
        messages = [{
            "role": "system",
            "content": "You are a helpful assistant."
        },
            {
                "content": f"{original_query}",
                "role": "user"
            }]

        for elem in path[1:-1]:
            elem.query = eval(str(elem.query))
            elem = [{
                "role": "assistant",
                "content": f"✿THOUGHT✿: {elem.thought} <thought>\n✿Action✿: Search({elem.query[0]})",
                'q_value_thought': elem.q_value_thought
            },
                {
                    "role": "function",
                    "content": f"{elem.observation} <search>",
                    'q_value_search': elem.q_value_search
                }]
            messages.extend(elem)

        try:
            final_thought = path[-1].thought.split('**Thought:**')[1].split('**Action:**')[0].strip().strip('-').strip('\n')
        except:
            continue

        if 'search query' in final_thought.lower():
            continue

        try:
            final_answer = path[-1].thought.split('**Thought:**')[1].split('inish(')[1].strip().strip('-').strip('\n').strip(')')
        except:
            if type(original_answer) == type([]):
                final_answer = original_answer[0]
            else:
                final_answer = original_answer

        elem = [{"content": f"✿THOUGHT✿: {final_thought} The answer is: {final_answer} <thought>",
                 "response_role": "assistant",
                 "role": "assistant",
                 'q_value_thought': path[-1].q_value_thought
                 }]
        messages.extend(elem)

        training_elem = {"type": "chatml", 'messages': messages,
                         'ext': {'golden_answer': q_answer[original_query]}}

        all_training_elem.append(training_elem)

    all_value_thought = []
    all_value_search = []
    for elem in all_training_elem:
        for tmp in elem['messages']:
            if 'q_value_thought' in tmp:
                all_value_thought.append(tmp['q_value_thought'])
            elif 'q_value_search' in tmp:
                all_value_search.append(tmp['q_value_search'])

    return all_training_elem

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_directory', type=str, default='')
    args = parser.parse_args()

    q_answer = {}
    all_data = []
    with open('data/mcts_labeling_data.jsonl', 'r') as file:
        for line in file:
            if not line.strip():
                continue
            data = json.loads(line)
            all_data.append(data)

    for i in range(len(all_data)):
        q_answer[all_data[i]['prompt']] = all_data[i]['answers']

    all_elem = load_roots_from_directory(args.label_directory)

    all_training_elem = obtain_training_data(all_elem, 1, 10, q_answer, 0.8, True, True)

    with open(f'data/train_data.json', 'w') as file:
        json.dump(all_training_elem, file, indent=4)