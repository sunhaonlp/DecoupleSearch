from utils import *
import argparse
from mcts_utils import MCTS, MCTSNode
import os
import json

def annotate_query_with_mcts(args, dataset):
    all_root = []
    new_root = []
    unfinished = dataset
    for elem in unfinished:
        elem['trial_times'] = 0
    while unfinished:
        elem = unfinished.pop(0)
        print(f'剩余{len(unfinished)}没inference完')
        try:
            original_query, answer = elem['prompt'],  str(elem['answers'][0])
            root = MCTSNode(True, 0,  original_query, answer,None, None, 0)
            mcts = MCTS(root, args.llm, args.local_llm_ip, args.local_llm_port, temperature=args.temperature, retriever_ip=args.retriever_ip, retriever_port=args.retriever_port, max_expansion_num=args.max_expansion_num, max_node_num_per_layer=args.max_node_num_per_layer, verbose=args.verbose)
            for i in range(args.iterations):
                print(f'===========The {i+1}-th Iteration===========')
                status = mcts.search()

                if not status:
                    print('No node can be expanded!')
                    break

                correct_nodes_num = root.correct_nodes_num(root)
                if correct_nodes_num >= args.threshold:
                    print('early stop!')
                    break

            all_root.append(root)
            new_root.append(root)
            save_roots_to_file(all_root, f"save/{args.unique_identifier}/mcts_tree_{args.split_index}.pkl")
            save_roots_to_file([elem.query for elem in all_root], f"save/{args.unique_identifier}/question_{args.split_index}.pkl")

        except Exception as e:
            elem['trial_times'] += 1
            if elem['trial_times'] < 3:
                unfinished.append(elem)
            import traceback
            traceback.print_exc()
            continue

# 示例使用
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data', type=str)
    parser.add_argument("--data_file", default='', type=str)
    parser.add_argument("--unique_identifier", type=str)
    parser.add_argument("--llm", default='', type=str)
    parser.add_argument("--max_num_examples", default=1000000, type=int)
    parser.add_argument("--max_expansion_num", default=3, type=int)
    parser.add_argument("--iterations", default=3, type=int)
    parser.add_argument("--verbose", default=True, type=bool)
    parser.add_argument("--max_node_num_per_layer", default=200, type=int)
    parser.add_argument("--retriever_ip", default='', type=str)
    parser.add_argument("--retriever_port", default=5001, type=int)
    parser.add_argument("--threshold", default=1, type=int)
    parser.add_argument("--local_llm_ip", default='', type=str)
    parser.add_argument("--local_llm_port", default=1, type=int)
    parser.add_argument("--total_parts", default=1, type=int)
    parser.add_argument("--split_index", default=0, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    args = parser.parse_args()

    args.unique_identifier = (str(time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday) + '_' + str(time.localtime().tm_hour)  + '_' +
                              str(args.llm) + '_' + str(args.max_expansion_num) + '_' + str(args.iterations) + '_' + str(args.threshold) +  '_'
                              + str(args.temperature) +  '_'+ args.data_file.replace('.jsonl', ''))

    all_data = []
    file_path = f'{args.data_dir}/{args.data_file}'
    # 根据文件扩展名判断如何加载数据
    if args.data_file.endswith('.jsonl'):
        with open(file_path, 'r') as file:
            for line in file:
                if not line.strip():
                    continue
                data = json.loads(line)
                all_data.append(data)

    dataset = all_data[:args.max_num_examples]
    parts = list(split_data(dataset, args.total_parts))
    dataset = parts[args.split_index]

    # 创建多层目录（包括中间目录）
    try:
        os.makedirs(f'save/{args.unique_identifier}')
    except:
        pass

    annotations = annotate_query_with_mcts(args, dataset)
