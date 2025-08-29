from utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import re
import math

class MCTSNode:
    def __init__(self, is_root, id, query, answer, thought, observation, depth, parent=None):
        self.query = query              # 当前节点的查询
        self.answer = answer              # 当前节点的查询
        self.parent = parent            # 父节点
        self.children = []              # 子节点
        self.id = id            # 访问次数
        self.depth = depth            # 访问次数
        self.visit_count = 0            # 访问次数

        self.q_value_thought = 0.0
        self.q_value_search = 0.0
        self.reward_thought = 0.0
        self.reward_search = 0.0

        self.thought = thought
        self.observation = observation
        self.is_terminal = False
        self.is_root = is_root
        self.unique_identifier = f"{id}: {query}"[:100]
        self.possible_plans = None

    def is_fully_expanded(self):
        return len(self.children) >= 3  # 假设每个节点最多生成3个子查询

    def has_non_terminal_node(self, node):
        flag = 0
        children_queue = [node]
        while(children_queue):
            new_children_queue = []
            for elem in children_queue:
                if not elem.is_terminal and len(elem.children) == 0:
                    return 1
                new_children_queue.extend(elem.children)
            children_queue = new_children_queue
        return flag

    def has_ok_nodes(self, node, layer_node_number, max_node_num_per_layer):
        all_paths = self.extract_paths(node)
        flag = False
        for path in all_paths:
            if layer_node_number[path[-1].depth + 1] < max_node_num_per_layer:
                flag = True
                break
        return flag

    def extract_paths(self, root):
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

    def correct_nodes_num(self, node):
        count = 0
        children_queue = [node]
        while(children_queue):
            new_children_queue = []
            for elem in children_queue:
                if elem.is_terminal and elem.reward_thought == 1:
                    count += 1
                new_children_queue.extend(elem.children)
            children_queue = new_children_queue

        return count

    def best_child(self, layer_node_number, max_node_num_per_layer, exploration_param=math.sqrt(2)):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            if child.is_terminal:
                continue
            if child.visit_count == 0:
                ucb_score = float('inf')
            else:
                ucb_score = (child.q_value_search / child.visit_count) + exploration_param * math.sqrt(
                    math.log(self.visit_count) / child.visit_count
                )
            if ucb_score > best_score and self.has_non_terminal_node(child) and self.has_ok_nodes(child, layer_node_number, max_node_num_per_layer):
                best_score = ucb_score
                best_child = child
        return best_child

    def to_dict(self):
        return {
            'query': self.query,
            'answer': self.answer,
            'parent': self.parent.id if self.parent else None,
            'children': [child.to_dict() for child in self.children],
            'id': self.id,
            'visit_count': self.visit_count,
            'q_value_thought': self.q_value_thought,
            'q_value_search': self.q_value_search,
            'reward_thought': self.reward_thought,
            'reward_search': self.reward_search,
            'thought': self.thought,
            'observation': self.observation,
            'is_terminal': self.is_terminal,
            'is_root': self.is_root,
            'unique_identifier': self.unique_identifier
        }


class MCTS:
    def __init__(self, root, llm, local_llm_ip, local_llm_port, temperature, retriever_ip, retriever_port, max_expansion_num=3, max_depth=10, max_node_num_per_layer=6, verbose=True):
        self.root = root
        self.max_depth = max_depth
        self.node_count = 1
        self.verbose = verbose
        self.llm = llm
        self.max_expansion_num = max_expansion_num
        self.layer_node_number = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.max_node_num_per_layer = max_node_num_per_layer
        self.temperature = temperature
        self.retriever_ip = retriever_ip
        self.retriever_port = retriever_port

        self.local_llm_ip = local_llm_ip
        self.local_llm_port = local_llm_port

    def search(self):
        node = self.root
        depth = 0

        # Selection
        while node and node.is_fully_expanded() and depth < self.max_depth:
            node = node.best_child(self.layer_node_number, self.max_node_num_per_layer)
            if self.verbose and node:
                print(f'Choose:{node.id}\n')
            depth += 1

        # 判断是否扩展完了所有节点
        if not node or depth == self.max_depth:
            return False

        # Expansion
        if self.verbose:
            print('Expanding.....................')
        new_queries = self.expand_query(node)
        child_nodes = []
        for q in new_queries:
            child_node = MCTSNode(False, self.node_count, q['Query'], None, q['Thought'], q['Observation'], node.depth+1, parent=node)
            self.layer_node_number[node.depth+1] += 1
            self.node_count += 1
            node.children.append(child_node)
            child_nodes.append(child_node)

        # Simulation
        all_estimated_thought_value, all_estimated_search_value = self.parallel_simulate(child_nodes)
        for index, child in enumerate(child_nodes):
            reward_thought = all_estimated_thought_value[index]
            reward_search = all_estimated_search_value[index]
            child.q_value_thought = 0
            child.q_value_search = 0
            try:
                child.reward_thought = float(reward_thought) # 更新当前节点的 Q 值
                child.reward_search = float(reward_search) # 更新当前节点的 Q 值
            except:
                child.reward_thought = 0  # 更新当前节点的 Q 值
                child.reward_search = 0  # 更新当前节点的 Q 值

        # Backpropagation
        if self.verbose:
            print('Backpropagating.....................')
        for child in child_nodes:
            self.backpropagate(child, child.reward_thought, child.reward_search)

        # self.visualize_tree(self.root)
        return True

    def expand_query(self, node):
        """
        生成子查询。
        """
        if node.is_root:
            possible_plans = parallel_call_llm(prompt_expand_initial_one_query.replace('{query}', self.root.query), self.llm, self.local_llm_ip, self.local_llm_port, num=self.max_expansion_num, temperature=self.temperature)
        else:
            current_reasoning, reasoning_history, all_reasoning_process = self.get_reasoning_process(node)
            possible_plans = parallel_call_llm(prompt_expand_subsequent_one_query.replace('{query}', self.root.query).replace('{history}', all_reasoning_process), self.llm, self.local_llm_ip, self.local_llm_port, num=self.max_expansion_num, temperature=self.temperature)

        plans = self.parse_plans_and_search(possible_plans)

        return plans

    def simulate(self, node, index):
        """
        使用API模拟单步roll-out，并评估是否为终止状态。
        """
        if 'finish(' in node.thought.lower():
            pred = node.thought.split('Finish(')[1].split(')')[0]
            estimated_value = call_llm(prompt_evaluate_final.replace('{query}', self.root.query).replace('{pred}', pred).replace('{gt}', self.root.answer), self.llm, self.local_llm_ip, self.local_llm_port)
            thought_reward, search_reward = estimated_value, estimated_value
            self.layer_node_number[node.depth + 1] += 1
            node.is_terminal = True
        else:
            current_reasoning, reasoning_history, all_reasoning_process = self.get_reasoning_process(node)
            estimated_value = call_llm(prompt_evaluate_intermediate.replace('{query}', self.root.query).replace('{history}', all_reasoning_process), self.llm, self.local_llm_ip, self.local_llm_port)
            matches = re.findall(r'the (thought|search result) is ###(-?\d+\.?\d*)###', estimated_value)
            thought_reward, search_reward = float(matches[0][1]) / 2,  float(matches[1][1]) / 2

        return thought_reward, search_reward, index

    def extract_value(self, states):
        all_value = []
        for state in states:
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
            all_value.append(eval(str(number)))
        return np.mean(all_value)

    def backpropagate(self, node, reward_thought, reward_search):
        """
        回溯更新节点路径上的Q值和访问次数。
        """
        while node is not None:
            node.visit_count += 1
            node.q_value_thought += (reward_thought - node.q_value_thought) / node.visit_count  # 更新Q值
            node.q_value_search += (reward_search - node.q_value_search) / node.visit_count  # 更新Q值
            node = node.parent

    def get_reasoning_process(self, node):
        current_reasoning = f"- **Thought:** {node.thought}\n- **Search Query:** {node.query}\n- **Search Result:** {node.observation}"
        node = node.parent

        reasoning_history_list = []
        while node.is_root is not True:
            reasoning_history_list.append(node)
            node = node.parent

        if len(reasoning_history_list) == 0:
            reasoning_history = 'None'
        else:
            reasoning_history = '\n\n'.join([f"## Step {index+1}:\n- **Thought:** {elem.thought}\n- **Search Query:** {elem.query}\n- **Search Result:** {elem.observation}" for index, elem in enumerate(reasoning_history_list[::-1])])

        if reasoning_history == 'None':
            all_reasoning_process = f"## Step 1:\n" + current_reasoning
        else:
            all_reasoning_process = reasoning_history + f"\n\n## Step {len(reasoning_history_list) + 1}:\n" + current_reasoning

        return current_reasoning, reasoning_history, all_reasoning_process

    def parallel_simulate(self, nodes):
        if self.verbose:
            print('Simulating...')
        all_estimated_thought_value = [None for _ in range(len(nodes))]
        all_estimated_search_value = [None for _ in range(len(nodes))]
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.simulate, node, index) for index, node in enumerate(nodes)]
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    thought_reward, search_reward, index = future.result()
                    all_estimated_thought_value[index] = thought_reward
                    all_estimated_search_value[index] = search_reward
                except Exception as e:
                    if self.verbose:
                        print(e)
        return all_estimated_thought_value, all_estimated_search_value

    def parallel_search(self, queries):
        if self.verbose:
            print('Searching...')
        all_search_result = [{'ctxs': [{'snippet': "No Information Found"}]} for _ in range(len(queries))]
        all_trials_times = 0
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(retrieve_from_api, self, query, index) for index, query in enumerate(queries)]
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    result, index, trial_times = future.result()
                    all_search_result[index] = result
                    all_trials_times += trial_times
                except Exception as e:
                    print(e)
        return all_search_result, all_trials_times

    def parse_plans_and_search(self, possible_plans):
        import re
        matches = []
        for plan in possible_plans:
            if 'finish(' in plan.lower():
                matches.append([plan, plan])
                continue

            pattern = r"(?:- )?\*\*Thought:\*\*\s*([\s\S]*?)\n\s*(?:- )?\*\*Action:\*\*\s*Search\((\[[\s\S]*?\])\)"
            match = re.findall(pattern, plan, re.DOTALL)
            count = 0
            while len(match) == 0 and count < 3:
                text = call_llm(prompt_parse.replace('{output}', plan), self.llm, self.local_llm_ip, self.local_llm_port)
                match = re.findall(pattern, text, re.DOTALL)
                count += 1
            try:
                matches.append(list(match[0]))
            except:
                pass

        # matches = [list(elem) for elem in matches]
        for match in matches:
            match[1] =  match[1].strip().strip('"')

        # Parse the results into a structured format
        plans = []

        try:
            all_search_result, all_trial_times = self.parallel_search([eval(match[1]) for match in matches])
        except Exception as e:
            for elem in matches:
                try:
                    eval(elem[1])
                except:
                    elem[1] = str([tmp.replace('[','').replace(']','').replace('"','') for tmp in elem[1].split(',')])

            all_search_result, all_trial_times = self.parallel_search([eval(match[1]) for match in matches])

        for i, match in enumerate(matches):
            thought, action = match
            search_result = all_search_result[i]
            plan = {
                "Plan": i,
                "Thought": thought.strip(),
                "Query": action.strip(),
                "Observation": search_result,
            }
            plans.append(plan)

        return plans

    def visualize_tree(self, root_node):
        """
        基于现有的 MCTSNode 树结构直接进行可视化。
        - 节点显示 Query 和 Reward。
        - 边显示 Q 值。
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        from textwrap import wrap

        # 使用 NetworkX 构建图结构（只在这里用于可视化）
        tree = nx.DiGraph()

        # 遍历树结构并添加节点和边
        def add_nodes_edges(node):
            # 添加当前节点，动态读取 reward 和 q_value
            tree.add_node(
                node.unique_identifier,
                reward=node.reward_thought,
                visit_count=node.visit_count,
                q_value=node.q_value_thought,
                query=node.query  # 添加 query 信息
            )
            # 添加子节点和边
            for child in node.children:
                tree.add_edge(node.unique_identifier, child.unique_identifier)
                add_nodes_edges(child)

        add_nodes_edges(root_node)  # 从根节点递归添加

        # 定义布局
        pos = nx.nx_agraph.graphviz_layout(tree, prog="dot")  # 使用层次布局

        # 定义节点颜色（默认颜色，如果 unique_identifier 中包含 'finish' 则为蓝色）
        node_colors = []
        for node, data in tree.nodes(data=True):
            if 'finish' in data['query'].lower():
                if data['reward'] == 1:
                    node_colors.append('lightgreen')  # 设为蓝色
                else:
                    node_colors.append('lightcoral')
            else:
                node_colors.append('lightblue')  # 默认颜色

        # 绘制节点
        plt.figure(figsize=(56, 48))
        nx.draw(
            tree,
            pos,
            with_labels=False,  # 节点本身不显示默认标签
            node_size=50000,
            node_color=node_colors,
            font_size=20,
            font_color="black",
            arrowsize=40,
        )

        # 自定义节点标签（显示 Query 和 Reward）
        custom_labels = {}
        for node, data in tree.nodes(data=True):
            query = node  # 节点的 Query 内容
            reward = data['reward']  # 节点的即时 Reward
            # 自动换行处理 Query
            wrapped_query = "\n".join(wrap(query, width=20))  # 每行最多 20 个字符
            custom_labels[node] = f"{wrapped_query}\nR: {reward:.2f}"  # 节点显示 Query 和 Reward

        # 在节点上绘制标签
        nx.draw_networkx_labels(
            tree,
            pos,
            labels=custom_labels,
            font_size=20,  # 调整节点文本大小
            font_color="black",
        )

        # 绘制边
        nx.draw_networkx_edges(
            tree,
            pos,
            width=4,
            arrowstyle='->',
            arrowsize=40,
        )

        # 添加边标签（显示 Q 值）
        edge_labels = {
            (u, v): f"Q: {tree.nodes[v]['q_value']:.2f}"  # 从目标节点动态读取 Q 值
            for u, v in tree.edges()
        }
        nx.draw_networkx_edge_labels(
            tree,
            pos,
            edge_labels=edge_labels,
            font_size=40,  # 边文本字体大小
            bbox=dict(facecolor="white", edgecolor="none"),  # 白色背景框提高可读性
            label_pos=0.5,  # 标签位置：0（起点）到1（终点）
            rotate=False,  # 标签不旋转
            clip_on=False,  # 标签不被边框裁剪
        )

        plt.title("MCTS Tree Visualization with Edge Q-Values")

        plt.show()
        # plt.close('all')

        return plt  # 返回