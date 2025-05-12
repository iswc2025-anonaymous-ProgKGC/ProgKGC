import os
import json

from typing import List
from dataclasses import dataclass
from collections import deque

from openai import OpenAI
import time
from logger_config import logger
from collections import defaultdict
import collections


# deepseek_url = "https://api.deepseek.com/v1"
# deepseek_api = "sk-13670473d25a4dcf8e3c7ed8a557a67a"
# deepseek_model = "deepseek-chat"

# class OpenAIInterface:
#     def __init__(self):
#         self.client = OpenAI(api_key=deepseek_api,
#                              base_url=deepseek_url)

#     def predict_text_logged(self, prompt, temp=1):
#         """
#         Queries OpenAI's GPT-3 model given the prompt and returns the prediction.
#         """
#         n_prompt_tokens = 0
#         n_completion_tokens = 0
#         start_query = time.perf_counter()
#         content = "-1"

#         message = [{"role": "user", "content": prompt}]
#         response = self.client.chat.completions.create(
#             model=deepseek_model, messages=message, temperature=temp
#         )
#         n_prompt_tokens = response.usage.prompt_tokens
#         n_completion_tokens = response.usage.completion_tokens
#         # end_query = time.perf_counter()
#         print(f"response.choices[0]:{response.choices[0]}")
#         content = response.choices[0].message.content
#         print(f"content:{content}")
#         end_query = time.perf_counter()

#         response_time = end_query - start_query
#         return {
#             "prompt": prompt,
#             "content": content,
#             "n_prompt_tokens": n_prompt_tokens,
#             "n_completion_tokens": n_completion_tokens,
#             "response_time": response_time,
#         }


# openai_interface = OpenAIInterface()

@dataclass
class EntityExample:
    entity_id: str
    entity: str
    entity_desc: str = ''


class TripletDict:

    def __init__(self, path_list: List[str]):
        self.path_list = path_list
        logger.info('Triplets path: {}'.format(self.path_list))
        self.relations = set()
        self.hr2tails = {}
        self.triplet_cnt = 0

        for path in self.path_list:
            self._load(path)
        logger.info('Triplet statistics: {} relations, {} triplets'.format(len(self.relations), self.triplet_cnt))

    def _load(self, path: str):
        examples = json.load(open(path, 'r', encoding='utf-8'))
        examples += [reverse_triplet(obj) for obj in examples]
        for ex in examples:
            self.relations.add(ex['relation'])
            key = (ex['head_id'], ex['relation'])
            if key not in self.hr2tails:
                self.hr2tails[key] = set()
            self.hr2tails[key].add(ex['tail_id'])
        self.triplet_cnt = len(examples)

    def get_neighbors(self, h: str, r: str) -> set:
        return self.hr2tails.get((h, r), set())


class EntityDict:

    def __init__(self, entity_dict_dir: str, inductive_test_path: str = None):
        path = os.path.join(entity_dict_dir, 'entities.json')
        assert os.path.exists(path)
        self.entity_exs = [EntityExample(**obj) for obj in json.load(open(path, 'r', encoding='utf-8'))]

        if inductive_test_path:
            examples = json.load(open(inductive_test_path, 'r', encoding='utf-8'))
            valid_entity_ids = set()
            for ex in examples:
                valid_entity_ids.add(ex['head_id'])
                valid_entity_ids.add(ex['tail_id'])
            self.entity_exs = [ex for ex in self.entity_exs if ex.entity_id in valid_entity_ids]

        self.id2entity = {ex.entity_id: ex for ex in self.entity_exs}
        self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)}
        logger.info('Load {} entities from {}'.format(len(self.id2entity), path))

    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]

    def get_entity_by_id(self, entity_id: str) -> EntityExample:
        return self.id2entity[entity_id]

    def get_entity_by_idx(self, idx: int) -> EntityExample:
        return self.entity_exs[idx]

    def __len__(self):
        return len(self.entity_exs)


# ********************************新增代码******************************
# 预处理阶段构建共尾实体图
class CoTailGraph_real:
    def __init__(self, train_path):
        # 存储每个尾实体对应的所有头实体
        triples = json.load(open(train_path, 'r', encoding='utf-8'))
        self.tail_to_heads = defaultdict(set)
        for tri in triples:
            h, t = tri['head_id'], tri['tail_id']
            self.tail_to_heads[t].add(h)

        # 构建邻接表：头实体 -> 共尾的其他头实体
        self.adjacency = defaultdict(set)
        for t, heads in self.tail_to_heads.items():
            for h1 in heads:
                for h2 in heads:
                    if h1 != h2:
                        self.adjacency[h1].add(h2)

    def get_cotail_neighbors(self, head_id):
        """获取与head_id共享尾实体的其他头实体"""
        return list(self.adjacency.get(head_id, set()))


class CoTailGraph:
    def __init__(self, train_path):
        # 加载训练集中的三元组
        triples = json.load(open(train_path, 'r', encoding='utf-8'))

        # 构建邻接表：实体 -> 直接相连的一阶邻居实体
        self.adjacency = defaultdict(set)
        for tri in triples:
            h, t = tri['head_id'], tri['tail_id']
            # 添加双向邻居关系（无向图）
            self.adjacency[h].add(t)
            self.adjacency[t].add(h)

    def get_cotail_neighbors(self, entity_id):
        """获取与指定实体直接相连的一阶邻居实体"""
        return list(self.adjacency.get(entity_id, set()))


class DynamicCache:
    def __init__(self, max_size=1000000):
        self.max_size = max_size
        self.tail_cache = collections.OrderedDict()
        self.hr_cache = collections.OrderedDict()

    def update_hr(self, hr_keys, hr_vectors):
        for key, vec in zip(hr_keys, hr_vectors):
            if key in self.hr_cache:
                self.hr_cache[key] = vec.detach().cpu()
                self.hr_cache.move_to_end(key)
            else:
                if len(self.hr_cache) >= self.max_size:
                    self.hr_cache.popitem(last=False)
                self.hr_cache[key] = vec.detach().cpu()

    def update_tail(self, tail_keys, tail_vectors):
        for key, vec in zip(tail_keys, tail_vectors):
            if key in self.tail_cache:
                self.tail_cache[key] = vec.detach().cpu()
                self.tail_cache.move_to_end(key)
            else:
                if len(self.tail_cache) >= self.max_size:
                    self.tail_cache.popitem(last=False)
                self.tail_cache[key] = vec.detach().cpu()

    def get_hr_vectors(self, keys):
        """批量获取向量并更新访问位置（LRU策略）"""
        vectors = []
        for key in keys:
            if key in self.hr_cache:
                # 标记为最近使用并获取值
                self.hr_cache.move_to_end(key)
                vectors.append(self.hr_cache[key])
            else:
                vectors.append(None)
        return vectors

    def get_tail_vectors(self, keys):
        """批量获取向量并更新访问位置（LRU策略）"""
        vectors = []
        for key in keys:
            if key in self.tail_cache:
                # 标记为最近使用并获取值
                self.tail_cache.move_to_end(key)
                vectors.append(self.tail_cache[key])
            else:
                vectors.append(None)
        return vectors


# ********************************新增代码******************************

class LinkGraph:

    def __init__(self, train_path: str):
        logger.info('Start to build link graph from {}'.format(train_path))
        # id -> set(id)
        self.graph = {}
        examples = json.load(open(train_path, 'r', encoding='utf-8'))
        for ex in examples:
            head_id, tail_id = ex['head_id'], ex['tail_id']
            if head_id not in self.graph:
                self.graph[head_id] = set()
            self.graph[head_id].add(tail_id)
            if tail_id not in self.graph:
                self.graph[tail_id] = set()
            self.graph[tail_id].add(head_id)
        logger.info('Done build link graph with {} nodes'.format(len(self.graph)))

    def get_neighbor_ids(self, entity_id: str, max_to_keep=10) -> List[str]:
        # make sure different calls return the same results
        neighbor_ids = self.graph.get(entity_id, set())
        return sorted(list(neighbor_ids))[:max_to_keep]

    def get_n_hop_entity_indices(self, entity_id: str,
                                 entity_dict: EntityDict,
                                 n_hop: int = 2,
                                 # return empty if exceeds this number
                                 max_nodes: int = 100000) -> set:
        if n_hop < 0:
            return set()

        seen_eids = set()
        seen_eids.add(entity_id)
        queue = deque([entity_id])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.graph.get(tp, set()):
                    if node not in seen_eids:
                        queue.append(node)
                        seen_eids.add(node)
                        if len(seen_eids) > max_nodes:
                            return set()
        return set([entity_dict.entity_to_idx(e_id) for e_id in seen_eids])


def reverse_triplet(obj):
    return {
        'head_id': obj['tail_id'],
        'head': obj['tail'],
        'relation': 'inverse {}'.format(obj['relation']),
        'tail_id': obj['head_id'],
        'tail': obj['head']
    }

# def reverse_triplet(obj, openai_interface: OpenAIInterface):
#     """
#     Generate the inverse relation using a language model.
#     """
#     head = obj['head']
#     relation = obj['relation']
#     tail = obj['tail']

#     # Construct a prompt to get the inverse relation
#     prompt = f"Given the following relationship: Head: '{head}', Tail: '{tail}', Relation: '{relation}', generate the inverse by swapping 'head' and 'tail' and reversing 'relation'.\n\n For Example: Head: 'Inception', Tail: 'Christopher Nolan', Relation: 'directed by' => Inverse: 'directs'. Please provide only the inverse relation without any additional explanation or entity details."

#     # Call the language model to predict the inverse relation
#     response = openai_interface.predict_text_logged(prompt)

#     # Extract the inverse relation from the model's response
#     inverse_relation = response['content'].strip().replace("'", "").replace("‘", "").replace("’", "")

#     return {
#         'head_id': obj['tail_id'],
#         'head': tail,
#         'relation': inverse_relation,
#         'tail_id': obj['head_id'],
#         'tail': head
#     }