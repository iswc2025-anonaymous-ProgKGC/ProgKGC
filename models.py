from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from dataclasses import dataclass
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoModel, AutoConfig
from dict_hub import get_cotail_graph, get_dynamic_cache, get_cotail_graph_valid
from triplet_mask import construct_mask
import torch.nn.functional as F


def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


class GNNLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # 多头注意力层（PyTorch默认维度顺序：seq_len, batch, embed_dim）
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # 确保输入为 (seq_len, batch, dim)
        )

        # 前馈层 + 归一化
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化所有线性层参数"""
        for layer in [self.attention, self.ffn]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, query: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [D] 当前实体的原始向量
            neighbors: [K, D] 邻居实体向量（K=0时返回原始向量）
        Returns:
            融合后的实体向量 [D]
        """
        if neighbors.size(0) == 0:  # 空邻居直接返回原始向量
            return query

        # ================= 维度调整 =================
        # query: [D] -> [1, 1, D] -> (seq_len=1, batch=1, dim)
        # neighbors: [K, D] -> [K, 1, D] -> (seq_len=K, batch=1, dim)
        query_3d = query.view(1, 1, -1)
        neighbors_3d = neighbors.unsqueeze(1)  # [K, 1, D]

        # ================= 多头注意力 =================
        attn_output, _ = self.attention(
            query=query_3d,  # [1, 1, D]
            key=neighbors_3d,  # [K, 1, D]
            value=neighbors_3d,  # [K, 1, D]
            need_weights=False
        )
        # 残差连接 + 归一化
        query_3d = self.norm1(query_3d + self.dropout(attn_output))  # [1, 1, D]

        # ================= 前馈层 =================
        ffn_output = self.ffn(query_3d)
        output = self.norm2(query_3d + self.dropout(ffn_output))  # [1, 1, D]

        return output.squeeze(0).squeeze(0)  # [D]


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained("model/bert-base-uncased")
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained("model/bert-base-uncased")
        self.tail_bert = deepcopy(self.hr_bert)
        self.hr_gnn = GNNLayer(self.config.hidden_size)
        self.tail_gnn = GNNLayer(self.config.hidden_size)

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                use_gnn=True,
                use_head_gnn=True,  # 新增参数：是否融合头实体
                use_tail_gnn=False,  # 新增参数：是否融合尾实体
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_mask,
                                              tail_token_type_ids=tail_token_type_ids)

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        # ========== GNN特征融合 ==========

        if not use_gnn:
            return {'hr_vector': hr_vector,
                    'tail_vector': tail_vector,
                    'head_vector': head_vector}

        if use_gnn and not self.training:
            cache = get_dynamic_cache()
            batch_data = kwargs.get('batch_data', [])

            # 处理HR向量 - 只在use_head_gnn=True时执行
            if use_head_gnn:
                updated_hr = []
                for i, ex in enumerate(batch_data):
                    # 获取当前头实体的共尾实体ID
                    cotail_heads = get_cotail_graph_valid().get_cotail_neighbors(ex.head_id)
                    # 从缓存获取对应的HR向量
                    neighbor_vectors = [vec for vec in cache.get_hr_vectors(cotail_heads) if vec is not None]
                    if neighbor_vectors:
                        neighbor_tensor = torch.stack(neighbor_vectors).to(hr_vector.device)
                        updated = self.hr_gnn(hr_vector[i], neighbor_tensor)
                        updated_hr.append(updated)
                    else:
                        updated_hr.append(hr_vector[i])
                hr_vector = torch.stack(updated_hr)

            # 处理Tail向量 - 只在use_tail_gnn=True时执行
            if use_tail_gnn:
                updated_tail = []
                for i, ex in enumerate(batch_data):
                    # 获取当前尾实体的共尾实体ID
                    cotail_tails = get_cotail_graph_valid().get_cotail_neighbors(ex.tail_id)
                    # 从缓存获取对应的Tail向量
                    neighbor_vectors = [vec for vec in cache.get_tail_vectors(cotail_tails) if vec is not None]
                    if neighbor_vectors:
                        neighbor_tensor = torch.stack(neighbor_vectors).to(tail_vector.device)
                        updated = self.tail_gnn(tail_vector[i], neighbor_tensor)
                        updated_tail.append(updated)
                    else:
                        updated_tail.append(tail_vector[i])
                tail_vector = torch.stack(updated_tail)

        if use_gnn and self.training:
            cache = get_dynamic_cache()
            batch_data = kwargs.get('batch_data', [])

            # 处理HR向量 - 只在use_head_gnn=True时执行
            if use_head_gnn:
                updated_hr = []
                for i, ex in enumerate(batch_data):
                    # 获取当前头实体的共尾实体ID
                    cotail_heads = get_cotail_graph().get_cotail_neighbors(ex.head_id)
                    # 从缓存获取对应的HR向量
                    neighbor_vectors = [vec for vec in cache.get_hr_vectors(cotail_heads) if vec is not None]
                    if neighbor_vectors:
                        neighbor_tensor = torch.stack(neighbor_vectors).to(hr_vector.device)
                        updated = self.hr_gnn(hr_vector[i], neighbor_tensor)
                        updated_hr.append(updated)
                    else:
                        updated_hr.append(hr_vector[i])
                hr_vector = torch.stack(updated_hr)

            # 处理Tail向量 - 只在use_tail_gnn=True时执行
            if use_tail_gnn:
                updated_tail = []
                for i, ex in enumerate(batch_data):
                    # 获取当前尾实体的共尾实体ID
                    cotail_tails = get_cotail_graph().get_cotail_neighbors(ex.tail_id)
                    # 从缓存获取对应的Tail向量
                    neighbor_vectors = [vec for vec in cache.get_tail_vectors(cotail_tails) if vec is not None]
                    if neighbor_vectors:
                        neighbor_tensor = torch.stack(neighbor_vectors).to(tail_vector.device)
                        updated = self.tail_gnn(tail_vector[i], neighbor_tensor)
                        updated_tail.append(updated)
                    else:
                        updated_tail.append(tail_vector[i])
                tail_vector = torch.stack(updated_tail)

        hr_vector = F.normalize(hr_vector, p=2, dim=1)
        tail_vector = F.normalize(tail_vector, p=2, dim=1)
        # ========== 融合结束 ==========

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)
        # =========================新增代码=========================
        # 处理RS负样本
        if self.args.use_rs_negative and self.training:
            rs_neg_ids = batch_dict['rs_neg_token_ids']
            if rs_neg_ids.size(0) > 0:
                # 编码RS负样本向量
                rs_neg_vec = self._encode(
                    self.tail_bert,
                    rs_neg_ids,
                    batch_dict['rs_neg_mask'],
                    batch_dict['rs_neg_token_type_ids']
                )
                # 重塑为 (batch_size, K, hidden_size)
                K = rs_neg_vec.size(0) // batch_size
                rs_neg_vec = rs_neg_vec.view(batch_size, K, -1)
                # 计算相似度 (batch_size, K)
                rs_logits = torch.bmm(
                    hr_vector.unsqueeze(1),  # (B,1,D)
                    rs_neg_vec.transpose(1, 2)  # (B,D,K)
                ).squeeze(1) * self.log_inv_t.exp()
                # 合并到总logits
                logits = torch.cat([logits, rs_logits], dim=-1)
        # =========================新增代码=========================

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
