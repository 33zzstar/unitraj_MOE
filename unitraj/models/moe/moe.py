import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np
from torch import optim
from torch.distributions import MultivariateNormal, Laplace
from torch.optim.lr_scheduler import MultiStepLR

from unitraj.models.moe.router import *
from unitraj.models.moe.model_dapter import *
from unitraj.models.base_model.base_model import BaseModel
from unitraj.models.wayformer.wayformer_utils import PerceiverEncoder
# from models import build_model
class TrajAttentionRouter(BaseModel):

    def __init__(self, config):
        super(TrajAttentionRouter, self).__init__(config)
        self.config = config
        self.d_k = config['hidden_size']
        self.past_T = config['past_len']
        self.map_attr = config['num_map_feature']
        self.k_attr = config['num_agent_feature']
        self.num_queries_enc = config['num_queries_enc']
        self.num_queries_dec = config['num_queries_dec']
        self.max_num_roads = config['max_num_roads']
        self.num_experts = config['num_experts']
        self._M = config['max_num_agents'] 
        

        
        self.perceiver_encoder = PerceiverEncoder(192, self.d_k,
                                                 num_cross_attention_qk_channels=self.d_k,
                                                  num_cross_attention_v_channels=self.d_k,
                                                  num_self_attention_qk_channels=self.d_k,
                                                  num_self_attention_v_channels=self.d_k)
        self.selu = nn.SELU(inplace=True) 

        self.mlp = nn.Sequential(
            nn.Linear(self.d_k, self.d_k),
            nn.PReLU(),

            nn.Linear(self.d_k, self.d_k // 2),
            nn.PReLU(),

            nn.Linear(self.d_k // 2, self.num_experts)
        )
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))
        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self._M + 1), self.d_k)),
            requires_grad=True
        )
        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.past_T, 1, self.d_k)),
            requires_grad=True
        )
        self.road_pts_lin = nn.Sequential(init_(nn.Linear(self.map_attr, self.d_k)))
    
    def process_observations(self, ego, agents):
        '''
        :param observations: (B, T, N+2, A+1) where N+2 is [ego, other_agents, env]
        :return: a tensor of only the agent dynamic states, active_agent masks and env masks.
        '''
        # ego stuff
        ego_tensor = ego[:, :, :self.k_attr]
        env_masks_orig = ego[:, :, -1]
        env_masks = (1.0 - env_masks_orig).to(torch.bool)
        env_masks = env_masks.unsqueeze(1).repeat(1, self.num_queries_dec, 1).view(ego.shape[0] * self.num_queries_dec,
                                                                                   -1)

        # Agents stuff
        temp_masks = torch.cat((torch.ones_like(env_masks_orig.unsqueeze(-1)), agents[:, :, :, -1]), dim=-1)
        opps_masks = (1.0 - temp_masks).to(torch.bool)  # only for agents.
        opps_tensor = agents[:, :, :, :self.k_attr]  # only opponent states

        return ego_tensor, opps_tensor, opps_masks, env_masks


    def forward(self, x):
        inputs = x['input_dict']
        agents_in, agents_mask, roads = inputs['obj_trajs'], inputs['obj_trajs_mask'], inputs['map_polylines']
        ego_in = torch.gather(agents_in, 1, inputs['track_index_to_predict'].view(-1, 1, 1, 1).repeat(1, 1,
                                                                                                      *agents_in.shape[
                                                                                                       -2:])).squeeze(1)
        ego_mask = torch.gather(agents_mask, 1, inputs['track_index_to_predict'].view(-1, 1, 1).repeat(1, 1,
                                                                                                       agents_mask.shape[
                                                                                                           -1])).squeeze(
            1)
        agents_in = torch.cat([agents_in, agents_mask.unsqueeze(-1)], dim=-1)
        agents_in = agents_in.transpose(1, 2)
        ego_in = torch.cat([ego_in, ego_mask.unsqueeze(-1)], dim=-1)
        roads = torch.cat([inputs['map_polylines'], inputs['map_polylines_mask'].unsqueeze(-1)], dim=-1)


        B = ego_in.size(0)
        num_agents = agents_in.shape[2] + 1
        # Encode all input observations (k_attr --> d_k)
        ego_tensor, _agents_tensor, opps_masks_agents, env_masks = self.process_observations(ego_in, agents_in)
        agents_tensor = torch.cat((ego_tensor.unsqueeze(2), _agents_tensor), dim=2)
        agents_emb = self.selu(self.agents_dynamic_encoder(agents_tensor))
        agents_emb = (agents_emb + self.agents_positional_embedding[:, :,
                                   :num_agents] + self.temporal_positional_embedding).view(B, -1, self.d_k)
        road_pts_feats = self.selu(self.road_pts_lin(roads[:, :self.max_num_roads, :, :self.map_attr]).view(B, -1,
                                                                                                            self.d_k))# + self.map_positional_embedding
        mixed_input_features = torch.concat([agents_emb, road_pts_feats], dim=1)
        opps_masks_roads = (1.0 - roads[:, :self.max_num_roads, :, -1]).to(torch.bool)
        mixed_input_masks = torch.concat([opps_masks_agents.view(B, -1), opps_masks_roads.view(B, -1)], dim=1)
        # Process through Wayformer's encoder

        context = self.perceiver_encoder(mixed_input_features, mixed_input_masks)                # [B,192,256]
        pooled = context.mean(dim=1)               # [B,256]
        out = self.mlp(pooled)                  # [B,num_experts]
        return F.softmax(out, dim=-1)             # [B,num_experts]
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=self.config['learning_rate_sched'], gamma=0.5,
                                verbose=True)
        return [optimizer], [scheduler]




class MOE(BaseModel):
    def __init__(self, config, init_cfg=None):
        from models import build_model
        super(MOE, self).__init__(config)
        
        self.experts = nn.ModuleList([build_model(cfg) for cfg in config.experts_cfg])
        self.router = TrajAttentionRouter(config)
        self.k = config.router['k']
        self.config = config
        self.d_k = config['hidden_size']
        self.past_T = config['past_len']
        self.map_attr = config['num_map_feature']
        self.k_attr = config['num_agent_feature']
        self.num_queries_enc = config['num_queries_enc']
        self.num_queries_dec = config['num_queries_dec']
        self.max_num_roads = config['max_num_roads']
        self.num_experts = config['num_experts']
        self._M = config['max_num_agents'] 
        

        
        self.perceiver_encoder = PerceiverEncoder(192, self.d_k,
                                                 num_cross_attention_qk_channels=self.d_k,
                                                  num_cross_attention_v_channels=self.d_k,
                                                  num_self_attention_qk_channels=self.d_k,
                                                  num_self_attention_v_channels=self.d_k)
        self.selu = nn.SELU(inplace=True) 

        self.mlp = nn.Linear(256, self.num_experts)
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.agents_dynamic_encoder = nn.Sequential(init_(nn.Linear(self.k_attr, self.d_k)))
        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self._M + 1), self.d_k)),
            requires_grad=True
        )
        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, self.past_T, 1, self.d_k)),
            requires_grad=True
        )
        self.road_pts_lin = nn.Sequential(init_(nn.Linear(self.map_attr, self.d_k)))


    def forward(self, x):
        def split_batch(batch):
            input=batch['input_dict']
            bs = next(iter(input.values())).shape[0]
            return [{k: v[i:i+1] for k, v in input.items()} for i in range(bs)]
        def merge_splits(splits, idx, batch_template=None):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            if isinstance(idx, int):
                idx = [idx]

            def cat_vals(vals):
                v0 = vals[0]
                if isinstance(v0, torch.Tensor):
                    return torch.cat(vals, dim=0)
                elif isinstance(v0, np.ndarray):
                    # 先转成 tensor
                    return np.concatenate(vals, axis=0)
                elif isinstance(v0, list):
                    return [cat_vals([v[j] for v in vals]) for j in range(len(v0))]
                elif isinstance(v0, str):
                    return vals          # 或 vals[0]
                else:
                    raise TypeError(f"Unsupported type: {type(v0)}")

            merged = {}
            for k in splits[0].keys():
                vals = [splits[i][k] for i in idx]
                merged[k] = cat_vals(vals)

            batch = {
                'batch_size': len(idx),
                'input_dict': merged,
                'batch_sample_count': len(idx)
            }
            if isinstance(batch_template, dict):
                for k, v in batch_template.items():
                    if k not in batch:
                        batch[k] = v
            return batch

        # Obtain experts routing probabilities
        routing_probs = self.router(x)
        self.routing_probs = routing_probs
        B = routing_probs.size(0)

    

        # print(routing_probs)
        self.last_routing_probs = routing_probs
        #决定使用哪些专家
        if self.training:
            # Use all experts
            weights = routing_probs
            indices = torch.arange(routing_probs.size(1), device=routing_probs.device).repeat(routing_probs.size(0), 1)
        else:
            # Use top-k experts
            weights, indices = torch.topk(routing_probs, k=self.k, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)

        expert_indices, counts = torch.unique(indices, return_counts=True)
        for i, count in zip(expert_indices, counts):
            expert_name = getattr(self.experts[i], "name", f"Expert {i}")
            percentage = (count / indices.numel()) * 100
        #     print(f"Expert {i + 1}: {count} times")
        #     print(f"{expert_name}: selected {count.item()} times ({percentage:.2f}%)")
        # print()
  
        expert_output_predicted_trajectory = torch.zeros((B, 6, 60, 5),device=routing_probs.device)
        expert_output_predicted_probability = torch.zeros((B, 6),device=routing_probs.device)
        # 每个专家处理对应的样本
        for i, expert in enumerate(self.experts):
            idx, top = torch.where(indices == i)
            print("indices:", indices)
            print("i:", i)

            if idx.numel() == 0: #如果没有样本分配给这个专家
                continue
            # expert_inputs = x[idx]
            
            #拆分批次
            splits = split_batch(x)
            new_batch = merge_splits(splits, idx, batch_template=len(idx))

                        
                                    
            expert_output,expert_output_Loss = expert(new_batch)
            
            expert_type = type(expert).__name__
            if expert_type == "MotionTransformer":
                  expert_output_predicted_trajectory[idx] = expert_output['predicted_trajectory']
                  expert_output_predicted_probability[idx] = expert_output['predicted_probability']
                # expert_output_2 = self.conv_swint_2(expert_output[1])
                # expert_output_3 = self.conv_swint_3(expert_output[2])
            elif expert_type == "AutoBotEgo":
                  expert_output_predicted_trajectory[idx] = expert_output['predicted_trajectory']
                  expert_output_predicted_probability[idx] = expert_output['predicted_probability']
                #   expert_output_2 = expert_output[1]
                #   expert_output_3 = expert_output[2]
            elif expert_type == "Wayformer":
                expert_output_predicted_trajectory[idx] = expert_output['predicted_trajectory']
                expert_output_predicted_probability[idx] = expert_output['predicted_probability']
                # expert_output_2 = self.conv_pvt_2(expert_output[1])
                # expert_output_3 = self.conv_pvt_3(expert_output[2])
            elif expert_type == "Smart":
                expert_output_predicted_trajectory[idx] = expert_output['predicted_trajectory']
                expert_output_predicted_probability[idx] = expert_output['predicted_probability']
                # expert_output_2 = self.conv_convnext_2(expert_output[1])
                # expert_output_3 = self.conv_convnext_3(expert_output[2])
            else:
                raise ValueError(f"Unsupported expert type {expert_type}")
            
            # 加权组合专家输出
            w = weights[idx, top].view(-1, 1, 1, 1)

            # final_output_2[idx] += expert_output_2 * w
            # final_output_3[idx] += expert_output_3 * w
            expert_output['predicted_trajectory'] = expert_output_predicted_trajectory
            expert_output['predicted_probability'] = expert_output_predicted_probability
        # return [final_output_1, final_output_2, final_output_3]
        return expert_output,expert_output_Loss,routing_probs




    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=self.config['learning_rate_sched'], gamma=0.5,
                                verbose=True)
        return [optimizer], [scheduler]
def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
# 封装了MoE模型

class MoENetwork(BaseModel):
    def __init__(self, num_experts, in_channels, experts_cfg, router, init_cfg=None):
        super(MoENetwork, self).__init__(init_cfg)
        self.moe = MOE(num_experts, in_channels, experts_cfg, router)

        self.lb_loss_weight_initial = 0.01
        self.lb_loss_decay_rate = 0.95
        self.lb_loss_weight = self.lb_loss_weight_initial
    
    def forward(self, x):
        return self.moe(x) 
    # 取路由概率
    def get_routing_probs(self):
        return self.moe.last_routing_probs
    # 包含负载平衡(load balancing)损失权重更新
    def update_lb_loss_weight(self, epoch):
        self.lb_loss_weight = self.lb_loss_weight_initial * (self.lb_loss_decay_rate ** epoch)
        print(f"lb_loss_weight updated to {self.lb_loss_weight}")
    def load_balance_loss(routing_probs):
        expert_mean = routing_probs.mean(dim=0)
        loss = (expert_mean * routing_probs.sum(dim=0)).sum()
        return loss
