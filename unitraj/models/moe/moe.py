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

class TrajAttentionRouter(BaseModel):

    pass

def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class MOE(BaseModel):
    def __init__(self, config):
        super(MOE, self).__init__(config)
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


    def context (self,inputs):
        inputs = inputs['input_dict']
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

        context = self.perceiver_encoder(mixed_input_features, mixed_input_masks)
        
        return context

    def forward(self, x):
        context = self.context(x)                  # [B,192,256]
        pooled = context.mean(dim=1)               # [B,256]
        out = self.mlp(pooled)                  # [B,num_experts]
        return F.softmax(out, dim=-1)              # [B,num_experts]
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'], eps=0.0001)
        scheduler = MultiStepLR(optimizer, milestones=self.config['learning_rate_sched'], gamma=0.5,
                                verbose=True)
        return [optimizer], [scheduler]
