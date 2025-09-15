import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np
from unitraj.models import build_model
# from unitraj.models.builder import MODELS, build_model
# from mmcv.runner import BaseModule
from unitraj.models.moe.router import *
from unitraj.models.moe.model_dapter import *
@MODELS.register_module()
class UniTrajMixtureOfExperts(BaseModule):
    """
    UniTraj轨迹预测的Mixture of Experts模型
    
    关键设计：
    1. 动态专家选择与加权
    2. 多模态轨迹融合
    3. 负载均衡机制
    4. 自适应输出标准化
    """
    
    def __init__(self,
                 num_experts: int = 4,
                 expert_names: List[str] = ['autobot', 'mtr', 'smart', 'wayformer'],
                 experts_cfg: Optional[List[Dict]] = None,
                 router_cfg: Dict = None,
                 k: int = 2,  # top-k experts
                 fusion_strategy: str = 'weighted_average',  # 'weighted_average', 'attention', 'gating'
                 normalize_outputs: bool = True,
                 load_balancing: bool = True,
                 lb_loss_weight_initial: float = 0.01,
                 lb_loss_decay_rate: float = 0.95,
                 auxiliary_loss_weight: float = 0.1,
                 temperature: float = 1.0,
                 init_cfg: Optional[Dict] = None):
        
        super().__init__(init_cfg)
        
        self.num_experts = num_experts
        self.expert_names = expert_names
        self.k = k
        self.fusion_strategy = fusion_strategy
        self.normalize_outputs = normalize_outputs
        self.load_balancing = load_balancing
        self.temperature = temperature
        
        # ============ 1. 构建专家模型 ============
        if experts_cfg is None:
            experts_cfg = self._get_default_expert_configs()
            
        self.experts = nn.ModuleList()
        self.expert_adapters = nn.ModuleList()
        
        for i, (name, cfg) in enumerate(zip(expert_names, experts_cfg)):
            # 构建专家
            expert = build_model(cfg)
            self.experts.append(expert)
            
            # 构建适配器
            if name == 'autobot':
                adapter = AutoBotAdapter(hidden_dim=cfg.get('hidden_size', 256))
            elif name == 'mtr':
                adapter = MTRAdapter(hidden_dim=cfg.get('hidden_dim', 512))
            elif name == 'smart':
                adapter = SMARTAdapter(feature_dim=cfg.get('feature_dim', 384))
            elif name == 'wayformer':
                adapter = WayformerAdapter(d_model=cfg.get('d_model', 768))
            else:
                raise ValueError(f"Unknown expert: {name}")
            
            self.expert_adapters.append(adapter)
        
        # ============ 2. 构建路由器 ============
        if router_cfg is None:
            router_cfg = {
                'type': 'UniTrajAttentionRouter',
                'embed_dim': 128,
                'num_heads': 4
            }
            
        if router_cfg['type'] == 'BasicTrajRouter':
            from .router import BasicTrajRouter
            self.router = BasicTrajRouter(
                input_dim=router_cfg.get('input_dim', 4),
                num_experts=num_experts
            )
        elif router_cfg['type'] == 'UniTrajAttentionRouter':
            from .router import UniTrajAttentionRouter
            self.router = UniTrajAttentionRouter(
                num_experts=num_experts,
                embed_dim=router_cfg.get('embed_dim', 128),
                num_heads=router_cfg.get('num_heads', 4),
                use_map_features=router_cfg.get('use_map_features', True)
            )
        else:
            raise ValueError(f"Unknown router type: {router_cfg['type']}")
        
        # ============ 3. 轨迹融合模块 ============
        if fusion_strategy == 'attention':
            self.fusion_module = TrajectoryAttentionFusion(
                num_experts=num_experts,
                traj_dim=2,
                hidden_dim=256
            )
        elif fusion_strategy == 'gating':
            self.fusion_module = TrajectoryGatingFusion(
                num_experts=num_experts,
                traj_dim=2,
                hidden_dim=256
            )
        else:  # weighted_average
            self.fusion_module = None
        
        # ============ 4. 输出标准化层 ============
        if normalize_outputs:
            self.output_normalizer = OutputNormalizer(
                traj_steps=30,
                traj_dim=2
            )
        else:
            self.output_normalizer = None
        
        # ============ 5. 负载均衡参数 ============
        self.lb_loss_weight = lb_loss_weight_initial
        self.lb_loss_weight_initial = lb_loss_weight_initial
        self.lb_loss_decay_rate = lb_loss_decay_rate
        self.auxiliary_loss_weight = auxiliary_loss_weight
        
        # ============ 6. 监控变量 ============
        self.last_routing_probs = None
        self.last_expert_outputs = None
        self.last_load_balance_loss = None
        self.expert_usage_counts = torch.zeros(num_experts)
        
    def _get_default_expert_configs(self) -> List[Dict]:
        """获取默认专家配置"""
        configs = []
        for name in self.expert_names:
            if name == 'autobot':
                cfg = {
                    'type': 'AutoBot',
                    'hidden_size': 256,
                    'num_layers': 6,
                    'num_heads': 8,
                    'num_modes': 6,
                    'future_steps': 30
                }
            elif name == 'mtr':
                cfg = {
                    'type': 'MTR', 
                    'hidden_dim': 512,
                    'num_intention_clusters': 64,
                    'num_future_frames': 30
                }
            elif name == 'smart':
                cfg = {
                    'type': 'SMART',
                    'feature_dim': 384,
                    'num_trajectories': 6,
                    'prediction_horizon': 30
                }
            elif name == 'wayformer':
                cfg = {
                    'type': 'Wayformer',
                    'd_model': 768,
                    'num_layers': 12,
                    'num_heads': 12,
                    'prediction_steps': 30
                }
            else:
                raise ValueError(f"Unknown expert: {name}")
            configs.append(cfg)
        return configs
    
    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            data_batch: UniTraj格式输入
            
        Returns:
            包含预测轨迹和辅助信息的字典
        """
        B = data_batch['agent']['position'].shape[0]
        device = data_batch['agent']['position'].device
        
        # ============ 1. 路由决策 ============
        routing_probs = self.router(data_batch)  # (B, num_experts)
        self.last_routing_probs = routing_probs
        
        # 温度缩放
        routing_probs = routing_probs / self.temperature
        routing_probs = F.softmax(routing_probs, dim=-1)
        
        # 打印路由信息（调试用）
        if self.training and np.random.random() < 0.01:  # 1%概率打印
            print(f"Routing probabilities: {routing_probs[0].detach().cpu().numpy()}")
        
        # ============ 2. 专家选择 ============
        if self.training:
            # 训练时：使用所有专家（软路由）
            weights = routing_probs
            indices = torch.arange(self.num_experts, device=device).repeat(B, 1)
        else:
            # 推理时：选择top-k专家（硬路由）
            weights, indices = torch.topk(routing_probs, k=min(self.k, self.num_experts), dim=-1)
            # 重新归一化权重
            weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # 更新专家使用统计
        for i in range(self.num_experts):
            self.expert_usage_counts[i] += (indices == i).sum().item()
        
        # ============ 3. 专家处理 ============
        expert_outputs = []
        expert_trajectories = []
        expert_scores = []
        
        for expert_idx in range(self.num_experts):
            # 找出分配给该专家的样本
            if self.training:
                # 软路由：所有样本都经过所有专家
                batch_indices = torch.arange(B, device=device)
                expert_weights = weights[:, expert_idx]
            else:
                # 硬路由：只处理被选中的样本
                mask = (indices == expert_idx).any(dim=-1)
                if not mask.any():
                    continue
                batch_indices = torch.where(mask)[0]
                # 获取对应权重
                weight_mask = (indices[mask] == expert_idx)
                expert_weights = torch.zeros(mask.sum(), device=device)
                expert_weights[weight_mask.any(dim=-1)] = weights[mask][weight_mask].mean(dim=-1)
            
            # 准备专家输入
            expert_batch = self._prepare_expert_batch(data_batch, batch_indices)
            
            # 专家前向传播
            with torch.cuda.amp.autocast(enabled=False):  # 某些专家可能不支持混合精度
                expert_output = self.experts[expert_idx](expert_batch)
            
            # 适配专家输出
            adapted_output = self.expert_adapters[expert_idx].adapt(expert_output)
            
            expert_outputs.append({
                'expert_idx': expert_idx,
                'batch_indices': batch_indices,
                'weights': expert_weights,
                'output': adapted_output
            })
            
            # 收集轨迹用于融合
            if self.training or (indices == expert_idx).any():
                expert_trajectories.append(adapted_output['best_mode'])  # (batch_size, T, 2)
                expert_scores.append(expert_weights)
        
        self.last_expert_outputs = expert_outputs
        
        # ============ 4. 轨迹融合 ============
        if self.fusion_strategy == 'weighted_average':
            # 加权平均融合
            final_trajectory = self._weighted_average_fusion(
                expert_outputs, B, device
            )
        elif self.fusion_strategy == 'attention':
            # 注意力融合
            final_trajectory = self.fusion_module(
                expert_trajectories, expert_scores, data_batch
            )
        elif self.fusion_strategy == 'gating':
            # 门控融合
            final_trajectory = self.fusion_module(
                expert_trajectories, expert_scores, routing_probs
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        # ============ 5. 输出标准化 ============
        if self.output_normalizer is not None:
            final_trajectory = self.output_normalizer(final_trajectory)
        
        # ============ 6. 计算辅助损失 ============
        auxiliary_losses = {}
        
        if self.load_balancing and self.training:
            # 负载均衡损失
            lb_loss = self._compute_load_balance_loss(routing_probs)
            self.last_load_balance_loss = lb_loss
            auxiliary_losses['load_balance'] = lb_loss * self.lb_loss_weight
            
            # 稀疏性损失（鼓励使用较少的专家）
            sparsity_loss = self._compute_sparsity_loss(routing_probs)
            auxiliary_losses['sparsity'] = sparsity_loss * 0.01
        
        # ============ 7. 构建输出 ============
        output = {
            'trajectory': final_trajectory,  # (B, T, 2)
            'routing_probs': routing_probs,  # (B, num_experts)
            'expert_weights': weights,       # (B, k)
            'expert_indices': indices,       # (B, k)
        }
        
        if auxiliary_losses:
            output['auxiliary_losses'] = auxiliary_losses
        
        # 如果是多模态预测，添加所有模态
        if len(expert_trajectories) > 1:
            all_trajectories = torch.stack([
                out['output']['trajectories'] for out in expert_outputs
                if 'trajectories' in out['output']
            ], dim=1)  # (B, num_experts, K, T, 2)
            output['all_trajectories'] = all_trajectories
        
        return output
    
    def _prepare_expert_batch(self, data_batch: Dict, indices: torch.Tensor) -> Dict:
        """为特定专家准备输入批次"""
        expert_batch = {}
        
        for key, value in data_batch.items():
            if isinstance(value, dict):
                expert_batch[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        expert_batch[key][sub_key] = sub_value[indices]
                    else:
                        expert_batch[key][sub_key] = sub_value
            elif isinstance(value, torch.Tensor):
                expert_batch[key] = value[indices]
            else:
                expert_batch[key] = value
                
        return expert_batch
    
    def _weighted_average_fusion(self, 
                                  expert_outputs: List[Dict],
                                  batch_size: int,
                                  device: torch.device) -> torch.Tensor:
        """加权平均融合策略"""
        # 初始化输出
        final_trajectory = torch.zeros((batch_size, 30, 2), device=device)
        total_weights = torch.zeros(batch_size, device=device)
        
        for expert_data in expert_outputs:
            indices = expert_data['batch_indices']
            weights = expert_data['weights']
            trajectory = expert_data['output']['best_mode']  # (batch_size, T, 2)
            
            # 累加加权轨迹
            if self.training:
                # 软路由：所有样本
                final_trajectory += trajectory * weights.unsqueeze(-1).unsqueeze(-1)
                total_weights += weights
            else:
                # 硬路由：部分样本
                final_trajectory[indices] += trajectory * weights.unsqueeze(-1).unsqueeze(-1)
                total_weights[indices] += weights
        
        # 归一化
        total_weights = total_weights.clamp(min=1e-8)
        final_trajectory = final_trajectory / total_weights.unsqueeze(-1).unsqueeze(-1)
        
        return final_trajectory
    
    def _compute_load_balance_loss(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """
        计算负载均衡损失
        鼓励均匀使用所有专家
        """
        # 计算每个专家的平均负载
        expert_loads = routing_probs.mean(dim=0)  # (num_experts,)
        
        # 理想负载（均匀分布）
        ideal_load = 1.0 / self.num_experts
        
        # L2损失
        lb_loss = ((expert_loads - ideal_load) ** 2).sum()
        
        return lb_loss
    
    def _compute_sparsity_loss(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """
        计算稀疏性损失
        鼓励每个样本只使用少数专家
        """
        # 使用熵作为稀疏性度量
        entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-8), dim=-1)
        
        # 我们希望低熵（高稀疏性）
        sparsity_loss = entropy.mean()
        
        return sparsity_loss
    
    def get_routing_probs(self) -> Optional[torch.Tensor]:
        """获取最后一次前向传播的路由概率"""
        return self.last_routing_probs
    
    def update_lb_loss_weight(self, epoch: int):
        """更新负载均衡损失权重（随训练衰减）"""
        self.lb_loss_weight = self.lb_loss_weight_initial * (self.lb_loss_decay_rate ** epoch)
        print(f"Load balance loss weight updated to {self.lb_loss_weight:.6f}")
    
    def get_expert_usage_stats(self) -> Dict[str, float]:
        """获取专家使用统计"""
        total_usage = self.expert_usage_counts.sum()
        if total_usage == 0:
            return {name: 0.0 for name in self.expert_names}
        
        usage_rates = self.expert_usage_counts / total_usage
        return {
            name: rate.item() 
            for name, rate in zip(self.expert_names, usage_rates)
        }
    
    def reset_usage_stats(self):
        """重置使用统计"""
        self.expert_usage_counts.zero_()


class TrajectoryAttentionFusion(nn.Module):
    """基于注意力的轨迹融合模块"""
    
    def __init__(self, num_experts: int, traj_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        
        self.num_experts = num_experts
        
        # 轨迹编码器
        self.traj_encoder = nn.LSTM(
            traj_dim, 
            hidden_dim // 2, 
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 输出解码器
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, traj_dim)
        )
    
    def forward(self, 
                trajectories: List[torch.Tensor],
                scores: List[torch.Tensor],
                context: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            trajectories: 专家轨迹列表
            scores: 专家权重列表
            context: 场景上下文
        Returns:
            融合后的轨迹
        """
        # 堆叠所有轨迹
        all_trajs = torch.stack(trajectories, dim=1)  # (B, num_experts, T, 2)
        B, E, T, D = all_trajs.shape
        
        # 编码每条轨迹
        trajs_flat = all_trajs.reshape(B * E, T, D)
        encoded, _ = self.traj_encoder(trajs_flat)  # (B*E, T, hidden_dim)
        encoded = encoded.reshape(B, E, T, -1)
        
        # 使用分数加权
        scores_tensor = torch.stack(scores, dim=1)  # (B, num_experts)
        weighted_encoded = encoded * scores_tensor.unsqueeze(-1).unsqueeze(-1)
        
        # 交叉注意力融合
        query = weighted_encoded.mean(dim=1)  # (B, T, hidden_dim)
        keys = values = encoded.reshape(B, E * T, -1)  # (B, E*T, hidden_dim)
        
        fused, _ = self.cross_attn(query, keys, values)  # (B, T, hidden_dim)
        
        # 解码为轨迹
        output = self.output_decoder(fused)  # (B, T, 2)
        
        return output


class TrajectoryGatingFusion(nn.Module):
    """基于门控的轨迹融合模块"""
    
    def __init__(self, num_experts: int, traj_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        
        self.num_experts = num_experts
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(num_experts + traj_dim * 30, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts * 30),
            nn.Sigmoid()
        )
    
    def forward(self,
                trajectories: List[torch.Tensor],
                scores: List[torch.Tensor],
                routing_probs: torch.Tensor) -> torch.Tensor:
        """
        门控融合
        """
        # 堆叠轨迹
        all_trajs = torch.stack(trajectories, dim=1)  # (B, num_experts, T, 2)
        B, E, T, D = all_trajs.shape
        
        # 计算门控值
        traj_features = all_trajs.reshape(B, -1)  # (B, E*T*D)
        gate_input = torch.cat([routing_probs, traj_features], dim=-1)
        gates = self.gate_network(gate_input)  # (B, E*T)
        gates = gates.reshape(B, E, T)
        
        # 应用门控
        gated_trajs = all_trajs * gates.unsqueeze(-1)  # (B, E, T, 2)
        
        # 加权求和
        scores_tensor = torch.stack(scores, dim=1).unsqueeze(-1).unsqueeze(-1)  # (B, E, 1, 1)
        output = (gated_trajs * scores_tensor).sum(dim=1)  # (B, T, 2)
        
        return output


class OutputNormalizer(nn.Module):
    """输出标准化模块"""
    
    def __init__(self, traj_steps: int = 30, traj_dim: int = 2):
        super().__init__()
        
        self.traj_steps = traj_steps
        self.traj_dim = traj_dim
        
        # 可学习的缩放和偏移参数
        self.scale = nn.Parameter(torch.ones(traj_steps, traj_dim))
        self.bias = nn.Parameter(torch.zeros(traj_steps, traj_dim))
    
    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        标准化轨迹输出
        """
        # 应用缩放和偏移
        normalized = trajectory * self.scale + self.bias
        
        # 确保轨迹连续性（可选）
        # 使用差分平滑
        if self.training:
            diff = normalized[:, 1:] - normalized[:, :-1]
            smoothness_loss = (diff ** 2).mean()
            # 这个损失可以添加到辅助损失中
        
        return normalized


# 注册为模型
@MODELS.register_module()
class UniTrajMoENetwork(BaseModule):
    """
    封装的MoE网络，提供完整的训练接口
    """
    
    def __init__(self,
                 num_experts: int = 4,
                 expert_names: List[str] = ['autobot', 'mtr', 'smart', 'wayformer'],
                 experts_cfg: Optional[List[Dict]] = None,
                 router_cfg: Dict = None,
                 moe_cfg: Dict = None,
                 init_cfg: Optional[Dict] = None):
        
        super().__init__(init_cfg)
        
        # 默认MoE配置
        if moe_cfg is None:
            moe_cfg = {
                'k': 2,
                'fusion_strategy': 'attention',
                'normalize_outputs': True,
                'load_balancing': True,
                'lb_loss_weight_initial': 0.01,
                'lb_loss_decay_rate': 0.95
            }
        
        # 创建MoE模型
        self.moe = UniTrajMixtureOfExperts(
            num_experts=num_experts,
            expert_names=expert_names,
            experts_cfg=experts_cfg,
            router_cfg=router_cfg,
            **moe_cfg
        )
        
        self.current_epoch = 0
    
    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        return self.moe(data_batch)
    
    def get_routing_probs(self) -> Optional[torch.Tensor]:
        """获取路由概率"""
        return self.moe.get_routing_probs()
    
    def update_epoch(self, epoch: int):
        """更新当前epoch"""
        self.current_epoch = epoch
        self.moe.update_lb_loss_weight(epoch)
    
    def get_expert_usage_stats(self) -> Dict[str, float]:
        """获取专家使用统计"""
        return self.moe.get_expert_usage_stats()
    
    def reset_stats(self):
        """重置统计信息"""
        self.moe.reset_usage_stats()