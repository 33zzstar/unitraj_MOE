import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np
# from unitraj.models.builder import MODELS, build_model
# from mmcv.runner import BaseModule


class ExpertAdapter(nn.Module):
    """
    专家输出适配器基类
    用于统一不同专家模型的输出格式
    """
    def __init__(self, expert_name: str, output_config: Dict):
        super().__init__()
        self.expert_name = expert_name
        self.output_config = output_config
        
    @abstractmethod
    def adapt(self, expert_output: Any) -> Dict[str, torch.Tensor]:
        """将专家输出适配为统一格式"""
        pass

class AutoBotAdapter(ExpertAdapter):
    """AutoBot模型适配器"""
    def __init__(self, hidden_dim: int = 256, output_steps: int = 30):
        super().__init__("autobot", {"hidden_dim": hidden_dim})
        self.output_steps = output_steps
        # AutoBot输出: (B, num_modes, T, 2)
        # 需要额外的模态选择和概率计算
        self.mode_selector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6个模态
        )
        
    def adapt(self, expert_output: Any) -> Dict[str, torch.Tensor]:
        if isinstance(expert_output, dict):
            trajectories = expert_output['trajectories']  # (B, K, T, 2)
            scores = expert_output.get('scores', None)
        else:
            trajectories = expert_output
            B, K, T, _ = trajectories.shape
            scores = torch.ones(B, K) / K  # 均匀分布
            
        return {
            'trajectories': trajectories,  # (B, K, T, 2)
            'scores': scores,              # (B, K)
            'best_mode': trajectories[:, 0]  # (B, T, 2) 最佳模态
        }

class MTRAdapter(ExpertAdapter):
    """MTR模型适配器"""
    def __init__(self, hidden_dim: int = 512, num_intentions: int = 64):
        super().__init__("mtr", {"hidden_dim": hidden_dim})
        self.num_intentions = num_intentions
        # MTR输出投影层
        self.output_proj = nn.Linear(hidden_dim, 30 * 2)  # 30步 x 2维
        
    def adapt(self, expert_output: Any) -> Dict[str, torch.Tensor]:
        if isinstance(expert_output, dict):
            # MTR返回意图和轨迹
            intentions = expert_output['intentions']  # (B, num_intentions, hidden)
            trajectories = expert_output['trajectories']  # (B, K, T, 2)
            scores = expert_output['scores']  # (B, K)
        else:
            # 处理原始输出
            B = expert_output.shape[0]
            trajectories = expert_output.reshape(B, -1, 30, 2)
            K = trajectories.shape[1]
            scores = torch.ones(B, K) / K
            
        return {
            'trajectories': trajectories,
            'scores': scores,
            'best_mode': trajectories[torch.arange(B), scores.argmax(dim=1)]
        }

class SMARTAdapter(ExpertAdapter):
    """SMART模型适配器"""
    def __init__(self, feature_dim: int = 384):
        super().__init__("smart", {"feature_dim": feature_dim})
        self.traj_decoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 30 * 2)
        )
        
    def adapt(self, expert_output: Any) -> Dict[str, torch.Tensor]:
        if isinstance(expert_output, tuple):
            features, trajectories = expert_output
            B, K, T, _ = trajectories.shape
            scores = torch.ones(B, K) / K
        elif isinstance(expert_output, dict):
            trajectories = expert_output['trajectories']
            scores = expert_output.get('scores', torch.ones(trajectories.shape[0], trajectories.shape[1]) / trajectories.shape[1])
        else:
            trajectories = expert_output
            B, K, T, _ = trajectories.shape
            scores = torch.ones(B, K) / K
            
        return {
            'trajectories': trajectories,
            'scores': scores,
            'best_mode': trajectories[:, 0]
        }

class WayformerAdapter(ExpertAdapter):
    """Wayformer模型适配器"""
    def __init__(self, d_model: int = 768):
        super().__init__("wayformer", {"d_model": d_model})
        self.output_head = nn.Linear(d_model, 30 * 2)
        
    def adapt(self, expert_output: Any) -> Dict[str, torch.Tensor]:
        if isinstance(expert_output, dict):
            waypoints = expert_output['waypoints']  # (B, num_agents, T, 2)
            # 聚焦于主要agent
            trajectories = waypoints[:, 0:1, :, :]  # (B, 1, T, 2)
            scores = torch.ones(waypoints.shape[0], 1)
        else:
            B = expert_output.shape[0]
            trajectories = expert_output.reshape(B, -1, 30, 2)
            scores = torch.ones(B, trajectories.shape[1]) / trajectories.shape[1]
            
        return {
            'trajectories': trajectories,
            'scores': scores,
            'best_mode': trajectories[:, 0]
        }
