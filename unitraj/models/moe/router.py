import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

class UniTrajAttentionRouter(nn.Module):
    """
    专为轨迹预测设计的注意力路由器
    
    主要改进：
    1. 将CNN替换为时序特征提取器（1D卷积/LSTM）
    2. 增加空间-时间交互模块
    3. 保留多头注意力机制但适配轨迹特征
    """
    
    def __init__(self, 
                 num_experts: int = 4,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 history_steps: int = 20,
                 num_agents: int = 128,
                 spatial_dim: int = 2,
                 use_map_features: bool = True):
        super().__init__()
        
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_map_features = use_map_features
        
        # ============ 1. 时序特征提取模块（替代原始CNN） ============
        # 使用1D卷积处理轨迹时序信息，类似于原始的2D卷积处理图像
        
        # 第一层：提取局部时序模式
        self.temporal_conv1 = nn.Sequential(
            nn.Conv1d(spatial_dim * num_agents, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 第二层：提取中期时序依赖
        self.temporal_conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 第三层：提取高级时序特征
        self.temporal_conv3 = nn.Sequential(
            nn.Conv1d(128, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.PReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局池化，获得固定长度输出
        )
        
        # ============ 2. 空间关系编码模块 ============
        # 处理agent之间的空间交互关系
        self.spatial_encoder = SpatialRelationEncoder(
            input_dim=spatial_dim,
            hidden_dim=64,
            output_dim=embed_dim
        )
        
        # ============ 3. 地图特征编码（可选） ============
        if use_map_features:
            self.map_encoder = MapFeatureEncoder(
                polyline_dim=7,  # x, y, z, dir_x, dir_y, dir_z, type
                hidden_dim=64,
                output_dim=embed_dim
            )
        
        # ============ 4. 特征融合层 ============
        fusion_input_dim = embed_dim * 2  # temporal + spatial
        if use_map_features:
            fusion_input_dim += embed_dim  # + map
            
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # ============ 5. 自注意力模块（保留原设计） ============
        self.self_attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=0.1,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        # ============ 6. 专家token（新增） ============
        # 为每个专家创建可学习的embedding，用于注意力计算
        self.expert_tokens = nn.Parameter(torch.randn(num_experts, embed_dim))
        nn.init.xavier_uniform_(self.expert_tokens)
        
        # ============ 7. 输出MLP（改进版） ============
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.PReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(embed_dim // 2, num_experts)
        )
        
        # Temperature参数用于控制softmax的锐度
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, data_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data_batch: UniTraj格式输入
                - agent['position']: (B, T, N, 2)
                - agent['velocity']: (B, T, N, 2) 
                - map['polylines']: (B, P, L, 7) if use_map_features
        
        Returns:
            routing_probs: (B, num_experts) 专家选择概率
        """
        
        # 提取输入
        agent_pos = data_batch['agent']['position']  # (B, T, N, 2)
        B, T, N, D = agent_pos.shape
        
        # ============ 1. 时序特征提取 ============
        # 重塑为1D卷积格式: (B, C, T)
        temporal_input = agent_pos.reshape(B, T, N * D).transpose(1, 2)  # (B, N*D, T)
        
        # 通过时序卷积层（类比原始的2D卷积）
        temp_feat1 = self.temporal_conv1(temporal_input)  # (B, 64, T/4)
        temp_feat2 = self.temporal_conv2(temp_feat1)      # (B, 128, T/8)
        temp_feat3 = self.temporal_conv3(temp_feat2)      # (B, embed_dim, 1)
        
        temporal_features = temp_feat3.squeeze(-1)  # (B, embed_dim)
        
        # ============ 2. 空间关系特征 ============
        # 使用最后时刻的位置计算agent间的空间关系
        last_positions = agent_pos[:, -1, :, :]  # (B, N, 2)
        spatial_features = self.spatial_encoder(last_positions)  # (B, embed_dim)
        
        # ============ 3. 地图特征（可选） ============
        if self.use_map_features and 'map' in data_batch:
            map_polylines = data_batch['map'].get('polylines', None)
            if map_polylines is not None:
                map_features = self.map_encoder(map_polylines)  # (B, embed_dim)
            else:
                map_features = torch.zeros(B, self.embed_dim, device=agent_pos.device)
        else:
            map_features = None
        
        # ============ 4. 特征融合 ============
        if map_features is not None:
            fused_features = torch.cat([
                temporal_features, 
                spatial_features, 
                map_features
            ], dim=-1)
        else:
            fused_features = torch.cat([
                temporal_features, 
                spatial_features
            ], dim=-1)
        
        fused_features = self.feature_fusion(fused_features)  # (B, embed_dim)
        
        # ============ 5. 自注意力增强 ============
        # 将场景特征与专家token进行交互
        # 扩展专家token到batch维度
        expert_tokens_expanded = self.expert_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, num_experts, embed_dim)
        
        # 将场景特征作为query，专家token作为key和value
        scene_query = fused_features.unsqueeze(1)  # (B, 1, embed_dim)
        
        # 组合query和keys
        combined_seq = torch.cat([scene_query, expert_tokens_expanded], dim=1)  # (B, 1+num_experts, embed_dim)
        
        # 自注意力
        attn_output, attn_weights = self.self_attn(
            combined_seq, 
            combined_seq, 
            combined_seq
        )
        
        # 残差连接和层归一化
        attn_output = self.attn_norm(combined_seq + attn_output)
        
        # 提取增强后的场景特征
        enhanced_scene_feat = attn_output[:, 0, :]  # (B, embed_dim)
        
        # ============ 6. 计算路由概率 ============
        # 通过MLP计算logits
        routing_logits = self.output_mlp(enhanced_scene_feat)  # (B, num_experts)
        
        # 应用温度缩放
        routing_logits = routing_logits / self.temperature
        
        # Softmax获得概率
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        return routing_probs


class SpatialRelationEncoder(nn.Module):
    """空间关系编码器"""
    
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        
        # 相对位置编码
        self.relative_pos_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 图注意力层
        self.gat = GraphAttentionLayer(hidden_dim, hidden_dim)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (B, N, 2) agent位置
        Returns:
            spatial_features: (B, output_dim)
        """
        B, N, D = positions.shape
        
        # 计算相对位置
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, N, N, 2)
        distances = torch.norm(pos_diff, dim=-1)  # (B, N, N)
        
        # 构建邻接矩阵（距离阈值）
        adj_matrix = (distances < 50.0).float()  # 50米内的agent视为邻居
        
        # 编码相对位置
        rel_features = []
        for i in range(N):
            # 获取agent i与其他所有agent的相对位置
            rel_pos = torch.cat([
                positions[:, i:i+1, :].expand(-1, N, -1),  # agent i的位置
                positions  # 所有agent的位置
            ], dim=-1)  # (B, N, 4)
            
            rel_feat = self.relative_pos_encoder(rel_pos)  # (B, N, hidden_dim)
            rel_features.append(rel_feat[:, i, :])  # 取agent i的特征
        
        node_features = torch.stack(rel_features, dim=1)  # (B, N, hidden_dim)
        
        # 图注意力聚合
        gat_output = self.gat(node_features, adj_matrix)  # (B, N, hidden_dim)
        
        # 全局池化
        pooled_features = gat_output.mean(dim=1)  # (B, hidden_dim)
        
        # 输出投影
        output = self.output_proj(pooled_features)  # (B, output_dim)
        
        return output


class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, N, in_features) 节点特征
            adj: (B, N, N) 邻接矩阵
        Returns:
            output: (B, N, out_features)
        """
        B, N, _ = h.shape
        
        # 线性变换
        Wh = self.W(h)  # (B, N, out_features)
        
        # 计算注意力系数
        a_input = self._prepare_attention_input(Wh)  # (B, N, N, 2*out_features)
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(-1))  # (B, N, N)
        
        # 掩码处理
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        
        # 应用注意力
        output = torch.bmm(attention, Wh)  # (B, N, out_features)
        
        return output
    
    def _prepare_attention_input(self, Wh):
        B, N, E = Wh.shape
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1).view(B, N, N, E)
        Wh_repeated_alternating = Wh.repeat(1, N, 1).view(B, N, N, E)
        
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=-1)
        
        return all_combinations_matrix


class MapFeatureEncoder(nn.Module):
    """地图特征编码器"""
    
    def __init__(self, polyline_dim: int = 7, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        
        # Polyline编码器
        self.polyline_encoder = nn.Sequential(
            nn.Linear(polyline_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Polyline聚合
        self.polyline_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, polylines: torch.Tensor) -> torch.Tensor:
        """
        Args:
            polylines: (B, P, L, 7) P个polyline，每个L个点
        Returns:
            map_features: (B, output_dim)
        """
        B, P, L, D = polylines.shape
        
        # 编码每个点
        polylines_flat = polylines.reshape(B * P * L, D)
        point_features = self.polyline_encoder(polylines_flat)  # (B*P*L, hidden_dim)
        point_features = point_features.reshape(B, P, L, -1)
        
        # 聚合每条polyline
        polyline_features = point_features.mean(dim=2)  # (B, P, hidden_dim)
        
        # 聚合所有polyline
        map_features = polyline_features.mean(dim=1)  # (B, hidden_dim)
        
        # 最终投影
        output = self.polyline_aggregator(map_features)  # (B, output_dim)
        
        return output