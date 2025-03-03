# -*- coding: utf-8 -*-
"""
深度Q网络模型

为钢铁行业数据设计的深度Q学习模型，处理6个特征的时序数据:
- Usage_kWh: 用电量
- Lagging_Current_Reactive.Power_kVarh: 滞后无功功率
- Leading_Current_Reactive_Power_kVarh: 超前无功功率
- CO2(tCO2): 二氧化碳排放量
- Lagging_Current_Power_Factor: 滞后功率因数
- Leading_Current_Power_Factor: 超前功率因数

每个样本间隔15分钟，共35040条数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Any, Optional


class DQNModel(nn.Module):
    """基础深度Q网络模型"""
    
    def __init__(self,
                 input_dim: int = 6,           # 输入特征数
                 seq_len: int = 96,            # 输入序列长度 (24h = 96 * 15min)
                 hidden_dim: int = 128,        # 隐藏层维度
                 num_layers: int = 2,          # LSTM层数
                 num_actions: int = 10,        # 动作空间大小
                 dropout: float = 0.2):        # Dropout比例
        """
        初始化DQN模型
        
        Args:
            input_dim: 输入特征维度，默认为6
            seq_len: 输入序列长度，默认为96（对应24小时，每15分钟一个点）
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            num_actions: 动作空间大小
            dropout: Dropout比例
        """
        super(DQNModel, self).__init__()
        
        # 保存参数
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_actions = num_actions
        
        # 特征提取层 - 使用1D卷积处理时序数据的局部特征
        self.feature_extractor = self._build_feature_extractor(dropout)
        
        # 序列建模层 - 使用LSTM处理时间依赖关系
        self.sequence_model = self._build_sequence_model(dropout)
        
        # 注意力层 - 关注重要的时间步
        self.attention = self._build_attention(dropout)
        
        # Q值输出层 - 映射到动作空间
        self.q_head = self._build_q_head(dropout)
        
        # 预测头 - 用于辅助训练，预测下一时刻的状态
        self.prediction_head = self._build_prediction_head(dropout)
        
        # 初始化权重
        self._initialize_weights()
    
    def _build_feature_extractor(self, dropout: float) -> nn.Module:
        """
        构建特征提取层
        
        Args:
            dropout: Dropout比例
            
        Returns:
            特征提取网络
        """
        return nn.Sequential(
            # 第一层卷积，捕获短期模式
            nn.Conv1d(self.input_dim, self.hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 第二层卷积，增加感受野
            nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def _build_sequence_model(self, dropout: float) -> nn.Module:
        """
        构建序列模型层
        
        Args:
            dropout: Dropout比例
            
        Returns:
            序列建模网络
        """
        return nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0,
            bidirectional=False
        )
    
    def _build_attention(self, dropout: float) -> nn.Module:
        """
        构建注意力层
        
        Args:
            dropout: Dropout比例
            
        Returns:
            注意力机制
        """
        return nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=dropout
        )
    
    def _build_q_head(self, dropout: float) -> nn.Module:
        """
        构建Q值输出头
        
        Args:
            dropout: Dropout比例
            
        Returns:
            Q值输出网络
        """
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, self.num_actions)
        )
    
    def _build_prediction_head(self, dropout: float) -> nn.Module:
        """
        构建预测头
        
        Args:
            dropout: Dropout比例
            
        Returns:
            预测下一状态的网络
        """
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.input_dim)
        )
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM权重正交初始化
                    nn.init.orthogonal_(param)
                elif 'conv' in name or 'linear' in name:
                    # 卷积和线性层使用Xavier初始化
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # 偏置初始化为0
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
            
        Returns:
            q_values: Q值，形状为 [batch_size, num_actions]
            next_state_pred: 下一状态预测，形状为 [batch_size, input_dim]
        """
        batch_size = x.size(0)
        
        # 特征提取 - 调整维度以适应卷积层 [batch, input_dim, seq_len]
        x_conv = x.transpose(1, 2)
        features = self.feature_extractor(x_conv)
        
        # 调整回序列形式 [batch, seq_len, hidden_dim]
        features = features.transpose(1, 2)
        
        # 序列建模
        lstm_out, _ = self.sequence_model(features)
        
        # 注意力机制
        # 调整维度以适应注意力层 [seq_len, batch, hidden_dim]
        attention_input = lstm_out.transpose(0, 1)
        attn_output, _ = self.attention(attention_input, attention_input, attention_input)
        
        # 调整回原始维度 [batch, seq_len, hidden_dim]
        attn_output = attn_output.transpose(0, 1)
        
        # 取最后一个时间步的输出
        final_hidden = attn_output[:, -1, :]
        
        # 计算Q值
        q_values = self.q_head(final_hidden)
        
        # 预测下一状态
        next_state_pred = self.prediction_head(final_hidden)
        
        return q_values, next_state_pred


class DuelingDQNModel(nn.Module):
    """
    双重网络架构的DQN模型
    
    将Q值分解为状态价值V(s)和优势函数A(s,a)
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """
    
    def __init__(self,
                 input_dim: int = 6,
                 seq_len: int = 96,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_actions: int = 10,
                 dropout: float = 0.2):
        """
        初始化Dueling DQN模型
        
        Args:
            input_dim: 输入特征维度
            seq_len: 输入序列长度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            num_actions: 动作空间大小
            dropout: Dropout比例
        """
        super(DuelingDQNModel, self).__init__()
        
        # 保存参数
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_actions = num_actions
        
        # 共享的特征提取层
        self.feature_extractor = nn.Sequential(
            # 第一层卷积
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 第二层卷积
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 序列建模层
        self.sequence_model = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout
        )
        
        # 特征处理层
        self.feature_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 状态价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 输出单一状态值
        )
        
        # 优势函数流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)  # 输出每个动作的优势
        )
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'conv' in name or 'linear' in name:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
            
        Returns:
            q_values: Q值，形状为 [batch_size, num_actions]
            next_state_pred: 下一状态预测，形状为 [batch_size, input_dim]
        """
        batch_size = x.size(0)
        
        # 特征提取
        x_conv = x.transpose(1, 2)
        features = self.feature_extractor(x_conv)
        features = features.transpose(1, 2)
        
        # 序列建模
        lstm_out, _ = self.sequence_model(features)
        
        # 注意力机制
        attention_input = lstm_out.transpose(0, 1)
        attn_output, _ = self.attention(attention_input, attention_input, attention_input)
        attn_output = attn_output.transpose(0, 1)
        
        # 提取最后一个时间步
        final_hidden = attn_output[:, -1, :]
        
        # 特征处理
        features = self.feature_layer(final_hidden)
        
        # 状态价值
        value = self.value_stream(features)
        
        # 优势函数
        advantage = self.advantage_stream(features)
        
        # 计算Q值: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        # 预测下一状态
        next_state_pred = self.prediction_head(final_hidden)
        
        return q_values, next_state_pred


# 测试代码
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建测试数据
    batch_size = 8
    seq_len = 96  # 24小时，每15分钟一个点
    input_dim = 6  # 6个特征
    num_actions = 10
    
    # 随机生成输入数据模拟时序特征
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print("=== 测试基础DQN模型 ===")
    model = DQNModel(input_dim=input_dim, seq_len=seq_len, num_actions=num_actions)
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 前向传播测试
    q_values, next_state = model(x)
    print(f"Q值形状: {q_values.shape}")
    print(f"下一状态预测形状: {next_state.shape}")
    print(f"Q值范围: [{q_values.min().item():.4f}, {q_values.max().item():.4f}]")
    print(f"Q值平均: {q_values.mean().item():.4f}")
    
    print("\n=== 测试Dueling DQN模型 ===")
    dueling_model = DuelingDQNModel(input_dim=input_dim, seq_len=seq_len, num_actions=num_actions)
    print(f"模型参数总数: {sum(p.numel() for p in dueling_model.parameters() if p.requires_grad)}")
    
    # 前向传播测试
    dueling_q_values, dueling_next_state = dueling_model(x)
    print(f"Q值形状: {dueling_q_values.shape}")
    print(f"下一状态预测形状: {dueling_next_state.shape}")
    print(f"Q值范围: [{dueling_q_values.min().item():.4f}, {dueling_q_values.max().item():.4f}]")
    print(f"Q值平均: {dueling_q_values.mean().item():.4f}")
    
    print("\n=== 梯度测试 ===")
    # 创建一个简单的损失
    loss = q_values.mean()
    loss.backward()
    print("DQN 模型梯度计算成功")
    
    # 创建另一个损失测试Dueling DQN
    loss = dueling_q_values.mean()
    loss.backward()
    print("Dueling DQN 模型梯度计算成功")
    
    print("\n=== 批量输入测试 ===")
    # 测试不同批量大小
    batch_sizes = [1, 4, 16, 32]
    for bs in batch_sizes:
        x = torch.randn(bs, seq_len, input_dim)
        q, ns = model(x)
        print(f"批量大小 {bs}: Q值形状 {q.shape}, 下一状态形状 {ns.shape}")
    
    print("\n=== 序列长度测试 ===")
    # 测试不同序列长度
    seq_lengths = [24, 48, 96, 192]  # 6小时, 12小时, 24小时, 48小时
    for sl in seq_lengths:
        x = torch.randn(batch_size, sl, input_dim)
        try:
            model_sl = DQNModel(input_dim=input_dim, seq_len=sl, num_actions=num_actions)
            q, ns = model_sl(x)
            print(f"序列长度 {sl} (对应 {sl/4} 小时): 成功")
        except Exception as e:
            print(f"序列长度 {sl} (对应 {sl/4} 小时): 失败 - {e}")
    
    print("\n=== 功能测试完成 ===")