# -*- coding: utf-8 -*-
"""
经验回放缓冲区

为深度Q学习提供经验存储和采样功能，提高训练效率和稳定性
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, List, Dict, Any, Optional


class ReplayBuffer:
    """
    标准经验回放缓冲区
    
    存储和采样(state, action, reward, next_state, done)五元组
    """
    
    def __init__(self, capacity: int = 10000):
        """
        初始化缓冲区
        
        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, 
             state: np.ndarray, 
             action: int, 
             reward: float, 
             next_state: np.ndarray, 
             done: bool,
             info: Dict = None):
        """
        存储一条经验到缓冲区
        
        Args:
            state: 当前状态，形状为[seq_len, input_dim]
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态，形状为[seq_len, input_dim]
            done: 是否终止状态
            info: 额外信息
        """
        # 确保输入是numpy数组，节省内存
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        
        # 如果没有提供额外信息，使用空字典
        if info is None:
            info = {}
        
        # 存储经验
        experience = (state, action, reward, next_state, done, info)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple:
        """
        从缓冲区随机采样一批经验
        
        Args:
            batch_size: 批量大小
            
        Returns:
            经验批次：(states, actions, rewards, next_states, dones, infos)
        """
        # 确保批量大小不超过缓冲区大小
        batch_size = min(batch_size, len(self.buffer))
        
        # 随机采样
        batch = random.sample(self.buffer, batch_size)
        
        # 解包样本
        states, actions, rewards, next_states, dones, infos = zip(*batch)
        
        # 转换为numpy数组
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, infos
    
    def __len__(self) -> int:
        """返回缓冲区当前大小"""
        return len(self.buffer)
    
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return len(self.buffer) == self.capacity
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """
    优先级经验回放缓冲区
    
    基于TD误差分配优先级，增强学习效率
    """
    
    def __init__(self, 
                 capacity: int = 10000, 
                 alpha: float = 0.6, 
                 beta: float = 0.4,
                 beta_increment: float = 0.001,
                 epsilon: float = 1e-6):
        """
        初始化优先级经验回放缓冲区
        
        Args:
            capacity: 缓冲区最大容量
            alpha: 优先级指数，控制优先级的程度(0=均匀采样，1=完全优先级采样)
            beta: 重要性采样权重系数，用于纠正优先级带来的偏差
            beta_increment: beta的增量，随训练进行逐渐从初始值增加到1
            epsilon: 防止优先级为0的小常数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # 存储结构
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.max_priority = 1.0  # 初始最大优先级
    
    def push(self, 
             state: np.ndarray, 
             action: int, 
             reward: float, 
             next_state: np.ndarray, 
             done: bool,
             info: Dict = None):
        """
        存储一条经验到缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止状态
            info: 额外信息
        """
        # 确保输入是numpy数组
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        
        if info is None:
            info = {}
        
        # 创建经验元组
        experience = (state, action, reward, next_state, done, info)
        
        # 如果缓冲区未满，直接添加
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            # 替换旧的经验
            self.buffer[self.position] = experience
        
        # 为新经验分配最大优先级，确保至少被采样一次
        self.priorities[self.position] = self.max_priority
        
        # 更新位置索引
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """
        基于优先级采样经验
        
        Args:
            batch_size: 批量大小
            
        Returns:
            经验批次和采样信息：(states, actions, rewards, next_states, dones, indices, weights, infos)
        """
        # 确保批量大小不超过缓冲区实际大小
        batch_size = min(batch_size, self.size)
        
        # 只使用已填充部分的优先级
        priorities = self.priorities[:self.size]
        
        # 计算采样概率
        probabilities = priorities ** self.alpha
        probabilities = probabilities / np.sum(probabilities)
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # 计算重要性采样权重
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # 归一化权重
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 收集样本
        experiences = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones, infos = zip(*experiences)
        
        # 转换为numpy数组
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, indices, weights, infos
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """
        更新经验的优先级
        
        Args:
            indices: 经验的索引
            priorities: 新优先级值
        """
        for idx, priority in zip(indices, priorities):
            # 添加小常数防止优先级为0
            self.priorities[idx] = priority + self.epsilon
            
            # 更新最大优先级
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self) -> int:
        """返回缓冲区当前大小"""
        return self.size
    
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return self.size == self.capacity
    
    def clear(self):
        """清空缓冲区"""
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.max_priority = 1.0


# 测试代码
if __name__ == "__main__":
    print("=== 测试标准经验回放缓冲区 ===")
    
    # 创建缓冲区
    buffer = ReplayBuffer(capacity=100)
    
    # 生成测试数据
    seq_len = 96
    input_dim = 6
    
    # 添加一些经验
    for i in range(50):
        state = np.random.randn(seq_len, input_dim).astype(np.float32)
        action = np.random.randint(0, 10)
        reward = np.random.random()
        next_state = np.random.randn(seq_len, input_dim).astype(np.float32)
        done = np.random.random() > 0.8
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"缓冲区大小: {len(buffer)}")
    
    # 测试采样
    batch_size = 16
    states, actions, rewards, next_states, dones, infos = buffer.sample(batch_size)
    
    print(f"采样批次大小: {len(states)}")
    print(f"状态形状: {states.shape}")
    print(f"动作形状: {actions.shape}")
    print(f"奖励形状: {rewards.shape}")
    print(f"下一状态形状: {next_states.shape}")
    print(f"终止标志形状: {dones.shape}")
    
    # 测试超出容量
    print("\n测试容量限制:")
    for i in range(60):
        state = np.random.randn(seq_len, input_dim).astype(np.float32)
        action = np.random.randint(0, 10)
        reward = np.random.random()
        next_state = np.random.randn(seq_len, input_dim).astype(np.float32)
        done = np.random.random() > 0.8
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"添加110个样本后缓冲区大小: {len(buffer)}")  # 应该是100
    
    print("\n=== 测试优先级经验回放缓冲区 ===")
    
    # 创建优先级缓冲区
    per_buffer = PrioritizedReplayBuffer(capacity=100)
    
    # 添加一些经验
    for i in range(50):
        state = np.random.randn(seq_len, input_dim).astype(np.float32)
        action = np.random.randint(0, 10)
        reward = np.random.random()
        next_state = np.random.randn(seq_len, input_dim).astype(np.float32)
        done = np.random.random() > 0.8
        
        per_buffer.push(state, action, reward, next_state, done)
    
    print(f"缓冲区大小: {len(per_buffer)}")
    
    # 测试优先级采样
    batch_size = 16
    states, actions, rewards, next_states, dones, indices, weights, infos = per_buffer.sample(batch_size)
    
    print(f"采样批次大小: {len(states)}")
    print(f"状态形状: {states.shape}")
    print(f"动作形状: {actions.shape}")
    print(f"权重形状: {weights.shape}")
    print(f"索引: {indices[:5]}...")  # 只打印前5个
    
    # 测试优先级更新
    print("\n测试优先级更新:")
    new_priorities = np.random.rand(batch_size) * 10  # 新的随机优先级
    per_buffer.update_priorities(indices, new_priorities)
    
    # 采样验证优先级影响
    print("更新优先级后重新采样:")
    new_states, new_actions, new_rewards, new_next_states, new_dones, new_indices, new_weights, new_infos = per_buffer.sample(100)
    
    # 计算与原始索引的交集数量
    intersection = np.intersect1d(indices, new_indices)
    print(f"新旧索引交集数量: {len(intersection)}")
    print(f"β值已更新到: {per_buffer.beta:.4f}")
    
    # 测试清空
    buffer.clear()
    per_buffer.clear()
    print(f"\n清空后标准缓冲区大小: {len(buffer)}")
    print(f"清空后优先级缓冲区大小: {len(per_buffer)}")
    
    print("\n=== 测试完成 ===")