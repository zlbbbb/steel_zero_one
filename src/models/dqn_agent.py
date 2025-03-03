# -*- coding: utf-8 -*-
"""
深度Q学习智能体

为钢铁行业数据设计的深度强化学习智能体，
实现完整的DQN算法并集成经验回放缓冲区
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random
import time
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional, Union
# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath("."))
# 导入自定义模块
from src.models.dqn_model import DQNModel, DuelingDQNModel
from src.models.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")
class DQNAgent:
    """
    深度Q学习智能体
    
    实现完整的DQN算法，包括Double DQN, Dueling DQN和优先级经验回放等改进
    """
    
    def __init__(self,
                 state_dim: int = 6,              # 状态特征维度
                 seq_len: int = 96,               # 序列长度
                 action_dim: int = 10,            # 离散动作空间大小
                 hidden_dim: int = 128,           # 隐藏层维度
                 learning_rate: float = 0.001,    # 学习率
                 gamma: float = 0.99,             # 折扣因子
                 buffer_size: int = 10000,        # 经验缓冲区大小
                 batch_size: int = 64,            # 批量大小
                 target_update: int = 10,         # 目标网络更新频率
                 use_dueling: bool = True,        # 是否使用Dueling架构
                 use_double: bool = True,         # 是否使用Double DQN
                 use_per: bool = False,           # 是否使用优先级经验回放
                 epsilon_start: float = 1.0,      # 初始探索率
                 epsilon_end: float = 0.01,       # 最终探索率
                 epsilon_decay: float = 0.995,    # 探索率衰减
                 device: str = None):
        """
        初始化DQN智能体
        
        Args:
            state_dim: 状态特征维度
            seq_len: 序列长度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度
            learning_rate: 学习率
            gamma: 折扣因子
            buffer_size: 经验回放缓冲区大小
            batch_size: 批量大小
            target_update: 目标网络更新频率(单位：回合数)
            use_dueling: 是否使用Dueling架构
            use_double: 是否使用Double DQN
            use_per: 是否使用优先级经验回放
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay: 探索率衰减
            device: 计算设备(如'cuda:0', 'cpu')
        """
        # 设置计算设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
                      if device is None else torch.device(device)
                      
        print(f"使用计算设备: {self.device}")
        
        # 保存参数
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.use_dueling = use_dueling
        self.use_double = use_double
        self.use_per = use_per
        
        # 探索参数
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 创建Q网络
        if use_dueling:
            self.q_network = DuelingDQNModel(
                input_dim=state_dim,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                num_actions=action_dim
            ).to(self.device)
            
            self.target_network = DuelingDQNModel(
                input_dim=state_dim,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                num_actions=action_dim
            ).to(self.device)
        else:
            self.q_network = DQNModel(
                input_dim=state_dim,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                num_actions=action_dim
            ).to(self.device)
            
            self.target_network = DQNModel(
                input_dim=state_dim,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                num_actions=action_dim
            ).to(self.device)
        
        # 初始化目标网络权重与Q网络相同
        self.update_target_network(tau=1.0)  # 完全复制
        self.target_network.eval()  # 设置为评估模式
        
        # 创建优化器
        self.optimizer = Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 创建经验回放缓冲区
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # 训练统计
        self.train_steps = 0
        self.train_episodes = 0
        self.total_rewards = []
        self.losses = []
        self.eval_rewards = []
        self.epsilon_history = []
        self.prediction_errors = []
        
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        根据当前状态选择动作
        
        Args:
            state: 当前状态，形状为[seq_len, state_dim]或[batch_size, seq_len, state_dim]
            eval_mode: 是否为评估模式(不使用探索)
            
        Returns:
            选择的动作
        """
        # 评估模式下完全贪婪，训练模式下使用epsilon-greedy策略
        if not eval_mode and random.random() < self.epsilon:
            # 随机探索
            return random.randint(0, self.action_dim - 1)
        
        # 确保state是张量并包含批次维度
        if isinstance(state, np.ndarray):
            # 若没有批次维度，添加一个
            if state.ndim == 2:
                state = np.expand_dims(state, 0)
            state = torch.FloatTensor(state).to(self.device)
        
        # 获取Q值预测
        with torch.no_grad():
            q_values, _ = self.q_network(state)
            # 选择Q值最大的动作
            action = q_values.max(1)[1].item()
        
        return action
    
    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
        
    def store_transition(self, 
                         state: np.ndarray, 
                         action: int, 
                         reward: float, 
                         next_state: np.ndarray, 
                         done: bool, 
                         info: Dict = None):
        """
        存储经验到回放缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否为终止状态
            info: 额外信息
        """
        self.replay_buffer.push(state, action, reward, next_state, done, info)
    
    def train_step(self) -> Dict[str, float]:
        """
        执行一步训练
        
        Returns:
            训练统计信息
        """
        # 如果缓冲区样本不足，跳过训练
        if len(self.replay_buffer) < self.batch_size:
            return {"q_loss": 0.0, "pred_loss": 0.0, "total_loss": 0.0}
        
        # 更新训练步数
        self.train_steps += 1
        
        # 从经验回放缓冲区采样
        if self.use_per:
            # 使用优先级经验回放
            states, actions, rewards, next_states, dones, indices, weights, _ = self.replay_buffer.sample(self.batch_size)
            weights_tensor = torch.FloatTensor(weights).to(self.device)
        else:
            # 使用普通经验回放
            states, actions, rewards, next_states, dones, _ = self.replay_buffer.sample(self.batch_size)
            weights_tensor = torch.ones(self.batch_size).to(self.device)
        
        # 转换为张量
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        q_values, next_state_pred = self.q_network(state_batch)
        q_value = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # 计算预测下一个状态的损失 (辅助任务)
        next_state_target = torch.FloatTensor(states[:, -1, :]).to(self.device)  # 使用当前序列最后一个状态作为目标
        pred_loss = F.mse_loss(next_state_pred, next_state_target)
        
        # 计算目标Q值
        with torch.no_grad():
            if self.use_double:
                # Double DQN: 用Q网络选择动作，用目标网络评估
                next_q_values, _ = self.q_network(next_state_batch)
                next_actions = next_q_values.max(1)[1].unsqueeze(1)
                next_q_values_target, _ = self.target_network(next_state_batch)
                next_q_value = next_q_values_target.gather(1, next_actions).squeeze(1)
            else:
                # 标准DQN: 用目标网络选择和评估
                next_q_values_target, _ = self.target_network(next_state_batch)
                next_q_value = next_q_values_target.max(1)[0]
            
            # 计算TD目标: r + gamma * max_a' Q_target(s', a')
            td_target = reward_batch + (1 - done_batch) * self.gamma * next_q_value
        
        # 计算TD误差
        td_error = torch.abs(q_value - td_target).detach().cpu().numpy()
        
        # 计算Q值损失(带权重)
        q_loss = (weights_tensor * F.smooth_l1_loss(q_value, td_target, reduction='none')).mean()
        
        # 总损失 = Q值损失 + 0.1 * 预测损失
        total_loss = q_loss + 0.1 * pred_loss
        
        # 优化模型
        self.optimizer.zero_grad()
        total_loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # 更新优先级
        if self.use_per:
            self.replay_buffer.update_priorities(indices, td_error + 1e-6)  # 添加小值防止优先级为0
        
        # 返回训练统计
        return {
            "q_loss": q_loss.item(),
            "pred_loss": pred_loss.item(),
            "total_loss": total_loss.item(),
            "max_q": q_values.max().item(),
            "mean_q": q_values.mean().item(),
            "td_error": td_error.mean()
        }
    
    def update_target_network(self, tau: float = 0.005):
        """
        更新目标网络
        
        Args:
            tau: 软更新系数(0~1)，1表示完全复制，小值表示部分更新
        """
        # 软更新目标网络
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def train(self, 
              env,
              episodes: int = 1000,
              max_steps: int = 500,
              update_every: int = 4,
              eval_interval: int = 20,
              save_interval: int = 100,
              save_dir: str = './checkpoints',
              log_interval: int = 10,
              warmup_episodes: int = 10,
              eval_episodes: int = 5):
        """
        训练智能体
        
        Args:
            env: 训练环境
            episodes: 训练回合数
            max_steps: 每个回合最大步数
            update_every: 每隔几步更新网络
            eval_interval: 每隔几个回合评估一次
            save_interval: 每隔几个回合保存一次模型
            save_dir: 模型保存路径
            log_interval: 每隔几个回合记录一次日志
            warmup_episodes: 预热回合数(纯随机动作)
            eval_episodes: 评估时的回合数
        
        Returns:
            训练统计信息
        """
        # 创建保存目录
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 创建训练日志
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = save_path / f"train_log_{timestamp}.txt"
        
        # 创建结果记录
        results = {
            "episodes": [],
            "rewards": [],
            "losses": [],
            "epsilons": [],
            "eval_rewards": [],
            "train_time": [],
            "config": {
                "state_dim": self.state_dim,
                "seq_len": self.seq_len,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "buffer_size": self.buffer_size,
                "batch_size": self.batch_size,
                "target_update": self.target_update,
                "use_dueling": self.use_dueling,
                "use_double": self.use_double,
                "use_per": self.use_per,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "epsilon_decay": self.epsilon_decay,
                "device": str(self.device)
            }
        }
        
        # 记录开始时间
        start_time = time.time()
        
        # 初始化最佳奖励
        best_eval_reward = float('-inf')
        
        print(f"\n开始训练...\n{'='*50}")
        print(f"总回合数: {episodes}, 每回合最大步数: {max_steps}")
        print(f"探索率: {self.epsilon_start} -> {self.epsilon_end}, 衰减率: {self.epsilon_decay}")
        print(f"使用Dueling: {self.use_dueling}, 使用Double DQN: {self.use_double}, 使用优先级回放: {self.use_per}")
        print(f"{'='*50}\n")
        
        # 主训练循环
        for episode in range(1, episodes + 1):
            # 重置环境
            state, _ = env.reset()
            episode_reward = 0
            episode_loss = []
            episode_start_time = time.time()
            
            # 是否为预热阶段
            is_warmup = episode <= warmup_episodes
            
            # 单回合循环
            for step in range(1, max_steps + 1):
                # 选择动作
                if is_warmup:
                    # 预热阶段: 纯随机动作
                    action = env.action_space.sample()
                else:
                    # 训练阶段: epsilon-greedy策略
                    action = self.select_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # 存储经验
                self.store_transition(state, action, reward, next_state, done, info)
                
                # 更新智能体
                if not is_warmup and step % update_every == 0:
                    # 训练网络
                    train_info = self.train_step()
                    # 记录损失
                    if train_info["total_loss"] > 0:
                        episode_loss.append(train_info["total_loss"])
                
                # 更新累积奖励
                episode_reward += reward
                
                # 更新状态
                state = next_state
                
                # 如果终止，结束当前回合
                if done:
                    break
            
            # 更新目标网络
            if episode % self.target_update == 0:
                self.update_target_network(tau=1.0)  # 硬更新
            
            # 更新探索率
            if not is_warmup:
                self.update_epsilon()
            
            # 计算回合时长
            episode_time = time.time() - episode_start_time
            total_time = time.time() - start_time
            
            # 记录训练统计
            self.train_episodes += 1
            self.total_rewards.append(episode_reward)
            mean_loss = np.mean(episode_loss) if episode_loss else 0
            self.losses.append(mean_loss)
            
            # 更新结果记录
            results["episodes"].append(episode)
            results["rewards"].append(episode_reward)
            results["losses"].append(mean_loss)
            results["epsilons"].append(self.epsilon)
            results["train_time"].append(total_time)
            
            # 记录日志
            if episode % log_interval == 0:
                log_msg = (
                    f"回合: {episode}/{episodes} | "
                    f"奖励: {episode_reward:.2f} | "
                    f"损失: {mean_loss:.4f} | "
                    f"探索率: {self.epsilon:.4f} | "
                    f"回合时长: {episode_time:.2f}s | "
                    f"总时长: {total_time:.2f}s | "
                    f"缓冲区: {len(self.replay_buffer)}/{self.buffer_size}"
                )
                print(log_msg)
                
                # 保存到日志文件
                with open(log_file, 'a') as f:
                    f.write(log_msg + "\n")
            
            # 评估智能体
            if episode % eval_interval == 0:
                eval_reward = self.evaluate(env, episodes=eval_episodes)
                self.eval_rewards.append(eval_reward)
                results["eval_rewards"].append(eval_reward)
                
                eval_msg = f"评估 (回合 {episode}) | 平均奖励: {eval_reward:.2f}"
                print(eval_msg)
                
                # 保存到日志文件
                with open(log_file, 'a') as f:
                    f.write(eval_msg + "\n")
                
                # 如果是最佳模型，保存它
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    self.save_model(save_path / f"best_model_{timestamp}.pt")
                    print(f"保存最佳模型，奖励: {best_eval_reward:.2f}")
            
            # 定期保存模型
            if episode % save_interval == 0:
                self.save_model(save_path / f"model_episode_{episode}_{timestamp}.pt")
                
                # 保存训练结果
                with open(save_path / f"results_{timestamp}.json", 'w') as f:
                    json.dump(results, f)
        
        # 训练结束，保存最终模型
        final_model_path = save_path / f"final_model_{timestamp}.pt"
        self.save_model(final_model_path)
        
        # 保存最终训练结果
        with open(save_path / f"final_results_{timestamp}.json", 'w') as f:
            json.dump(results, f)
        
        total_train_time = time.time() - start_time
        print(f"\n训练完成！总时长: {total_train_time:.2f}秒")
        print(f"最终模型已保存至: {final_model_path}")
        
        return results
    
    def evaluate(self, env, episodes: int = 5) -> float:
        """
        评估智能体性能
        
        Args:
            env: 评估环境
            episodes: 评估回合数
            
        Returns:
            平均奖励
        """
        # 切换到评估模式
        self.q_network.eval()
        
        total_rewards = []
        
        for _ in range(episodes):
            state, _ = env.reset(options={'eval_mode': True})
            episode_reward = 0
            done = False
            
            while not done:
                # 选择动作(无探索)
                action = self.select_action(state, eval_mode=True)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 更新累积奖励
                episode_reward += reward
                
                # 更新状态
                state = next_state
            
            total_rewards.append(episode_reward)
        
        # 恢复训练模式
        self.q_network.train()
        
        # 计算平均奖励
        mean_reward = np.mean(total_rewards)
        return mean_reward
    
    def predict(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        预测动作和Q值
        
        Args:
            state: 当前状态
            
        Returns:
            选择的动作和所有动作的Q值
        """
        # 确保state是张量并包含批次维度
        if isinstance(state, np.ndarray):
            if state.ndim == 2:  # [seq_len, state_dim]
                state = np.expand_dims(state, 0)  # [1, seq_len, state_dim]
            state = torch.FloatTensor(state).to(self.device)
        
        # 切换到评估模式并预测
        self.q_network.eval()
        with torch.no_grad():
            q_values, _ = self.q_network(state)
            q_values_np = q_values.cpu().numpy()[0]  # 转为numpy并取第一个批次
            action = q_values.argmax(dim=1).item()
        
        # 恢复训练模式
        self.q_network.train()
        
        return action, q_values_np
    
    def save_model(self, path: str = None):
        """
        保存模型
        
        Args:
            path: 保存路径，如果为None则使用时间戳创建
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"./models/dqn_model_{timestamp}.pt"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型状态字典和训练参数
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'train_episodes': self.train_episodes,
            'epsilon': self.epsilon,
            'model_config': {
                'state_dim': self.state_dim,
                'seq_len': self.seq_len,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                'use_dueling': self.use_dueling,
                'use_double': self.use_double
            },
            'timestamp': datetime.now().isoformat()
        }, path)
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # 检查模型配置是否匹配
        model_config = checkpoint['model_config']
        config_match = (
            model_config['state_dim'] == self.state_dim and
            model_config['seq_len'] == self.seq_len and
            model_config['action_dim'] == self.action_dim and
            model_config['hidden_dim'] == self.hidden_dim and
            model_config['use_dueling'] == self.use_dueling
        )
        
        if not config_match:
            print("警告：加载的模型配置与当前配置不匹配！")
            print(f"加载配置: {model_config}")
            print(f"当前配置: 状态维度={self.state_dim}, 序列长度={self.seq_len}, "
                  f"动作维度={self.action_dim}, 隐藏维度={self.hidden_dim}, "
                  f"使用Dueling={self.use_dueling}")
        
        # 加载模型权重
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载训练状态
        self.train_steps = checkpoint.get('train_steps', 0)
        self.train_episodes = checkpoint.get('train_episodes', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        
        print(f"模型已从 {path} 加载")
        print(f"训练步数: {self.train_steps}, 训练回合: {self.train_episodes}, 探索率: {self.epsilon:.4f}")
    
    def plot_training_history(self, save_path: str = None, show_plot: bool = True):
        """
        绘制训练历史
        
        Args:
            save_path: 保存图表的路径，如果为None则不保存
            show_plot: 是否显示图表
        """
        if not self.total_rewards or not self.losses:
            print("没有可用的训练历史数据")
            return
        
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        
        # 绘制奖励曲线
        axes[0].plot(self.total_rewards, label='每回合奖励')
        if self.eval_rewards:  # 如果有评估奖励
            # 创建评估奖励的x轴索引
            eval_indices = np.linspace(0, len(self.total_rewards) - 1, len(self.eval_rewards)).astype(int)
            axes[0].plot(eval_indices, self.eval_rewards, 'r--', label='评估奖励', linewidth=2)
        
        # 添加移动平均线
        if len(self.total_rewards) > 10:
            window_size = min(10, len(self.total_rewards) // 5)
            rewards_avg = np.convolve(self.total_rewards, 
                                      np.ones(window_size) / window_size, 
                                      mode='valid')
            axes[0].plot(np.arange(window_size-1, len(self.total_rewards)), 
                         rewards_avg, 'g-', label=f'移动平均 (窗口={window_size})')
        
        axes[0].set_title('训练奖励')
        axes[0].set_xlabel('回合')
        axes[0].set_ylabel('奖励')
        axes[0].legend()
        axes[0].grid(True)
        
        # 绘制损失曲线
        axes[1].plot(self.losses, label='损失')
        
        # 添加移动平均线
        if len(self.losses) > 10:
            window_size = min(10, len(self.losses) // 5)
            losses_avg = np.convolve(self.losses, 
                                     np.ones(window_size) / window_size, 
                                     mode='valid')
            axes[1].plot(np.arange(window_size-1, len(self.losses)), 
                         losses_avg, 'r-', label=f'移动平均 (窗口={window_size})')
        
        axes[1].set_title('训练损失')
        axes[1].set_xlabel('回合')
        axes[1].set_ylabel('损失')
        axes[1].set_yscale('log')  # 使用对数尺度以更好地显示损失变化
        axes[1].legend()
        axes[1].grid(True)
        
        # 绘制探索率曲线
        axes[2].plot(self.epsilon_history, label='探索率 (epsilon)')
        axes[2].set_title('探索率变化')
        axes[2].set_xlabel('回合')
        axes[2].set_ylabel('Epsilon')
        axes[2].legend()
        axes[2].grid(True)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图表已保存至: {save_path}")
        
        # 显示图表
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def predict_sequence(self, env, state=None, n_steps: int = 24):
        """
        从给定状态开始，预测未来n_steps步
        
        Args:
            env: 环境
            state: 起始状态，如果为None则重置环境获取
            n_steps: 预测步数
            
        Returns:
            预测序列和实际序列
        """
        self.q_network.eval()  # 评估模式
        
        # 如果没有提供状态，则重置环境
        if state is None:
            state, _ = env.reset(options={'eval_mode': True})
        
        predictions = []
        actual_values = []
        rewards = []
        
        current_state = state.copy()
        
        for _ in range(n_steps):
            # 选择动作
            action = self.select_action(current_state, eval_mode=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 保存信息
            predictions.append(info['predicted'])
            actual_values.append(info['actual'])
            rewards.append(reward)
            
            # 更新状态
            current_state = next_state
            
            if terminated or truncated:
                break
        
        self.q_network.train()  # 恢复训练模式
        
        # 转换为numpy数组
        predictions = np.array(predictions)
        actual_values = np.array(actual_values)
        rewards = np.array(rewards)
        
        return predictions, actual_values, rewards
    
    def plot_prediction(self, predictions, actual_values, feature_names=None, save_path=None, show_plot=True):
        """
        绘制预测结果
        
        Args:
            predictions: 预测序列
            actual_values: 实际序列
            feature_names: 特征名称
            save_path: 保存图表的路径
            show_plot: 是否显示图表
        """
        if len(predictions) == 0:
            print("没有预测数据")
            return
        
        n_steps, n_features = predictions.shape
        
        if feature_names is None:
            feature_names = [f"特征 {i+1}" for i in range(n_features)]
        
        # 创建图表
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 2*n_features))
        if n_features == 1:
            axes = [axes]
        
        for i in range(n_features):
            ax = axes[i]
            feature_name = feature_names[i] if i < len(feature_names) else f"特征 {i+1}"
            
            # 绘制预测和实际值
            ax.plot(predictions[:, i], 'r-', label='预测')
            ax.plot(actual_values[:, i], 'b--', label='实际')
            
            ax.set_title(feature_name)
            ax.set_xlabel('时间步')
            ax.set_ylabel('值')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测结果图表已保存至: {save_path}")
        
        # 显示图表
        if show_plot:
            plt.show()
        else:
            plt.close()


# 测试代码
if __name__ == "__main__":
    print("测试DQN智能体...")
    
    # 仅在导入时测试DQN智能体的基本功能，不实际训练
    state_dim = 6
    seq_len = 96
    action_dim = 10
    
    # 创建智能体
    agent = DQNAgent(
        state_dim=state_dim,
        seq_len=seq_len,
        action_dim=action_dim,
        hidden_dim=64,
        use_dueling=True,
        use_double=True
    )
    
    # 创建随机状态并测试动作选择
    random_state = np.random.randn(seq_len, state_dim).astype(np.float32)
    action = agent.select_action(random_state, eval_mode=True)
    
    print(f"随机状态形状: {random_state.shape}")
    print(f"选择的动作: {action}")
    
    # 测试存储经验
    random_next_state = np.random.randn(seq_len, state_dim).astype(np.float32)
    random_reward = np.random.rand()
    agent.store_transition(random_state, action, random_reward, random_next_state, False)
    
    print(f"缓冲区大小: {len(agent.replay_buffer)}")
    
    # 测试模型保存和加载
    tmp_path = "./tmp_model.pt"
    agent.save_model(tmp_path)
    print(f"模型已保存到: {tmp_path}")
    
    agent.load_model(tmp_path)
    print("模型已加载")
    
    # 删除临时文件
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        print(f"临时模型文件已删除: {tmp_path}")