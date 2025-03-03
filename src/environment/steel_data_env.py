# -*- coding: utf-8 -*-
"""
钢铁行业数据强化学习环境

基于Gymnasium设计的环境，用于处理钢铁行业时序数据，
支持深度Q学习算法训练和预测
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Tuple, Any, Optional, Union
import gymnasium as gym
from gymnasium import spaces

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")
class SteelDataEnv(gym.Env):
    """
    钢铁行业数据环境
    
    处理6个特征的时序数据，支持强化学习训练：
    - Usage_kWh: 用电量
    - Lagging_Current_Reactive.Power_kVarh: 滞后无功功率
    - Leading_Current_Reactive_Power_kVarh: 超前无功功率
    - CO2(tCO2): 二氧化碳排放量
    - Lagging_Current_Power_Factor: 滞后功率因数
    - Leading_Current_Power_Factor: 超前功率因数
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self,
                 data_path: str = None,
                 lookback_window: int = 96,           # 观察窗口大小 (24h = 96 * 15min)
                 prediction_horizon: int = 4,          # 预测步长
                 test_ratio: float = 0.2,              # 测试集比例
                 reward_type: str = 'mse',             # 奖励函数类型
                 max_steps: int = 1000,                # 每个回合最大步数
                 discrete_actions: int = 10,           # 每个特征的离散动作数
                 render_mode: str = None):
        """
        初始化环境
        
        Args:
            data_path: 数据文件路径，如果为None则使用默认路径
            lookback_window: 观察窗口大小，即每个状态的时间步长
            prediction_horizon: 预测未来多少个时间步
            test_ratio: 测试集比例
            reward_type: 奖励函数类型，'mse'、'mae'、'rmse'等
            max_steps: 每个回合的最大步数
            discrete_actions: 每个特征的离散动作数
            render_mode: 渲染模式，'human'或'rgb_array'
        """
        super(SteelDataEnv, self).__init__()
        
        # 保存参数
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.test_ratio = test_ratio
        self.reward_type = reward_type
        self.max_steps = max_steps
        self.discrete_actions = discrete_actions
        self.render_mode = render_mode
        # 初始化渲染相关属性
        self.fig = None
        self.axes = None
        self.rewards_history = []
        self.errors_history = []
        self.predicted = None
        self.actual = None


        # 加载数据
        self._load_data(data_path)
        
        # 定义特征列
        self.feature_columns = [
            'Usage_kWh', 
            'Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh', 
            'CO2(tCO2)',
            'Lagging_Current_Power_Factor',
            'Leading_Current_Power_Factor'
        ]
        
        # 特征数量
        self.num_features = len(self.feature_columns)
        
        # 归一化数据
        self._normalize_data()
        
        # 划分训练集和测试集
        self._split_train_test()
        
        # 定义动作空间和观察空间
        # 动作空间：每个特征有discrete_actions个可能值
        # 例如，如果discrete_actions=10，则action的范围是0-999
        self.action_space = spaces.Discrete(self.discrete_actions ** self.num_features)
        
        # 观察空间：[lookback_window, num_features]的连续空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lookback_window, self.num_features),
            dtype=np.float32
        )
        
        # 初始化环境状态
        self.reset()
        
    
    def _load_data(self, data_path: str = None):
        """
        加载数据
        
        Args:
            data_path: 数据文件路径
        """
        if data_path is None:
            # 使用默认路径
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_path = os.path.join(base_dir, 'data', 'processed', 'steel_data_processed.csv')
        
        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        # 加载数据
        print(f"加载数据: {data_path}")
        self.df = pd.read_csv(data_path)
        
        # 将日期列转换为datetime类型
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            # 设置为索引
            self.df.set_index('date', inplace=True)
        
        print(f"数据形状: {self.df.shape}")
    
    def _normalize_data(self):
        """
        归一化数据
        
        对每个特征进行标准化，保存均值和标准差用于还原
        """
        # 检查要使用的特征是否在数据中
        for col in self.feature_columns:
            if col not in self.df.columns:
                raise ValueError(f"特征 {col} 不在数据中. 可用特征: {self.df.columns.tolist()}")
        
        # 只使用需要的特征
        self.data = self.df[self.feature_columns].values
        
        # 计算均值和标准差
        self.feature_means = np.mean(self.data, axis=0)
        self.feature_stds = np.std(self.data, axis=0)
        
        # 避免除以0
        self.feature_stds = np.where(self.feature_stds == 0, 1, self.feature_stds)
        
        # 标准化数据
        self.normalized_data = (self.data - self.feature_means) / self.feature_stds
        
        print("数据归一化完成")
    
    def _split_train_test(self):
        """
        将数据分为训练集和测试集
        """
        # 计算分割点
        split_idx = int(len(self.normalized_data) * (1 - self.test_ratio))
        
        # 划分数据
        self.train_data = self.normalized_data[:split_idx]
        self.test_data = self.normalized_data[split_idx:]
        
        # 设置最大索引
        self.max_train_idx = len(self.train_data) - self.lookback_window - self.prediction_horizon
        self.max_test_idx = len(self.test_data) - self.lookback_window - self.prediction_horizon
        
        print(f"训练集大小: {len(self.train_data)}, 测试集大小: {len(self.test_data)}")
    
    def _decode_action(self, action: int) -> np.ndarray:
        """
        将离散动作解码为连续值
        
        Args:
            action: 离散动作索引
            
        Returns:
            解码后的连续动作值
        """
        # 将单一整数动作解码为每个特征的动作
        feature_actions = []
        remaining_action = action
        
        for _ in range(self.num_features):
            feature_action = remaining_action % self.discrete_actions
            feature_actions.append(feature_action)
            remaining_action //= self.discrete_actions
        
        # 将离散动作转换为连续值 [-1, 1]
        continuous_actions = 2 * (np.array(feature_actions) / (self.discrete_actions - 1)) - 1
        
        return continuous_actions
    
    def _get_state(self, idx: int, data: np.ndarray = None) -> np.ndarray:
        """
        获取指定索引处的状态
        
        Args:
            idx: 数据索引
            data: 使用的数据，默认为训练数据
            
        Returns:
            状态数据，形状为 [lookback_window, num_features]
        """
        if data is None:
            data = self.train_data
        
        return data[idx:idx+self.lookback_window]
    
    def _get_reward(self, 
                    predicted: np.ndarray, 
                    actual: np.ndarray) -> float:
        """
        根据预测值和实际值计算奖励
        
        Args:
            predicted: 预测值，形状为 [num_features]
            actual: 实际值，形状为 [num_features]
            
        Returns:
            奖励值
        """
        if self.reward_type == 'mse':
            # 平均平方误差
            error = np.mean((predicted - actual) ** 2)
            # 转换为奖励：误差越小，奖励越大
            reward = np.exp(-error) - 1  # 范围：[-1, 0]
        elif self.reward_type == 'mae':
            # 平均绝对误差
            error = np.mean(np.abs(predicted - actual))
            reward = np.exp(-error) - 1
        elif self.reward_type == 'rmse':
            # 均方根误差
            error = np.sqrt(np.mean((predicted - actual) ** 2))
            reward = np.exp(-error) - 1
        else:
            raise ValueError(f"不支持的奖励类型: {self.reward_type}")
        
        # 保存误差用于记录
        self.current_error = error
        
        return float(reward)
    
    def reset(self, seed: Optional[int] = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 重置选项
            
        Returns:
            初始状态和信息
        """
        # 设置随机种子
        if seed is not None:
            np.random.seed(seed)
        
        # 解析选项
        if options is None:
            options = {}
        
        # 确定使用训练模式还是测试模式
        self.eval_mode = options.get('eval_mode', False)
        data = self.test_data if self.eval_mode else self.train_data
        max_idx = self.max_test_idx if self.eval_mode else self.max_train_idx
        
        # 随机选择起始位置
        self.current_idx = np.random.randint(0, max_idx) if max_idx > 0 else 0
        
        # 获取初始状态
        self.current_state = self._get_state(self.current_idx, data)
        
        # 重置步数和累积奖励
        self.steps = 0
        self.total_reward = 0.0
        self.total_error = 0.0
        
        # 清除渲染相关数据
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
        
        return self.current_state, {"eval_mode": self.eval_mode}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行动作并获取下一个状态
        
        Args:
            action: 离散动作索引
            
        Returns:
            (next_state, reward, terminated, truncated, info)
        """
        # 增加步数
        self.steps += 1
        
        # 解码动作为连续值
        continuous_action = self._decode_action(action)
        
        # 确定使用的数据
        data = self.test_data if self.eval_mode else self.train_data
        
        # 获取当前真实数据的下一个值
        next_actual = data[self.current_idx + self.lookback_window]
        
        # 使用动作作为预测调整因子
        # 这里我们将动作视为对当前值的调整
        current_values = self.current_state[-1]  # 最新的观测值
        predicted_values = current_values + continuous_action * 0.1  # 动作范围在[-1,1]，调整为[-0.1,0.1]
        
        # 保存预测和实际值用于渲染
        self.predicted = predicted_values
        self.actual = next_actual
        
        # 计算奖励
        reward = self._get_reward(predicted_values, next_actual)
        
        # 更新历史记录
        self.rewards_history.append(reward)
        self.errors_history.append(self.current_error)
        
        # 更新累积指标
        self.total_reward += reward
        self.total_error += self.current_error
        
        # 更新索引
        self.current_idx += 1
        
        # 获取下一个状态
        next_state = self._get_state(self.current_idx, data)
        self.current_state = next_state
        
        # 检查是否终止
        terminated = False  # 环境本身不会终止
        truncated = self.steps >= self.max_steps  # 达到最大步数时截断
        
        # 构建信息字典
        info = {
            "error": self.current_error,
            "total_error": self.total_error,
            "total_reward": self.total_reward,
            "avg_error": self.total_error / self.steps,
            "predicted": predicted_values,
            "actual": next_actual
        }
        
        return next_state, reward, terminated, truncated, info
    
    def render(self):
        """
        渲染环境
        
        Returns:
            如果render_mode是'rgb_array'，则返回RGB数组
            如果render_mode是'human'，则显示可视化结果
        """
        if self.render_mode is None:
            return
        
        if self.fig is None:
            # 创建图形
            self.fig, self.axes = plt.subplots(self.num_features + 1, 1, figsize=(10, 12))
            if not isinstance(self.axes, np.ndarray):
                self.axes = np.array([self.axes])
            plt.tight_layout(pad=3.0)
            self.fig.suptitle("钢铁行业数据强化学习环境", fontsize=16)
        
        # 确定使用的数据
        data = self.test_data if self.eval_mode else self.train_data
        
        # 获取历史数据 (当前窗口的数据)
        history_data = self.current_state
        
        # 反归一化数据以便显示实际值
        history_data_denorm = history_data * self.feature_stds + self.feature_means
        
        # 绘制每个特征
        for i in range(self.num_features):
            ax = self.axes[i] if i < len(self.axes) else self.axes[0]
            feature_name = self.feature_columns[i]
            
            # 清除当前轴
            ax.clear()
            
            # 绘制历史数据
            ax.plot(range(self.lookback_window), history_data_denorm[:, i], 
                    label='历史数据', color='blue')
            
            # 如果有预测和真实值的记录，也绘制它们
            if hasattr(self, 'predicted') and self.predicted is not None and \
            hasattr(self, 'actual') and self.actual is not None:
                # 获取最后一步的预测和实际值
                pred = self.predicted[i] * self.feature_stds[i] + self.feature_means[i]
                actual = self.actual[i] * self.feature_stds[i] + self.feature_means[i]
                
                # 绘制预测点和实际点
                ax.scatter(self.lookback_window, pred, color='red', label='预测')
                ax.scatter(self.lookback_window, actual, color='green', label='实际')
            
            # 设置标签
            ax.set_title(f'{feature_name}')
            ax.set_xlabel('时间步')
            ax.set_ylabel('值')
            ax.legend()
            ax.grid(True)
        
        # 绘制奖励曲线
        if self.num_features < len(self.axes):
            ax = self.axes[-1]
        else:
            # 如果没有足够的子图，创建新的
            if hasattr(self, 'reward_fig'):
                plt.close(self.reward_fig)
            self.reward_fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.clear()
        
        if hasattr(self, 'rewards_history') and len(self.rewards_history) > 0:
            steps_range = range(1, len(self.rewards_history) + 1)
            ax.plot(steps_range, self.rewards_history, label='奖励', color='purple')
        
        if hasattr(self, 'errors_history') and len(self.errors_history) > 0:
            steps_range = range(1, len(self.errors_history) + 1)
            ax.plot(steps_range, self.errors_history, label='误差', color='orange')
        
        ax.set_title('奖励和误差')
        ax.set_xlabel('步数')
        ax.set_ylabel('值')
        ax.legend()
        ax.grid(True)
        
        # 调整布局
        plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
        
        # 根据render_mode处理渲染结果
        if self.render_mode == 'human':
            plt.pause(0.1)  # 暂停一小段时间让图形更新
        
        elif self.render_mode == 'rgb_array':
            self.fig.canvas.draw()
            # 将图形转换为RGB数组
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img
    
    def close(self):
        """关闭环境"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None
    
    def get_normalized_data(self, original_data: np.ndarray) -> np.ndarray:
        """
        将原始数据归一化
        
        Args:
            original_data: 原始数据
            
        Returns:
            归一化后的数据
        """
        return (original_data - self.feature_means) / self.feature_stds
    
    def get_denormalized_data(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        将归一化数据转换回原始尺度
        
        Args:
            normalized_data: 归一化数据
            
        Returns:
            原始尺度的数据
        """
        return normalized_data * self.feature_stds + self.feature_means
    
    def predict_sequence(self, agent, start_idx: int, steps: int, eval_mode: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用智能体预测一段时间序列
        
        Args:
            agent: DQN智能体
            start_idx: 起始索引
            steps: 预测步数
            eval_mode: 是否使用评估模式
            
        Returns:
            预测序列和实际序列
        """
        # 使用适当的数据集
        data = self.test_data if eval_mode else self.train_data
        
        # 检查索引有效性
        max_idx = len(data) - self.lookback_window - steps
        if start_idx > max_idx:
            raise ValueError(f"起始索引过大，最大允许值为 {max_idx}")
        
        # 获取初始状态
        current_state = self._get_state(start_idx, data)
        
        # 存储预测和实际值
        predictions = np.zeros((steps, self.num_features))
        actuals = np.zeros((steps, self.num_features))
        
        # 逐步预测
        for i in range(steps):
            # 智能体选择动作
            action = agent.select_action(current_state, eval_mode=True)
            
            # 解码动作
            continuous_action = self._decode_action(action)
            
            # 获取实际值
            actual = data[start_idx + self.lookback_window + i]
            actuals[i] = actual
            
            # 使用动作作为预测调整
            current_values = current_state[-1]
            predicted = current_values + continuous_action * 0.1
            predictions[i] = predicted
            
            # 如果不是最后一步，更新状态
            if i < steps - 1:
                # 构建新的状态
                new_state = np.vstack([current_state[1:], actual.reshape(1, -1)])
                current_state = new_state
        
        return predictions, actuals
    
    def evaluate_agent(self, agent, num_episodes: int = 10, max_steps: int = 200) -> Dict[str, float]:
        """
        评估智能体性能
        
        Args:
            agent: DQN智能体
            num_episodes: 评估回合数
            max_steps: 每回合的最大步数
            
        Returns:
            评估指标
        """
        total_rewards = []
        total_errors = []
        
        for episode in range(num_episodes):
            # 重置环境
            state, _ = self.reset(options={'eval_mode': True})
            episode_reward = 0
            episode_error = 0
            
            for step in range(max_steps):
                # 选择动作
                action = agent.select_action(state, eval_mode=True)
                
                # 执行动作
                next_state, reward, terminated, truncated, info = self.step(action)
                
                # 累积奖励和误差
                episode_reward += reward
                episode_error += info['error']
                
                # 更新状态
                state = next_state
                
                if terminated or truncated:
                    break
            
            # 保存回合指标
            total_rewards.append(episode_reward)
            total_errors.append(episode_error / (step + 1))  # 平均误差
        
        # 计算评估指标
        metrics = {
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "mean_error": np.mean(total_errors),
            "std_error": np.std(total_errors),
            "episodes": num_episodes
        }
        
        return metrics


# 测试代码部分
if __name__ == "__main__":
    # 使用随机数据测试环境
    import tempfile
    
    # 创建临时CSV文件
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        # 生成测试数据
        np.random.seed(42)
        num_samples = 1000
        feature_cols = [
            'Usage_kWh', 
            'Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh', 
            'CO2(tCO2)',
            'Lagging_Current_Power_Factor',
            'Leading_Current_Power_Factor'
        ]
        
        # 创建日期索引
        dates = pd.date_range('2023-01-01', periods=num_samples, freq='15min')
        
        # 创建随机数据
        data = np.random.randn(num_samples, len(feature_cols))
        
        # 添加一些趋势和周期性
        t = np.arange(num_samples)
        trend = 0.01 * t
        daily_cycle = np.sin(2 * np.pi * t / 96)  # 96个15分钟点 = 24小时
        
        data[:, 0] += trend + 2 * daily_cycle  # Usage_kWh
        data[:, 3] += 0.8 * trend + 1.5 * daily_cycle  # CO2
        
        # 创建DataFrame
        df = pd.DataFrame(data, index=dates, columns=feature_cols)
        
        # 保存到临时文件
        df.to_csv(tmp.name)
        
        tmp_filename = tmp.name  # 保存文件名
    
    # 测试环境
    print("测试钢铁行业数据环境...")
    try:
        env = SteelDataEnv(
            data_path=tmp_filename,
            lookback_window=96,
            prediction_horizon=4,
            render_mode='human'
        )
        
        # 随机动作测试
        print("执行随机动作测试...")
        state, _ = env.reset()
        print(f"状态形状: {state.shape}")
        
        total_reward = 0
        for step in range(100):
            # 随机选择动作
            action = env.action_space.sample()
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            if step % 10 == 0:
                print(f"步骤 {step}, 奖励: {reward:.4f}, 误差: {info['error']:.4f}")
                env.render()
            
            if terminated or truncated:
                break
        
        print(f"测试完成。总奖励: {total_reward:.4f}")
    
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    
    finally:
        # 清理资源
        if 'env' in locals():
            env.close()
        plt.close('all')  # 关闭所有matplotlib窗口
        
        # 尝试删除临时文件
        try:
            os.unlink(tmp_filename)
            print(f"临时文件 {tmp_filename} 已删除")
        except Exception as e:
            print(f"无法删除临时文件 {tmp_filename}: {e}")
            print("可以稍后手动删除或忽略此文件")