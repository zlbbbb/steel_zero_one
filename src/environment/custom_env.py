import gymnasium as gym
import numpy as np
from gymnasium import spaces

class CustomEnv(gym.Env):
    """
    自定义环境示例
    以此为基础进行不断拓展
    """
    
    def __init__(self):
        super(CustomEnv, self).__init__()
        
        # 定义动作空间
        self.action_space = spaces.Discrete(4)  # 4个离散动作

        
        # 定义观察空间
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )  # 图像观察
        
        self.state = None
        self.steps = 0
        
    def reset(self):
        """重置环境"""
        self.state = np.zeros((84, 84, 3), dtype=np.uint8)
        self.steps = 0
        return self.state
        
    def step(self, action):
        """执行动作"""
        # 这里是环境动态的实现
        self.steps += 1
        
        # 示例：简单的奖励函数
        reward = 1.0 if action == 1 else -0.1
        
        # 示例：判断是否终止
        done = self.steps >= 100
        
        # 更新状态
        self.state = np.random.randint(0, 255, size=(84, 84, 3), dtype=np.uint8)
        
        # 提供额外信息
        info = {}
        
        return self.state, reward, done, info
        
    def render(self, mode='human'):
        """渲染环境"""
        pass