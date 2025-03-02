import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from gymnasium import spaces
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 最大值常量定义
MAX_USAGE_POWER = 168             # 最大用电量
MAX_LAGG_REAC_POWER = 100         # 最大滞后电流无功功率
MAX_LEAD_REAC_POWER = 30          # 最大超前电流无功功率
MAX_TCO2 = 0.1                    # 最大二氧化碳排放量
MAX_LAGG_REAC_FAC = 100           # 最大滞后电流无功功率因数
MAX_LEAD_REAC_FAC = 100           # 最大超前电流无功功率因数
MAX_NSM = 86400                   # 用秒表示一天中的时刻

class PowerDemandPredictionEnv(gym.Env):
    """
    电力需求预测环境 - 专为预测未来电力负载而设计
    基于15分钟为一个采样单位节点
    
    状态空间包含:
    - 当前时间特征 (小时, 星期几, 是否工作日)
    - 历史电力使用数据及相关特征 (过去n个时间步)
    
    动作空间:
    - 连续值，表示对未来用电量的预测值
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, df=None, history_window=96, forecast_horizon=4, render_mode=None):
        """
        初始化电力需求预测环境
        
        参数:
        - df: 包含电力数据的DataFrame
        - history_window: 历史数据窗口长度 (过去多少个15分钟采样点的数据，默认96表示一天)
        - forecast_horizon: 预测时间长度 (预测未来多少个15分钟采样点，默认4表示未来1小时)
        - render_mode: 渲染模式
        """
        super(PowerDemandPredictionEnv, self).__init__()
        
        self.render_mode = render_mode
        self.history_window = history_window  # 历史窗口长度
        self.forecast_horizon = forecast_horizon  # 预测时间长度
        
        # 加载数据或创建示例数据
        if df is not None:
            self.df = df
        else:
            print("未提供数据，创建随机示例数据...")
            self.df = self._create_sample_data()
        
        # 检查必要的列是否存在
        required_columns = ['date', 'Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh',
                           'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)', 
                           'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM']
        
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"数据中缺少以下列: {missing_columns}")
        
        # 将日期列转换为datetime类型
        if 'date' in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            self.df['date'] = pd.to_datetime(self.df['date'])
            
        # 提取时间特征
        self.df['hour'] = self.df['date'].dt.hour
        self.df['minute'] = self.df['date'].dt.minute
        self.df['time_of_day'] = self.df['hour'] + self.df['minute'] / 60.0  # 一天中的时间(小时)
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['is_weekday'] = self.df['day_of_week'].apply(lambda x: 1 if x < 5 else 0)
        self.df['month'] = self.df['date'].dt.month
        self.df['day'] = self.df['date'].dt.day
        self.df['quarter_of_day'] = (self.df['hour'] * 4 + self.df['minute'] // 15)  # 一天中的第几个15分钟段
        
        # 定义动作空间 (连续值，表示预测的负载)
        self.action_space = spaces.Box(
            low=0, 
            high=1,  # 这里使用归一化的值
            shape=(self.forecast_horizon,), 
            dtype=np.float32
        )
        
        # 定义观察空间
        # 观察向量包含:
        # - 6个时间特征 (小时+分钟的小数表示, 是第几个15分钟段, 星期几, 是否工作日, 月份, 日期)
        # - history_window个历史电力使用及相关数据 (每个时间步7个特征)
        obs_dim = 6 + self.history_window * 7
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # 初始化环境状态
        self.current_step = None
        self.state = None
        self.steps_done = 0
        self.max_steps = len(self.df) - self.history_window - self.forecast_horizon
        self.performance_metrics = {
            'mse': [], 'mae': [], 'r2': [], 
            'predictions': [], 'actual_values': [],
            'timestamps': []
        }
        
        # 渲染相关
        self.fig = None
        self.axes = None
        self.prediction_history = []  # 存储预测历史
    
    def _create_sample_data(self, num_days=60):
        """
        创建示例数据用于测试
        基于15分钟为一个采样单位节点
        """
        num_samples = num_days * 24 * 4  # 15分钟一个采样点，一天96个点
        start_date = datetime(2025, 1, 1)  # 从2025年1月1日开始
        dates = [start_date + timedelta(minutes=15*i) for i in range(num_samples)]
        
        # 生成用电量数据 (考虑时间模式)
        usage_pattern = []
        for date in dates:
            hour = date.hour
            minute = date.minute
            time_of_day = hour + minute/60.0
            day_of_week = date.weekday()
            is_weekday = day_of_week < 5
            month = date.month
            day = date.day
            
            # 季节性影响 (冬夏用电量高，春秋较低)
            season_factor = 1.0
            if month in [12, 1, 2]:  # 冬季
                season_factor = 1.2
            elif month in [6, 7, 8]:  # 夏季
                season_factor = 1.3
            
            # 工作日模式
            if is_weekday:
                if 8 <= time_of_day < 12:  # 上午工作时间
                    base = 120 + 20 * np.sin((time_of_day-8)/4 * np.pi)
                elif 12 <= time_of_day < 14:  # 午休
                    base = 80 + 10 * np.sin((time_of_day-12)/2 * np.pi)
                elif 14 <= time_of_day < 18:  # 下午工作时间
                    base = 130 + 15 * np.sin((time_of_day-14)/4 * np.pi)
                elif 18 <= time_of_day < 22:  # 晚上
                    base = 70 - 5 * (time_of_day-18)
                else:  # 夜间
                    base = 40
            # 周末模式
            else:
                if 9 <= time_of_day < 20:  # 白天
                    base = 60 + 10 * np.sin((time_of_day-9)/11 * np.pi)
                else:  # 夜间
                    base = 35
            
            # 添加随机噪声和季节性影响
            usage = max(1, base * season_factor * (0.9 + 0.2 * np.random.random()))
            usage_pattern.append(usage)
        
        # 基于用电量生成其他特征
        lagging_reactive = []
        leading_reactive = []
        co2 = []
        lagging_factor = []
        leading_factor = []
        
        for i, usage in enumerate(usage_pattern):
            # 添加一些趋势和季节性变化
            time_factor = i / num_samples
            seasonal_factor = np.sin(2 * np.pi * i / (96 * 7))  # 一周的季节性
            
            # 生成相关数据，确保与用电量有一定相关性
            lag_reactive = usage * (0.4 + 0.2 * np.random.random() + 0.1 * seasonal_factor)
            lead_reactive = usage * (0.15 + 0.1 * np.random.random() - 0.05 * seasonal_factor)
            co2_emission = usage * (0.0005 + 0.0001 * np.random.random() + 0.00005 * time_factor)
            lag_factor = 75 + 15 * np.random.random() + 5 * seasonal_factor
            lead_factor = 85 + 10 * np.random.random() - 3 * seasonal_factor
            
            lagging_reactive.append(lag_reactive)
            leading_reactive.append(lead_reactive)
            co2.append(co2_emission)
            lagging_factor.append(lag_factor)
            leading_factor.append(lead_factor)
        
        # NSM (用秒表示的一天中的时刻)
        nsm = []
        for date in dates:
            seconds_since_midnight = date.hour * 3600 + date.minute * 60 + date.second
            nsm.append(seconds_since_midnight)
        
        # 创建DataFrame
        data = {
            'date': dates,
            'Usage_kWh': usage_pattern,
            'Lagging_Current_Reactive.Power_kVarh': lagging_reactive,
            'Leading_Current_Reactive_Power_kVarh': leading_reactive,
            'CO2(tCO2)': co2,
            'Lagging_Current_Power_Factor': lagging_factor,
            'Leading_Current_Power_Factor': leading_factor,
            'NSM': nsm
        }
        
        return pd.DataFrame(data)
    
    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)
        
        # 确保有足够的历史数据和未来数据用于训练
        min_idx = self.history_window
        max_idx = len(self.df) - self.forecast_horizon - 1
        
        if max_idx <= min_idx:
            raise ValueError("数据集太小，无法同时满足历史窗口和预测窗口的要求")
        
        # 随机选择一个起点
        self.current_step = np.random.randint(min_idx, max_idx)
        self.steps_done = 0
        
        # 获取当前状态
        self.state = self._get_observation()
        
        # 重置性能指标
        self.performance_metrics = {
            'mse': [], 'mae': [], 'r2': [], 
            'predictions': [], 'actual_values': [],
            'timestamps': []
        }
        self.prediction_history = []
        
        # 对于gymnasium 0.28+版本兼容，返回额外信息
        info = {}
        return self.state, info
    
    def _get_observation(self):
        """构建当前观察向量"""
        # 当前日期和时间特征
        current_row = self.df.iloc[self.current_step]
        time_of_day = current_row['time_of_day'] / 24.0  # 归一化 (0-1)
        quarter_of_day = current_row['quarter_of_day'] / 96.0  # 归一化 (0-1)
        day_of_week = current_row['day_of_week'] / 6.0  # 归一化 (0-1)
        is_weekday = current_row['is_weekday']  # 二元值 (0或1)
        month = current_row['month'] / 12.0  # 归一化 (0-1)
        day = current_row['day'] / 31.0  # 归一化 (0-1)
        
        # 历史电力数据窗口
        obs_window = []
        for i in range(self.current_step - self.history_window, self.current_step):
            row = self.df.iloc[i]
            # 归一化每个特征
            window_data = [
                row['Usage_kWh'] / MAX_USAGE_POWER,
                row['Lagging_Current_Reactive.Power_kVarh'] / MAX_LAGG_REAC_POWER,
                row['Leading_Current_Reactive_Power_kVarh'] / MAX_LEAD_REAC_POWER,
                row['CO2(tCO2)'] / MAX_TCO2,
                row['Lagging_Current_Power_Factor'] / MAX_LAGG_REAC_FAC,
                row['Leading_Current_Power_Factor'] / MAX_LEAD_REAC_FAC,
                row['NSM'] / MAX_NSM
            ]
            obs_window.extend(window_data)
        
        # 组合所有观察值
        obs = np.array([time_of_day, quarter_of_day, day_of_week, is_weekday, month, day] + obs_window, dtype=np.float32)
        
        return obs
    
    # def step(self, action):
    #     """
    #     执行一步预测
        
    #     参数:
    #     - action: 对未来forecast_horizon个时间点的用电量预测(归一化值)
        
    #     返回:
    #     - observation: 新状态
    #     - reward: 根据预测准确度计算的奖励
    #     - terminated: 是否结束
    #     - truncated: 是否截断
    #     - info: 额外信息
    #     """
    #     # 确保动作在有效范围内
    #     action = np.clip(action,