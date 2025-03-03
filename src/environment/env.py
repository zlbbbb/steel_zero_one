"""
Deep Q-Learning 自定义环境
"""
import gymnasium as gym
import pandas as pd
import numpy as np
import matplotlib as plt
import torch
import random 
#设置的最大值都达不到  功率因数除外  NSM是每天时刻的秒记表达方式
MAX_USAGE_POWER=168 #最大用电量
MAX_LAGG_POWER=100 #最大滞后电流无功功率
MAX_LEAD_POWER=30  #最大超前电流无功功率
MAX_TCO2=0.1       #二氧化碳最大排放量
MAX_LAGG_FACTOR=100 #最大滞后无功功率因数
MAX_LEAD_FACTOR=100 #最大超前无功功率因数
"""
神经网络 卷积神经网络来进行拟合Q值
"""
