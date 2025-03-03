# -*- coding: utf-8 -*-
"""
钢铁行业数据处理脚本
- 加载data/raw/steel_industry_data.csv数据文件
- 进行简单数据处理
- 计算皮尔逊相关系数
- 生成相关系数可视化
- 保存处理结果到data/processed文件夹
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pathlib

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

def load_data():
    """加载原始数据文件"""
    print("正在加载原始数据...")
    # 确定数据路径
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                            'data', 'raw', 'steel_industry_data.csv')
    
    # 加载CSV文件
    df = pd.read_csv(data_path)
    print(f"成功加载数据，共 {df.shape[0]} 行, {df.shape[1]} 列")
    return df

def process_data(df):
    """简单数据处理"""
    print("正在处理数据...")
    
    # 数据质量检查
    print("检查数据质量...")
    missing_values = df.isnull().sum()
    print(f"缺失值统计:\n{missing_values}")
    
    # 转换日期列为datetime类型
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
    
    # 提取日期的年、月、日、小时、分钟信息
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    
    # 根据WeekStatus和Day_of_week创建一个工作时间区间（工作日/周末）
    df['work_period'] = df['WeekStatus'] + '_' + df['Day_of_week']
    
    # 数据概述
    print("\n数据基本统计信息:")
    print(df.describe())
    
    return df

def calculate_correlations(df):
    """计算皮尔逊相关系数"""
    print("\n计算皮尔逊相关系数...")
    
    # 选择数值型列进行相关系数计算
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_columns].corr(method='pearson')
    
    return correlation_matrix

def visualize_correlations(correlation_matrix, output_dir):
    """将相关系数矩阵进行可视化"""
    print("\n生成相关系数可视化图表...")
    
    # 设置图表大小
    plt.figure(figsize=(14, 12))
    
    # 创建热图
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                linewidths=0.5, vmin=-1, vmax=1)
    
    # 设置标题
    plt.title('特征之间的皮尔逊相关系数热图', fontsize=16)
    
    # 调整布局
    plt.tight_layout()
    
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    # 保存图表
    correlation_heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(correlation_heatmap_path, dpi=300, bbox_inches='tight')
    
    print(f"相关系数热图已保存至: {correlation_heatmap_path}")
    
    # 关闭图表窗口
    plt.close()

def save_results(df, correlation_matrix, output_dir):
    """保存处理结果"""
    print("\n保存处理结果...")
    
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    # 保存处理后的数据
    processed_data_path = os.path.join(output_dir, 'steel_data_processed.csv')
    df.to_csv(processed_data_path, index=False, encoding='utf-8')
    print(f"处理后的数据已保存至: {processed_data_path}")
    
    # 保存相关系数矩阵
    correlation_matrix_path = os.path.join(output_dir, 'correlation_matrix.csv')
    correlation_matrix.to_csv(correlation_matrix_path, encoding='utf-8')
    print(f"相关系数矩阵已保存至: {correlation_matrix_path}")
    
    # 生成描述性统计结果
    stats_path = os.path.join(output_dir, 'descriptive_stats.csv')
    df.describe().to_csv(stats_path)
    print(f"描述性统计结果已保存至: {stats_path}")
    
    # 将处理信息写入README文件
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# 钢铁行业数据处理结果\n\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 处理步骤\n")
        f.write("1. 加载原始数据文件\n")
        f.write("2. 检查数据质量（缺失值和异常值）\n")
        f.write("3. 数据类型转换和特征提取\n")
        f.write("4. 计算各特征间的皮尔逊相关系数\n")
        f.write("5. 生成相关系数可视化图表\n\n")
        f.write("## 文件说明\n")
        f.write("- `steel_data_processed.csv`: 处理后的数据文件\n")
        f.write("- `correlation_matrix.csv`: 皮尔逊相关系数矩阵\n")
        f.write("- `correlation_heatmap.png`: 相关系数热图\n")
        f.write("- `descriptive_stats.csv`: 数据描述性统计\n")
    
    print(f"README文件已写入: {readme_path}")

def main():
    """主函数"""
    start_time = datetime.now()
    print(f"开始处理时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 确定输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'data', 'processed')
    
    # 加载数据
    df = load_data()
    
    # 处理数据
    processed_df = process_data(df)
    
    # 计算相关系数
    correlation_matrix = calculate_correlations(processed_df)
    
    # 可视化相关系数
    visualize_correlations(correlation_matrix, output_dir)
    
    # 保存处理结果
    save_results(processed_df, correlation_matrix, output_dir)
    
    end_time = datetime.now()
    processing_time = end_time - start_time
    print(f"\n处理完成!")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总处理时间: {processing_time.total_seconds():.2f} 秒")

if __name__ == "__main__":
    main()