# -*- coding: utf-8 -*-
"""
钢铁行业数据分析脚本
- 加载处理后的数据进行更深入的分析
- 生成其他有用的图表和洞见
- 保存分析结果到data/processed文件夹
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

def load_processed_data(output_dir):
    """加载处理后的数据"""
    processed_data_path = os.path.join(output_dir, 'steel_data_processed.csv')
    df = pd.read_csv(processed_data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def analyze_usage_by_time(df, output_dir):
    """分析不同时间段的用电量情况"""
    print("分析不同时间段的用电量情况...")
    
    # 按小时统计平均用电量
    hourly_usage = df.groupby('hour')['Usage_kWh'].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='hour', y='Usage_kWh', data=hourly_usage, color='steelblue')
    plt.title('各小时平均用电量')
    plt.xlabel('小时')
    plt.ylabel('平均用电量 (kWh)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    hourly_usage_path = os.path.join(output_dir, 'hourly_usage.png')
    plt.savefig(hourly_usage_path, dpi=300, bbox_inches='tight')
    print(f"小时用电量分析图已保存至: {hourly_usage_path}")
    plt.close()
    
    # 按星期几统计平均用电量
    daily_usage = df.groupby('Day_of_week')['Usage_kWh'].mean().reset_index()
    # 确定星期几的顺序
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_usage['Day_of_week'] = pd.Categorical(daily_usage['Day_of_week'], categories=day_order, ordered=True)
    daily_usage = daily_usage.sort_values('Day_of_week')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Day_of_week', y='Usage_kWh', data=daily_usage, palette='viridis')
    plt.title('各星期平均用电量')
    plt.xlabel('星期几')
    plt.ylabel('平均用电量 (kWh)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    daily_usage_path = os.path.join(output_dir, 'daily_usage.png')
    plt.savefig(daily_usage_path, dpi=300, bbox_inches='tight')
    print(f"星期用电量分析图已保存至: {daily_usage_path}")
    plt.close()
    
    # 工作日vs周末
    weekstatus_usage = df.groupby('WeekStatus')['Usage_kWh'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='WeekStatus', y='Usage_kWh', data=weekstatus_usage, palette='Set2')
    plt.title('工作日vs周末平均用电量')
    plt.xlabel('日期类型')
    plt.ylabel('平均用电量 (kWh)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    weekstatus_usage_path = os.path.join(output_dir, 'weekstatus_usage.png')
    plt.savefig(weekstatus_usage_path, dpi=300, bbox_inches='tight')
    print(f"工作日/周末用电量分析图已保存至: {weekstatus_usage_path}")
    plt.close()

def analyze_load_types(df, output_dir):
    """分析不同负载类型的用电情况"""
    print("分析不同负载类型的用电情况...")
    
    # 按负载类型统计平均用电量和CO2排放
    load_stats = df.groupby('Load_Type')[['Usage_kWh', 'CO2(tCO2)']].mean().reset_index()
    
    # 用电量比较
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Load_Type', y='Usage_kWh', data=load_stats, order=['Light_Load', 'Medium_Load', 'Maximum_Load'], palette='rocket')
    plt.title('不同负载类型的平均用电量')
    plt.xlabel('负载类型')
    plt.ylabel('平均用电量 (kWh)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    load_usage_path = os.path.join(output_dir, 'load_type_usage.png')
    plt.savefig(load_usage_path, dpi=300, bbox_inches='tight')
    print(f"负载类型用电量分析图已保存至: {load_usage_path}")
    plt.close()
    
    # CO2排放比较
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Load_Type', y='CO2(tCO2)', data=load_stats, order=['Light_Load', 'Medium_Load', 'Maximum_Load'], palette='crest')
    plt.title('不同负载类型的平均CO2排放量')
    plt.xlabel('负载类型')
    plt.ylabel('平均CO2排放量 (tCO2)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    load_co2_path = os.path.join(output_dir, 'load_type_co2.png')
    plt.savefig(load_co2_path, dpi=300, bbox_inches='tight')
    print(f"负载类型CO2排放分析图已保存至: {load_co2_path}")
    plt.close()
    
    # 负载类型分布
    load_counts = df['Load_Type'].value_counts().reset_index()
    load_counts.columns = ['Load_Type', 'Count']
    
    # 设置顺序
    load_counts['Load_Type'] = pd.Categorical(load_counts['Load_Type'], 
                                             categories=['Light_Load', 'Medium_Load', 'Maximum_Load'],
                                             ordered=True)
    load_counts = load_counts.sort_values('Load_Type')
    
    plt.figure(figsize=(10, 6))
    plt.pie(load_counts['Count'], labels=load_counts['Load_Type'], autopct='%1.1f%%', 
            colors=sns.color_palette('pastel'), startangle=140, shadow=True)
    plt.title('负载类型分布')
    plt.axis('equal')  # 保持圆形
    plt.tight_layout()
    
    # 保存图表
    load_dist_path = os.path.join(output_dir, 'load_type_distribution.png')
    plt.savefig(load_dist_path, dpi=300, bbox_inches='tight')
    print(f"负载类型分布图已保存至: {load_dist_path}")
    plt.close()

def analyze_power_factor(df, output_dir):
    """分析功率因数情况"""
    print("分析功率因数情况...")
    
    # 选择相关列
    pf_cols = ['Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor']
    
    # 不同负载类型的功率因数比较
    pf_by_load = df.groupby('Load_Type')[pf_cols].mean().reset_index()
    pf_long = pd.melt(pf_by_load, id_vars=['Load_Type'], value_vars=pf_cols, 
                     var_name='Power_Factor_Type', value_name='Power_Factor')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Load_Type', y='Power_Factor', hue='Power_Factor_Type', data=pf_long, palette='Set1')
    plt.title('不同负载类型下的功率因数比较')
    plt.xlabel('负载类型')
    plt.ylabel('功率因数 (%)')
    plt.legend(title='功率因数类型')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    pf_load_path = os.path.join(output_dir, 'power_factor_by_load.png')
    plt.savefig(pf_load_path, dpi=300, bbox_inches='tight')
    print(f"功率因数分析图已保存至: {pf_load_path}")
    plt.close()
    
    # 功率因数与用电量的散点图
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Lagging_Current_Power_Factor', y='Usage_kWh', data=df, alpha=0.6, hue='Load_Type', palette='viridis')
    plt.title('滞后功率因数 vs 用电量')
    plt.xlabel('滞后功率因数 (%)')
    plt.ylabel('用电量 (kWh)')
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='Leading_Current_Power_Factor', y='Usage_kWh', data=df, alpha=0.6, hue='Load_Type', palette='viridis')
    plt.title('超前功率因数 vs 用电量')
    plt.xlabel('超前功率因数 (%)')
    plt.ylabel('用电量 (kWh)')
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图表
    pf_usage_path = os.path.join(output_dir, 'power_factor_vs_usage.png')
    plt.savefig(pf_usage_path, dpi=300, bbox_inches='tight')
    print(f"功率因数与用电量关系图已保存至: {pf_usage_path}")
    plt.close()

def analyze_reactive_power(df, output_dir):
    """分析无功功率情况"""
    print("分析无功功率情况...")
    
    # 选择相关列
    reactive_cols = ['Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh']
    
    # 无功功率与用电量的关系
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='Lagging_Current_Reactive.Power_kVarh', y='Usage_kWh', data=df, alpha=0.5, hue='Load_Type')
    plt.title('滞后无功功率 vs 用电量')
    plt.xlabel('滞后无功功率 (kVarh)')
    plt.ylabel('用电量 (kWh)')
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='Leading_Current_Reactive_Power_kVarh', y='Usage_kWh', data=df, alpha=0.5, hue='Load_Type')
    plt.title('超前无功功率 vs 用电量')
    plt.xlabel('超前无功功率 (kVarh)')
    plt.ylabel('用电量 (kWh)')
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 保存图表
    reactive_usage_path = os.path.join(output_dir, 'reactive_power_vs_usage.png')
    plt.savefig(reactive_usage_path, dpi=300, bbox_inches='tight')
    print(f"无功功率与用电量关系图已保存至: {reactive_usage_path}")
    plt.close()
    
    # 不同负载类型的平均无功功率比较
    reactive_by_load = df.groupby('Load_Type')[reactive_cols].mean().reset_index()
    reactive_long = pd.melt(reactive_by_load, id_vars=['Load_Type'], value_vars=reactive_cols, 
                           var_name='Reactive_Power_Type', value_name='Reactive_Power')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Load_Type', y='Reactive_Power', hue='Reactive_Power_Type', data=reactive_long, palette='Set2')
    plt.title('不同负载类型下的平均无功功率')
    plt.xlabel('负载类型')
    plt.ylabel('无功功率 (kVarh)')
    plt.legend(title='无功功率类型')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    reactive_load_path = os.path.join(output_dir, 'reactive_power_by_load.png')
    plt.savefig(reactive_load_path, dpi=300, bbox_inches='tight')
    print(f"不同负载类型无功功率分析图已保存至: {reactive_load_path}")
    plt.close()

def update_readme(output_dir):
    """更新README文件，添加分析结果"""
    readme_path = os.path.join(output_dir, 'README.md')
    
    # 读取现有的README内容
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
    except FileNotFoundError:
        readme_content = "# 钢铁行业数据处理与分析结果\n\n"
    
    # 添加新的分析内容
    analysis_content = f"""
## 深入分析结果
更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 时间模式分析
- `hourly_usage.png`: 各小时平均用电量分析
- `daily_usage.png`: 各星期平均用电量分析
- `weekstatus_usage.png`: 工作日与周末用电量对比

### 负载类型分析
- `load_type_usage.png`: 不同负载类型的平均用电量
- `load_type_co2.png`: 不同负载类型的平均CO2排放量
- `load_type_distribution.png`: 负载类型分布情况

### 功率因数分析
- `power_factor_by_load.png`: 不同负载类型下的功率因数比较
- `power_factor_vs_usage.png`: 功率因数与用电量关系

### 无功功率分析
- `reactive_power_vs_usage.png`: 无功功率与用电量关系
- `reactive_power_by_load.png`: 不同负载类型无功功率分析
"""
    
    # 如果没有深入分析部分，则添加；否则更新
    if "## 深入分析结果" not in readme_content:
        updated_content = readme_content + analysis_content
    else:
        # 替换现有的深入分析部分
        updated_content = readme_content.split("## 深入分析结果")[0] + analysis_content
    
    # 写入更新后的README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"README文件已更新: {readme_path}")

def main():
    """主函数"""
    start_time = datetime.now()
    print(f"开始分析时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 确定输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'data', 'processed')
    ensure_dir(output_dir)
    
    # 加载处理后的数据
    df = load_processed_data(output_dir)
    print(f"成功加载处理后的数据，共 {df.shape[0]} 行, {df.shape[1]} 列")
    
    # 分析不同时间段的用电量
    analyze_usage_by_time(df, output_dir)
    
    # 分析不同负载类型
    analyze_load_types(df, output_dir)
    
    # 分析功率因数
    analyze_power_factor(df, output_dir)
    
    # 分析无功功率
    analyze_reactive_power(df, output_dir)
    
    # 更新README
    update_readme(output_dir)
    
    end_time = datetime.now()
    processing_time = end_time - start_time
    print(f"\n分析完成!")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总分析时间: {processing_time.total_seconds():.2f} 秒")

if __name__ == "__main__":
    main()