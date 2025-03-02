import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from pathlib import Path
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']  # 首选 SimHei 字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def create_directory_if_not_exists(directory_path):
    """创建目录（如果不存在）"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"创建目录: {directory_path}")

def load_data(file_path):
    """加载CSV数据"""
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载数据，形状: {df.shape}")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def extract_features(df):
    """提取和处理数据特征"""
    print("开始提取特征...")
    
    # 1. 基本统计特征
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    stats_df = pd.DataFrame()
    
    for col in numerical_columns:
        stats_df[f'{col}_mean'] = [df[col].mean()]
        stats_df[f'{col}_std'] = [df[col].std()]
        stats_df[f'{col}_min'] = [df[col].min()]
        stats_df[f'{col}_max'] = [df[col].max()]
        stats_df[f'{col}_q25'] = [df[col].quantile(0.25)]
        stats_df[f'{col}_median'] = [df[col].median()]
        stats_df[f'{col}_q75'] = [df[col].quantile(0.75)]
    
    # 2. 标准化数据
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[numerical_columns]),
        columns=[f'{col}_scaled' for col in numerical_columns]
    )
    
    # 3. 相关性分析 - 这部分将在 pearson_correlation_analysis 函数中详细处理
    correlation_matrix = df[numerical_columns].corr().stack().reset_index()
    correlation_matrix.columns = ['Feature_1', 'Feature_2', 'Correlation']
    high_correlations = correlation_matrix[
        (correlation_matrix['Feature_1'] != correlation_matrix['Feature_2']) & 
        (correlation_matrix['Correlation'].abs() > 0.7)
    ]
    
    # 4. 时间特征（如果存在时间列）
    time_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    if time_columns:
        for col in time_columns:
            try:
                df[f'{col}_datetime'] = pd.to_datetime(df[col])
                df[f'{col}_hour'] = df[f'{col}_datetime'].dt.hour
                df[f'{col}_dayofweek'] = df[f'{col}_datetime'].dt.dayofweek
                df[f'{col}_month'] = df[f'{col}_datetime'].dt.month
            except:
                print(f"无法将 {col} 转换为日期时间")
    
    # 合并原始数据和标准化数据
    result_df = pd.concat([df, df_scaled], axis=1)
    
    # 生成特征汇总表
    feature_summary = pd.DataFrame({
        'Feature': df.columns,
        'DataType': df.dtypes.values,
        'NonNullCount': df.count().values,
        'NullPercentage': (1 - df.count() / len(df)) * 100
    })
    
    return {
        'processed_data': result_df,
        'feature_summary': feature_summary,
        'statistical_features': stats_df,
        'high_correlations': high_correlations,
        'numerical_columns': numerical_columns,
        'original_data': df
    }

def pearson_correlation_analysis(data_dict, output_dir):
    """详细的皮尔逊相关性分析"""
    print("开始进行皮尔逊相关性分析...")
    
    # 获取原始数据和数值列
    df = data_dict['original_data']
    numerical_columns = data_dict['numerical_columns']
    
    # 1. 计算完整的皮尔逊相关系数矩阵
    corr_matrix = df[numerical_columns].corr(method='pearson')
    
    # 2. 生成热力图并保存
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('皮尔逊相关系数热力图', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pearson_correlation_heatmap.png", dpi=300)
    plt.close()
    print(f"保存皮尔逊相关系数热力图到 {output_dir}/pearson_correlation_heatmap.png")
    
    # 3. 保存相关系数矩阵
    corr_matrix.to_csv(f"{output_dir}/pearson_correlation_matrix.csv")
    print(f"保存皮尔逊相关系数矩阵到 {output_dir}/pearson_correlation_matrix.csv")
    
    # 4. 找出对每个特征相关性最强的前3个特征
    top_correlations = {}
    for col in numerical_columns:
        # 排除自身相关性
        correlations = corr_matrix[col].drop(col).sort_values(ascending=False)
        top_correlations[col] = correlations.head(3)
    
    # 5. 保存每个特征的主要相关特征
    with open(f"{output_dir}/top_pearson_correlations.txt", 'w') as f:
        f.write("特征与其他特征的主要皮尔逊相关性分析:\n")
        f.write("=" * 60 + "\n")
        for feature, top_corrs in top_correlations.items():
            f.write(f"\n特征: {feature}\n")
            for other_feature, corr_value in top_corrs.items():
                f.write(f"  - 与 {other_feature} 的相关系数: {corr_value:.4f}\n")
            f.write("-" * 40 + "\n")
    print(f"保存主要相关性分析到 {output_dir}/top_pearson_correlations.txt")
    
    # 6. 计算每列与其他列的平均相关性
    avg_correlations = {}
    for col in numerical_columns:
        # 排除自身相关性
        avg_correlations[col] = corr_matrix[col].drop(col).abs().mean()
    
    # 将平均相关性从高到低排序
    avg_correlations = pd.Series(avg_correlations).sort_values(ascending=False)
    avg_correlations.to_csv(f"{output_dir}/average_pearson_correlations.csv", header=['平均相关性'])
    print(f"保存平均相关性到 {output_dir}/average_pearson_correlations.csv")
    
    # 7. 计算特征重要性评分（基于相关性）
    feature_importance = {}
    for col in numerical_columns:
        # 计算该列与所有其他列的绝对相关性总和
        feature_importance[col] = corr_matrix[col].drop(col).abs().sum()
    
    # 将特征重要性从高到低排序
    feature_importance = pd.Series(feature_importance).sort_values(ascending=False)
    feature_importance.to_csv(f"{output_dir}/feature_importance.csv", header=['特征重要性'])
    print(f"保存特征重要性评分到 {output_dir}/feature_importance.csv")
    
    return {
        'correlation_matrix': corr_matrix,
        'top_correlations': top_correlations,
        'avg_correlations': avg_correlations,
        'feature_importance': feature_importance
    }

def save_processed_data(data_dict, output_dir):
    """保存处理后的数据"""
    create_directory_if_not_exists(output_dir)
    
    # 保存主数据集
    data_dict['processed_data'].to_csv(f"{output_dir}/processed_steel_data.csv", index=False)
    print(f"保存处理后的数据到 {output_dir}/processed_steel_data.csv")
    
    # 保存特征汇总
    data_dict['feature_summary'].to_csv(f"{output_dir}/feature_summary.csv", index=False)
    print(f"保存特征汇总到 {output_dir}/feature_summary.csv")
    
    # 保存统计特征
    data_dict['statistical_features'].to_csv(f"{output_dir}/statistical_features.csv", index=False)
    print(f"保存统计特征到 {output_dir}/statistical_features.csv")
    
    # 保存相关性数据
    data_dict['high_correlations'].to_csv(f"{output_dir}/high_correlations.csv", index=False)
    print(f"保存高相关性特征到 {output_dir}/high_correlations.csv")

def main():
    """主函数"""
    # 定义文件路径
    base_path = Path(__file__).parent.parent.parent
    input_file = base_path / "data" / "raw" / "steel_industry_data.csv"
    output_dir = base_path / "data" / "processed"
    
    print(f"读取文件: {input_file}")
    print(f"输出目录: {output_dir}")
    
    # 加载数据
    df = load_data(input_file)
    if df is None:
        return
    
    # 提取特征
    processed_data_dict = extract_features(df)
    
    # 保存处理后的数据
    save_processed_data(processed_data_dict, output_dir)
    
    # 执行皮尔逊相关性分析
    create_directory_if_not_exists(output_dir / "correlation_analysis")
    pearson_results = pearson_correlation_analysis(processed_data_dict, output_dir / "correlation_analysis")
    
    print("数据处理完成!")

if __name__ == "__main__":
    main()