import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import scipy.stats as stats
from matplotlib.font_manager import FontProperties

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

def detailed_correlation_analysis(df, output_dir):
    """进行详细的相关性分析"""
    # 选择数值列
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # 创建保存结果的目录
    create_directory_if_not_exists(output_dir)
    
    # 1. 计算皮尔逊相关系数矩阵
    print("计算皮尔逊相关系数...")
    pearson_corr = df[numerical_columns].corr(method='pearson')
    
    # 2. 计算斯皮尔曼相关系数矩阵，与皮尔逊相关系数对比
    print("计算斯皮尔曼相关系数...")
    spearman_corr = df[numerical_columns].corr(method='spearman')
    
    # 3. 计算肯德尔相关系数矩阵，与皮尔逊相关系数对比
    print("计算肯德尔相关系数...")
    kendall_corr = df[numerical_columns].corr(method='kendall')
    
    # 保存相关系数矩阵
    pearson_corr.to_csv(f"{output_dir}/pearson_correlation.csv")
    spearman_corr.to_csv(f"{output_dir}/spearman_correlation.csv")
    kendall_corr.to_csv(f"{output_dir}/kendall_correlation.csv")
    
    # 4. 绘制皮尔逊相关性热力图（修复标题方块问题）
    print("绘制皮尔逊相关性热力图...")
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # 使用 sns.heatmap 并控制注释大小
    sns.heatmap(pearson_corr, mask=mask, annot=True, fmt=".2f", cmap=cmap, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, annot_kws={"size": 8})
    
    # 设置标题并解决方块问题
    plt.title('皮尔逊相关系数热力图', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pearson_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 分析主要特征的分布情况和相关性散点图
    print("绘制主要特征的分布和散点图...")
    # 选择相关系数排名前10的特征对
    corr_pairs = []
    for i, col1 in enumerate(numerical_columns):
        for j, col2 in enumerate(numerical_columns):
            if i < j:  # 避免重复和自相关
                corr_pairs.append((col1, col2, abs(pearson_corr.loc[col1, col2])))
    
    # 按相关系数绝对值排序并取前10
    top_pairs = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:10]
    
    for col1, col2, corr_value in top_pairs:
        plt.figure(figsize=(10, 8))
        
        # 散点图
        plt.scatter(df[col1], df[col2], alpha=0.6)
        
        # 添加趋势线
        z = np.polyfit(df[col1], df[col2], 1)
        p = np.poly1d(z)
        plt.plot(df[col1], p(df[col1]), "r--", alpha=0.8)
        
        # 计算p值
        r, p_value = stats.pearsonr(df[col1], df[col2])
        
        plt.title(f'{col1} vs {col2}\n相关系数: {corr_value:.4f}, p值: {p_value:.4e}', fontsize=14)
        plt.xlabel(col1, fontsize=12)
        plt.ylabel(col2, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/scatter_{col1}_vs_{col2}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. 绘制对角线图：同时展示分布和相关性
    print("绘制特征分布和相关性的对角线图...")
    selected_columns = [pair[0] for pair in top_pairs[:5]] + [pair[1] for pair in top_pairs[:5]]
    selected_columns = list(set(selected_columns))[:6]  # 取前6个不重复特征
    
    sns.set(style="ticks")
    sns.pairplot(df[selected_columns], diag_kind="kde", markers="o", 
                 plot_kws={'alpha': 0.5, 's': 20, 'edgecolor': 'k'},
                 diag_kws={'shade': True})
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pairplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. 绘制分层热力图，展示特征分组
    print("绘制分层热力图...")
    plt.figure(figsize=(14, 10))
    # 使用层次聚类重新排序相关矩阵
    corr = pearson_corr
    # 生成聚类
    sns.clustermap(corr, cmap=cmap, annot=True, fmt=".2f", linewidths=.5, 
                  annot_kws={"size": 8}, figsize=(15, 12))
    plt.savefig(f"{output_dir}/clustered_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. 绘制相关性网络图
    print("生成特征重要性排序...")
    # 计算每个特征的平均相关性强度
    avg_importance = pearson_corr.abs().mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    avg_importance.plot(kind='bar')
    plt.title('特征重要性（基于平均绝对相关系数）', fontsize=14)
    plt.xlabel('特征', fontsize=12)
    plt.ylabel('平均相关性强度', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 9. 绘制热力图的另一种可视化方式，避免标题方块问题
    print("绘制替代热力图...")
    plt.figure(figsize=(14, 12))
    
    # 使用matplotlib的imshow而不是seaborn的heatmap来避免标题问题
    im = plt.imshow(pearson_corr, cmap='coolwarm', interpolation='none')
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('相关系数')
    
    # 设置刻度和标签
    plt.xticks(np.arange(len(pearson_corr.columns)), pearson_corr.columns, rotation=45, ha='right')
    plt.yticks(np.arange(len(pearson_corr.index)), pearson_corr.index)
    
    # 添加数值注释
    for i in range(len(pearson_corr.index)):
        for j in range(len(pearson_corr.columns)):
            text = plt.text(j, i, f"{pearson_corr.iloc[i, j]:.2f}",
                           ha="center", va="center", color="black" if abs(pearson_corr.iloc[i, j]) < 0.5 else "white",
                           fontsize=8)
    
    plt.title('皮尔逊相关系数矩阵 (Alternative Visualization)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pearson_matrix_alt.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'kendall': kendall_corr,
        'top_correlated_pairs': top_pairs
    }

def main():
    """主函数"""
    # 定义文件路径
    base_path = Path(__file__).parent.parent.parent
    input_file = base_path / "data" / "raw" / "steel_industry_data.csv"
    output_dir = base_path / "data" / "processed" / "correlation_analysis"
    
    print(f"读取文件: {input_file}")
    print(f"输出目录: {output_dir}")
    
    # 加载数据
    df = load_data(input_file)
    if df is None:
        return
    
    # 执行详细的相关性分析
    results = detailed_correlation_analysis(df, output_dir)
    
    # 输出分析摘要
    with open(f"{output_dir}/correlation_summary.txt", 'w', encoding='utf-8') as f:
        f.write("# 相关性分析摘要报告\n\n")
        f.write(f"## 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 数据形状: {df.shape[0]} 行 x {df.shape[1]} 列\n\n")
        
        f.write("## 最强相关性特征对 (Top 10):\n\n")
        for i, (col1, col2, corr) in enumerate(results['top_correlated_pairs'][:10], 1):
            f.write(f"{i}. {col1} 与 {col2}: 相关系数 = {corr:.4f}\n")
        
        f.write("\n## 相关性分析结果文件:\n\n")
        f.write("- pearson_correlation.csv - 皮尔逊相关系数矩阵\n")
        f.write("- spearman_correlation.csv - 斯皮尔曼相关系数矩阵\n")
        f.write("- kendall_correlation.csv - 肯德尔相关系数矩阵\n")
        f.write("- pearson_heatmap.png - 皮尔逊相关性热力图\n")
        f.write("- pearson_matrix_alt.png - 替代方式绘制的皮尔逊相关性矩阵\n")
        f.write("- clustered_heatmap.png - 分层聚类热力图\n")
        f.write("- feature_importance.png - 特征重要性排序\n")
        f.write("- pairplot.png - 主要特征分布和相关性图\n")
        f.write("- scatter_*.png - 各个特征对的散点图和趋势线\n")
    
    print("相关性分析完成！")

if __name__ == "__main__":
    main()