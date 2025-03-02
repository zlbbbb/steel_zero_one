# 数据处理模块

## 概述
该模块用于处理钢铁行业数据集，提取特征参数并生成处理后的数据文件。

## 功能
- 加载原始钢铁行业数据
- 提取统计特征
- 数据标准化
- 相关性分析
- 时间特征提取（如果数据中包含时间字段）

## 输入
原始数据文件位置: `data/raw/steel_industry_data.csv`

## 输出
处理后的数据文件存储在 `data/processed/` 目录下:
- `processed_steel_data.csv`: 包含原始数据和标准化特征的完整数据集
- `feature_summary.csv`: 特征概述，包括数据类型和缺失值统计
- `statistical_features.csv`: 统计特征，包括每个数值列的均值、标准差、最小值、最大值等
- `high_correlations.csv`: 高相关性特征对（相关系数绝对值>0.7）

## 使用方法
在项目根目录下运行:
```bash
python src/data/process_data.py
```