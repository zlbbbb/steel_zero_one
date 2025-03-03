# 钢铁行业数据处理结果

处理时间: 2025-03-03 10:26:32

## 处理步骤
1. 加载原始数据文件
2. 检查数据质量（缺失值和异常值）
3. 数据类型转换和特征提取
4. 计算各特征间的皮尔逊相关系数
5. 生成相关系数可视化图表

## 文件说明
- `steel_data_processed.csv`: 处理后的数据文件
- `correlation_matrix.csv`: 皮尔逊相关系数矩阵
- `correlation_heatmap.png`: 相关系数热图
- `descriptive_stats.csv`: 数据描述性统计

## 深入分析结果
更新时间: 2025-03-03 10:27:13

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
