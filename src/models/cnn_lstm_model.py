import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, Model # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Input, Concatenate, Dropout# type: ignore
from tensorflow.keras.optimizers import Adam# type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau# type: ignore
import tensorflow as tf
import seaborn as sns
import time
from matplotlib.ticker import PercentFormatter

# 设置随机种子以获得可重复的结果
np.random.seed(42)
tf.random.set_seed(42)

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

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

def preprocess_data(df, use_scaled=True):
    """预处理数据，提取特征和目标变量"""
    # 选择特征列
    features = ['Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)', 'Lagging_Current_Power_Factor']
    target = 'Usage_kWh'  # 预测的是电力使用量
    
    print(f"使用特征: {features}")
    print(f"目标变量: {target}")
    
    # 检查特征和目标列是否存在
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        print(f"警告: 数据中缺少以下列: {missing_cols}")
        if use_scaled:
            # 检查是否有标准化列
            scaled_features = [f + '_scaled' for f in features]
            missing_scaled = [col for col in scaled_features if col not in df.columns]
            if missing_scaled:
                print(f"警告: 数据中也缺少标准化列: {missing_scaled}")
                print("尝试创建标准化特征...")
                # 创建标准化特征
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[features])
                for i, feature in enumerate(features):
                    df[f'{feature}_scaled'] = scaled_data[:, i]
                print("已创建标准化特征")
    
    # 如果使用已经标准化的列
    if use_scaled:
        scaled_features = [f + '_scaled' for f in features]
        X = df[scaled_features].values
        # 检查目标变量是否有标准化版本
        scaled_target = target + '_scaled'
        if scaled_target in df.columns:
            y = df[scaled_target].values
            print(f"使用标准化目标变量: {scaled_target}")
        else:
            y = df[target].values
            print(f"使用原始目标变量: {target}")
        used_features = scaled_features
    else:
        X = df[features].values
        y = df[target].values
        used_features = features
    
    # 分割训练集和测试集
    train_size = int(len(X) * 0.75)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"训练集形状: X={X_train.shape}, y={y_train.shape}")
    print(f"测试集形状: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, X_test, y_train, y_test, used_features, target

def create_sequences(X, y, time_steps=10):
    """创建序列数据用于LSTM模型"""
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"序列数据形状: X_seq={X_seq.shape}, y_seq={y_seq.shape}")
    return X_seq, y_seq

def build_cnn_lstm_model(input_shape, dropout_rate=0.2):
    """构建CNN-LSTM组合模型"""
    model = Sequential()
    
    # CNN部分
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(dropout_rate))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    # LSTM部分
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(30))
    model.add(Dropout(dropout_rate))
    
    # 输出层
    model.add(Dense(1))
    
    # 打印模型摘要
    model.summary()
    
    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model

def evaluate_model(model, X_test, y_test, target_scaler=None):
    """评估模型性能"""
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # 如果有scaler并且使用了标准化数据，将预测值和实际值转换回原始尺度
    if target_scaler is not None:
        y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_original = target_scaler.inverse_transform(y_pred).flatten()
    else:
        y_test_original = y_test
        y_pred_original = y_pred.flatten()
    
    # 计算评估指标
    mse = mean_squared_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)
    
    # 计算平均绝对百分比误差 (MAPE)
    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
    
    # 计算准确率 (100% - MAPE)
    accuracy = 100 - mape
    
    print(f"\n模型评估结果:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"准确率: {accuracy:.2f}%")
    print(f"预测用时: {prediction_time:.4f} 秒")
    
    # 计算每个预测点的准确率
    individual_accuracies = 100 - (np.abs((y_test_original - y_pred_original) / y_test_original) * 100)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'accuracy': accuracy,
        'y_pred': y_pred_original,
        'y_test': y_test_original,
        'individual_accuracies': individual_accuracies,
        'prediction_time': prediction_time
    }

def plot_results(y_test, y_pred, title, output_dir, individual_accuracies=None):
    """绘制预测结果图"""
    # 1. 预测值与实际值对比图
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='实际值', color='blue', linewidth=2)
    plt.plot(y_pred, label='预测值', color='red', linewidth=2, alpha=0.7)
    plt.title(title, fontsize=14)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('目标变量值', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 散点图
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.6, color='#2E86C1')
    # 添加理想预测线
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.title(f"{title} - 实际值 vs 预测值", fontsize=14)
    plt.xlabel('实际值', fontsize=12)
    plt.ylabel('预测值', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title.replace(' ', '_')}_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 预测精度变化曲线图 (显示百分比)
    if individual_accuracies is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(individual_accuracies, color='green', linewidth=2)
        plt.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='90%精度线')
        plt.axhline(y=individual_accuracies.mean(), color='blue', linestyle='--', 
                   alpha=0.7, label=f'平均精度: {individual_accuracies.mean():.2f}%')
        plt.title(f"{title} - 预测精度变化曲线", fontsize=14)
        plt.xlabel('样本索引', fontsize=12)
        plt.ylabel('精度 (%)', fontsize=12)
        plt.ylim(0, 110)  # 设置y轴范围为0-110%
        plt.gca().yaxis.set_major_formatter(PercentFormatter())  # 显示百分比符号
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{title.replace(' ', '_')}_accuracy_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 精度分布直方图
        plt.figure(figsize=(10, 6))
        sns.histplot(individual_accuracies, bins=20, kde=True, color='#3498DB')
        plt.axvline(x=individual_accuracies.mean(), color='red', linestyle='--', 
                   label=f'平均精度: {individual_accuracies.mean():.2f}%')
        plt.title(f"{title} - 预测精度分布", fontsize=14)
        plt.xlabel('精度 (%)', fontsize=12)
        plt.ylabel('频数', fontsize=12)
        plt.gca().xaxis.set_major_formatter(PercentFormatter())  # 显示百分比符号
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{title.replace(' ', '_')}_accuracy_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 误差百分比柱状图（排序后）
        error_percentages = 100 - individual_accuracies
        sorted_indices = np.argsort(error_percentages)[-20:]  # 取误差最大的20个点
        top_errors = error_percentages[sorted_indices]
        top_indices = sorted_indices
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_errors)), top_errors, color='#E74C3C')
        plt.title(f"{title} - Top 20 误差百分比", fontsize=14)
        plt.xlabel('样本索引', fontsize=12)
        plt.ylabel('误差 (%)', fontsize=12)
        plt.gca().yaxis.set_major_formatter(PercentFormatter())  # 显示百分比符号
        plt.xticks(range(len(top_errors)), top_indices, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{title.replace(' ', '_')}_top_errors.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # 定义文件路径
    base_path = Path(__file__).parent.parent.parent
    data_file = base_path / "data" / "processed" / "processed_steel_data.csv"
    output_dir = base_path / "models" / "cnn_lstm_results"
    model_dir = base_path / "models" / "saved_models"
    
    create_directory_if_not_exists(output_dir)
    create_directory_if_not_exists(model_dir)
    
    print(f"读取数据文件: {data_file}")
    print(f"输出目录: {output_dir}")
    
    # 加载数据
    df = load_data(data_file)
    if df is None:
        return
    
    # 设置序列长度和其他超参数
    time_steps = 10
    dropout_rate = 0.2
    epochs = 100
    batch_size = 32
    
    # 存储结果的字典
    results = {}
    
    # 检查目标变量是否有标准化版本
    target = 'Usage_kWh'
    target_scaler = None
    if f'{target}_scaled' in df.columns:
        # 如果有标准化列，我们需要保存scaler以便反向转换
        # 这里我们根据原始数据和标准化数据估算scaler
        target_data = df[target].values.reshape(-1, 1)
        target_scaled_data = df[f'{target}_scaled'].values.reshape(-1, 1)
        
        # 估算scaler参数（假设使用的是StandardScaler）
        scale = np.std(target_data) / np.std(target_scaled_data)
        mean = np.mean(target_data) - np.mean(target_scaled_data) * scale
        
        # 创建一个简单的scaler对象
        class SimpleScaler:
            def __init__(self, mean, scale):
                self.mean = mean
                self.scale = scale
            
            def inverse_transform(self, data):
                return data * self.scale + self.mean
        
        target_scaler = SimpleScaler(mean, scale)
        print(f"创建目标变量反标准化器，均值: {mean}, 缩放比例: {scale}")
    
    # 1. 使用原始特征数据
    print("\n--- 使用原始特征数据 ---")
    X_train, X_test, y_train, y_test, features, target_col = preprocess_data(df, use_scaled=False)
    
    # 创建序列数据
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)
    
    # 构建和训练模型
    print("\n构建CNN-LSTM模型 (原始特征)")
    model_raw = build_cnn_lstm_model((time_steps, X_train.shape[1]), dropout_rate)
    
    # 设置早停和模型保存回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(f"{model_dir}/cnn_lstm_raw.h5", save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
    
    print("\n开始训练模型 (原始特征)")
    start_time = time.time()
    history_raw = model_raw.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"训练完成，用时: {training_time:.2f} 秒")
    
    # 评估模型
    print("\n评估模型 (原始特征)")
    results['raw'] = evaluate_model(model_raw, X_test_seq, y_test_seq, target_scaler)
    results['raw']['features'] = features
    results['raw']['training_time'] = training_time
    results['raw']['history'] = history_raw.history
    
    # 绘制结果
    plot_results(
        results['raw']['y_test'], 
        results['raw']['y_pred'], 
        '原始特征数据预测结果', 
        output_dir,
        results['raw']['individual_accuracies']
    )
    
    # 绘制训练历史
    plt.figure(figsize=(10, 6))
    plt.plot(history_raw.history['loss'], label='训练损失')
    plt.plot(history_raw.history['val_loss'], label='验证损失')
    plt.title('原始特征数据 - 训练历史', fontsize=14)
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/raw_features_training_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 使用标准化特征数据
    print("\n--- 使用标准化特征数据 ---")
    X_train, X_test, y_train, y_test, scaled_features, target_col = preprocess_data(df, use_scaled=True)
    
    # 创建序列数据
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)
    
    # 构建和训练模型
    print("\n构建CNN-LSTM模型 (标准化特征)")
    model_scaled = build_cnn_lstm_model((time_steps, X_train.shape[1]), dropout_rate)
    
    # 设置早停和模型保存回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    model_checkpoint = ModelCheckpoint(f"{model_dir}/cnn_lstm_scaled.h5", save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
    
    print("\n开始训练模型 (标准化特征)")
    start_time = time.time()
    history_scaled = model_scaled.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"训练完成，用时: {training_time:.2f} 秒")
    
    # 评估模型
    print("\n评估模型 (标准化特征)")
    results['scaled'] = evaluate_model(model_scaled, X_test_seq, y_test_seq, target_scaler)
    results['scaled']['features'] = scaled_features
    results['scaled']['training_time'] = training_time
    results['scaled']['history'] = history_scaled.history
    
    # 绘制结果
    plot_results(
        results['scaled']['y_test'], 
        results['scaled']['y_pred'], 
        '标准化特征数据预测结果', 
        output_dir,
        results['scaled']['individual_accuracies']
    )
    
    # 绘制训练历史
    plt.figure(figsize=(10, 6))
    plt.plot(history_scaled.history['loss'], label='训练损失')
    plt.plot(history_scaled.history['val_loss'], label='验证损失')
    plt.title('标准化特征数据 - 训练历史', fontsize=14)
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scaled_features_training_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建详细的结果报告
    report = f"""
# CNN-LSTM 模型预测结果报告
生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

## 数据集信息
- 总样本数: {len(df)}
- 训练集样本数: {int(len(df) * 0.75)}
- 测试集样本数: {int(len(df) * 0.25)}
- 特征数量: {len(features)}
- 序列长度: {time_steps}

## 使用的特征:
- 原始特征: {', '.join(results['raw']['features'])}
- 标准化特征: {', '.join(results['scaled']['features'])}

## 模型架构:
- 卷积层: 2个 (64和32个滤波器)
- LSTM层: 2个 (50和30个单元)
- Dropout率: {dropout_rate}

## 训练参数:
- Epochs: {epochs}
- 批量大小: {batch_size}
- 优化器: Adam (学习率=0.001)
- 损失函数: MSE

## 性能对比:

| 指标 | 原始特征 | 标准化特征 |
|------|----------|------------|
| MSE | {results['raw']['mse']:.4f} | {results['scaled']['mse']:.4f} |
| RMSE | {results['raw']['rmse']:.4f} | {results['scaled']['rmse']:.4f} |
| MAE | {results['raw']['mae']:.4f} | {results['scaled']['mae']:.4f} |
| R² | {results['raw']['r2']:.4f} | {results['scaled']['r2']:.4f} |
| MAPE | {results['raw']['mape']:.2f}% | {results['scaled']['mape']:.2f}% |
| 准确率 | {results['raw']['accuracy']:.2f}% | {results['scaled']['accuracy']:.2f}% |
| 训练时间 | {results['raw']['training_time']:.2f}秒 | {results['scaled']['training_time']:.2f}秒 |
| 预测时间 | {results['raw']['prediction_time']:.4f}秒 | {results['scaled']['prediction_time']:.4f}秒 |

## 结论:
- {'标准化特征' if results['scaled']['accuracy'] > results['raw']['accuracy'] else '原始特征'}模型表现更好，准确率提高了{abs(results['scaled']['accuracy'] - results['raw']['accuracy']):.2f}%
- 模型能够以{max(results['raw']['accuracy'], results['scaled']['accuracy']):.2f}%的准确度预测电力使用量
    """
    
    # 保存报告
    with open(f"{output_dir}/model_performance_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n性能报告已保存到: {output_dir}/model_performance_report.md")
    
    # 比较两个模型的性能
    performance_comparison = pd.DataFrame({
        '原始特征': [
            results['raw']['mse'],
            results['raw']['rmse'],
            results['raw']['mae'],
            results['raw']['r2'],
            results['raw']['mape'],
            results['raw']['accuracy'],
            results['raw']['training_time'],
            results['raw']['prediction_time']
        ],
        '标准化特征': [
            results['scaled']['mse'],
            results['scaled']['rmse'],
            results['scaled']['mae'],
            results['scaled']['r2'],
            results['scaled']['mape'],
            results['scaled']['accuracy'],
            results['scaled']['training_time'],
            results['scaled']['prediction_time']
        ]
    }, index=['MSE', 'RMSE', 'MAE', 'R²', 'MAPE (%)', '准确率 (%)', '训练时间 (秒)', '预测时间 (秒)'])
    
    # 保存性能比较表
    performance_comparison.to_csv(f"{output_dir}/performance_comparison.csv", encoding="utf-8")
    
    # 绘制性能对比条形图
    plt.figure(figsize=(12, 8))
    performance_comparison.loc[['MSE', 'RMSE', 'MAE']].plot(kind='bar', rot=0)
    plt.title('CNN-LSTM模型误差指标对比', fontsize=14)
    plt.ylabel('误差值', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制准确率和R²对比图
    plt.figure(figsize=(10, 6))
    performance_comparison.loc[['准确率 (%)', 'R²']].plot(kind='bar', rot=0)
    plt.title('CNN-LSTM模型准确率和R²对比', fontsize=14)
    plt.ylabel('值', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_r2_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制预测精度变化对比图
    plt.figure(figsize=(14, 7))
    plt.plot(results['raw']['individual_accuracies'], label='原始特征', alpha=0.7, linewidth=1.5)
    plt.plot(results['scaled']['individual_accuracies'], label='标准化特征', alpha=0.7, linewidth=1.5)
    plt.axhline(y=results['raw']['accuracy'], color='blue', linestyle='--', 
               alpha=0.7, label=f'原始特征平均: {results["raw"]["accuracy"]:.2f}%')
    plt.axhline(y=results['scaled']['accuracy'], color='orange', linestyle='--', 
               alpha=0.7, label=f'标准化特征平均: {results["scaled"]["accuracy"]:.2f}%')
    plt.title('预测精度变化曲线对比', fontsize=14)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('精度 (%)', fontsize=12)
    plt.ylim(0, 110)
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_comparison_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制精度分布对比
    plt.figure(figsize=(12, 6))
    sns.kdeplot(results['raw']['individual_accuracies'], label='原始特征', fill=True, alpha=0.4)
    sns.kdeplot(results['scaled']['individual_accuracies'], label='标准化特征', fill=True, alpha=0.4)
    plt.title('预测精度分布对比', fontsize=14)
    plt.xlabel('精度 (%)', fontsize=12)
    plt.ylabel('密度', fontsize=12)
    plt.gca().xaxis.set_major_formatter(PercentFormatter())
    plt.axvline(x=results['raw']['accuracy'], color='blue', linestyle='--', 
               alpha=0.7, label=f'原始特征平均: {results["raw"]["accuracy"]:.2f}%')
    plt.axvline(x=results['scaled']['accuracy'], color='orange', linestyle='--', 
               alpha=0.7, label=f'标准化特征平均: {results["scaled"]["accuracy"]:.2f}%')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建一个单一样本的精度随时间变化图
    num_samples = min(len(results['raw']['individual_accuracies']), len(results['scaled']['individual_accuracies']))
    step_size = max(1, num_samples // 10)  # 每10个点选一个作为时间点
    
    # 选择一些关键时间点
    selected_indices = list(range(0, num_samples, step_size))
    if num_samples-1 not in selected_indices:
        selected_indices.append(num_samples-1)  # 添加最后一个点
        
    raw_selected = [results['raw']['individual_accuracies'][i] for i in selected_indices]
    scaled_selected = [results['scaled']['individual_accuracies'][i] for i in selected_indices]
    
    plt.figure(figsize=(12, 8))
    
    x = np.array(selected_indices)
    width = 0.35
    
    plt.bar(x - width/2, raw_selected, width, label='原始特征', alpha=0.7)
    plt.bar(x + width/2, scaled_selected, width, label='标准化特征', alpha=0.7)
    
    plt.title('关键时间点预测精度对比', fontsize=14)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('预测精度 (%)', fontsize=12)
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.xticks(x)
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/timepoint_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制特征重要性分析 (此处使用模型权重的绝对值作为简单估计)
    # 注意：这只是一个简化的方法，实际上CNN-LSTM的特征重要性分析更复杂
    
    # 从第一个卷积层提取权重
    conv_weights_raw = model_raw.layers[0].get_weights()[0]
    conv_weights_scaled = model_scaled.layers[0].get_weights()[0]
    
    # 计算每个特征的权重绝对值之和
    feature_importance_raw = np.sum(np.abs(conv_weights_raw), axis=(0, 2))
    feature_importance_scaled = np.sum(np.abs(conv_weights_scaled), axis=(0, 2))
    
    # 归一化重要性分数
    feature_importance_raw = 100 * feature_importance_raw / np.sum(feature_importance_raw)
    feature_importance_scaled = 100 * feature_importance_scaled / np.sum(feature_importance_scaled)
    
    # 特征名称 (去掉可能的 '_scaled' 后缀)
    feature_names = [f.replace('_scaled', '') for f in results['raw']['features']]
    
    # 绘制特征重要性对比图
    plt.figure(figsize=(10, 6))
    x = np.arange(len(feature_names))
    width = 0.35
    
    plt.bar(x - width/2, feature_importance_raw, width, label='原始特征模型', alpha=0.7)
    plt.bar(x + width/2, feature_importance_scaled, width, label='标准化特征模型', alpha=0.7)
    
    plt.title('特征重要性估计对比', fontsize=14)
    plt.xlabel('特征', fontsize=12)
    plt.ylabel('相对重要性 (%)', fontsize=12)
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.xticks(x, feature_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建预测结果的对比表格
    prediction_comparison = pd.DataFrame({
        '实际值': results['raw']['y_test'],
        '原始特征预测': results['raw']['y_pred'],
        '原始特征误差': results['raw']['y_test'] - results['raw']['y_pred'],
        '原始特征精度(%)': results['raw']['individual_accuracies'],
        '标准化特征预测': results['scaled']['y_pred'],
        '标准化特征误差': results['scaled']['y_test'] - results['scaled']['y_pred'],
        '标准化特征精度(%)': results['scaled']['individual_accuracies']
    })
    
    # 保存完整预测结果表
    prediction_comparison.to_csv(f"{output_dir}/prediction_comparison.csv", encoding="utf-8")
    
    # 保存预测表格的前20行和后20行（作为摘要）
    prediction_sample = pd.concat([prediction_comparison.head(20), prediction_comparison.tail(20)])
    prediction_sample.to_csv(f"{output_dir}/prediction_sample.csv", encoding="utf-8")
    
    # 计算每个时间段的平均预测精度
    # 将测试集分为5个时间段
    segments = 5
    segment_size = len(results['raw']['individual_accuracies']) // segments
    
    segment_accuracies = {
        '原始特征': [],
        '标准化特征': []
    }
    
    for i in range(segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < segments - 1 else len(results['raw']['individual_accuracies'])
        
        raw_segment_acc = np.mean(results['raw']['individual_accuracies'][start_idx:end_idx])
        scaled_segment_acc = np.mean(results['scaled']['individual_accuracies'][start_idx:end_idx])
        
        segment_accuracies['原始特征'].append(raw_segment_acc)
        segment_accuracies['标准化特征'].append(scaled_segment_acc)
    
    # 绘制时间段平均精度
    plt.figure(figsize=(10, 6))
    x = np.arange(segments)
    width = 0.35
    
    plt.bar(x - width/2, segment_accuracies['原始特征'], width, label='原始特征', alpha=0.7)
    plt.bar(x + width/2, segment_accuracies['标准化特征'], width, label='标准化特征', alpha=0.7)
    
    plt.title('不同时间段预测精度对比', fontsize=14)
    plt.xlabel('时间段', fontsize=12)
    plt.ylabel('平均预测精度 (%)', fontsize=12)
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.xticks(x, [f'第{i+1}段' for i in range(segments)])
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_segment_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算每个特征的相关性
    original_features = ['Lagging_Current_Reactive.Power_kVarh', 'CO2(tCO2)', 'Lagging_Current_Power_Factor']
    target = 'Usage_kWh'
    
    correlation_df = df[original_features + [target]].corr()
    
    # 绘制相关性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('特征与目标变量相关性', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印最终结论
    print("\n" + "="*80)
    print("CNN-LSTM组合预测模型训练与评估完成!")
    print(f"最佳模型: {'标准化特征模型' if results['scaled']['accuracy'] > results['raw']['accuracy'] else '原始特征模型'}")
    print(f"最佳准确率: {max(results['raw']['accuracy'], results['scaled']['accuracy']):.2f}%")
    print(f"模型已保存至: {model_dir}")
    print(f"结果图表已保存至: {output_dir}")
    print("="*80)
    
    # 返回结果字典，方便后续分析或使用
    return results

if __name__ == "__main__":
    main()