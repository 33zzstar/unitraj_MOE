import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

def plot_brier_fde(csv_file, output_dir=None):
    """
    读取CSV文件并绘制brier_fde随训练步数的变化曲线
    
    参数:
        csv_file: CSV文件路径
        output_dir: 输出目录，默认为CSV文件所在目录
    """
    # 设置绘图样式
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 检查是否包含brier_fde列
    if 'brier_fde' not in df.columns:
        # 尝试查找包含brier_fde的列
        brier_cols = [col for col in df.columns if 'brier' in col.lower() and 'fde' in col.lower()]
        if not brier_cols:
            raise ValueError(f"未找到brier_fde相关列。可用列: {df.columns.tolist()}")
        brier_col = brier_cols[0]
        print(f"使用列: {brier_col} 替代brier_fde")
    else:
        brier_col = 'brier_fde'
    
    # 确保数据是数值型
    df[brier_col] = pd.to_numeric(df[brier_col], errors='coerce')
    
    # 创建训练步数列(如果不存在)
    if 'step' not in df.columns:
        df['step'] = np.arange(len(df))
    
    # 绘制曲线
    plt.plot(df['step'], df[brier_col], marker='o', linestyle='-', markersize=4, alpha=0.7, 
             color='#1f77b4', label=brier_col)
    
    # 添加滑动平均线(如果数据点超过10个)
    if len(df) > 10:
        window_size = min(len(df) // 5, 20)  # 自适应窗口大小
        df['smooth'] = df[brier_col].rolling(window=window_size, center=True).mean()
        plt.plot(df['step'], df['smooth'], color='#ff7f0e', linewidth=2.5, 
                 label=f'滑动平均 (窗口={window_size})')
    
    # 添加标题和标签
    plt.title(f'Training Progress - {brier_col}', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel(brier_col, fontsize=14)
    plt.legend(fontsize=12)
    
    # 网格线
    plt.grid(True, alpha=0.3)
    
    # 添加数据点标注(每n个点)
    n = max(len(df) // 10, 1)
    for i in range(0, len(df), n):
        plt.annotate(f'{df[brier_col].iloc[i]:.4f}', 
                     (df['step'].iloc[i], df[brier_col].iloc[i]),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     fontsize=8)
    
    # 确定输出路径
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取实验名称
    exp_name = os.path.basename(csv_file).replace('.csv', '')
    
    # 保存SVG图
    output_path = os.path.join(output_dir, f"{exp_name}_brier_fde_plot.svg")
    plt.tight_layout()
    plt.savefig(output_path, format='svg')
    print(f"图表已保存到: {output_path}")
    
    # 同时保存PNG图以便快速预览
    plt.savefig(output_path.replace('.svg', '.png'), dpi=300)
    
    # 显示图表
    plt.close()
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='绘制brier_fde随训练步数的变化曲线')
    parser.add_argument('--csv_file', type=str, default='', help='CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='/home/zzs/zzs/unitraj__MOE_logs_plt/', help='输出目录，默认为CSV文件所在目录')
    
    args = parser.parse_args()
    
    plot_brier_fde(args.csv_file, args.output_dir)