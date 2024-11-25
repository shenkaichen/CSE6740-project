import numpy as np
import pandas as pd

# 示例数据
data = {
    'feature': [0.1, 0.4, 0.35, 0.8, 0.5, 0.9, 0.7],
    'target': [0, 1, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

print(df)
print()

# 步骤 1: 按特征值排序
df = df.sort_values(by='feature')

print(df)
print()

# 步骤 2: 计算正类和负类累积分布函数（CDF）
df['cum_positive'] = (df['target'] == 1).cumsum() / (df['target'] == 1).sum()  # 正类累积占比
df['cum_negative'] = (df['target'] == 0).cumsum() / (df['target'] == 0).sum()  # 负类累积占比

# 步骤 3: 计算 K-S 统计量
df['diff'] = np.abs(df['cum_positive'] - df['cum_negative'])
ks_statistic = df['diff'].max()

# 打印结果
print(df)
print()
print("K-S Statistic:", ks_statistic)