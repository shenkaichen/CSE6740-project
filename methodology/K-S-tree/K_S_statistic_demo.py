import numpy as np
import pandas as pd


# example dataframe with continuous feature
data = {
    'feature': [0.1, 0.4, 0.35, 0.8, 0.5, 0.9, 0.7],
    'target': [0, 1, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
df = df.sort_values(by='feature')
print(df)
# use CDF to calculate K-S statistic if the feature is continuous
df['cum_positive'] = (df['target'] == 1).cumsum() / (df['target'] == 1).sum()
df['cum_negative'] = (df['target'] == 0).cumsum() / (df['target'] == 0).sum()
df['diff'] = np.abs(df['cum_positive'] - df['cum_negative'])
print(df)
ks_statistic = df['diff'].max()
print("K-S Statistic (if feature is continuous): ", ks_statistic)
print()


# example dataframe with categorical/binary feature
data = {
    'feature': [2, 2, 2, 2, 1, 1, 1, 2],
    'target': [0, 1, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
df = df.sort_values(by='feature')
print(df)
# use proportion to calculate K-S statistic if the feature is categorical/binary
category_stats = df.groupby('feature')['target'].value_counts(normalize=False).unstack(fill_value=0)
label_counts = df['target'].value_counts()
category_stats[0] = category_stats[0] / label_counts[0]
category_stats[1] = category_stats[1] / label_counts[1]
category_stats['abs_diff'] = (category_stats[0] - category_stats[1]).abs()
print(category_stats)
ks_statistic = category_stats['abs_diff'].max()
print("K-S Statistic (if feature is categorical/binary): ", ks_statistic)
print()