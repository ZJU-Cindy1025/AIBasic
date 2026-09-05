from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
# 加载数据
data = load_iris()
y = data['target']
X = data['data']
# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# PCA降维，原有特征数为4
pca = PCA()
reduced_X = pca.fit_transform(X)

# 输出降维信息
explained_variance_ratio = pca.explained_variance_ratio_
print(f"解释方差比例: {explained_variance_ratio}")
print("主成分的特征向量:")
for i, component in enumerate(pca.components_):
    print(f"PC{i+1}: {component}")

print("\n原始特征与主成分的相关性:")
for i, component in enumerate(pca.components_):
    correlations = component * np.sqrt(pca.explained_variance_[i])
    print(f"PC{i+1}:")
    for j, corr in enumerate(correlations):
        print(f"  {data.feature_names[j]}: {corr:.3f}")

# 绘制降维后的数据
reduced_data = pd.DataFrame(reduced_X, columns=['特征1', '特征2', '特征3', '特征4'])
reduced_data['target'] = y

fig = make_subplots(rows=1, cols=2, subplot_titles=['原始数据', 'PCA降维后的数据'])
for label, group in pd.DataFrame({'x': data['data'][:, 0], 'y': data['data'][:, 1], 'label': y}).groupby('label'):
    fig.add_trace(px.scatter(group, x='x', y='y',
                  title=f'original {label}').data[0], row=1, col=1)
for label, group in reduced_data.groupby('target'):
    fig.add_trace(px.scatter(group, x='特征1', y='特征2',
                  title=f'PCA {label}').data[0], row=1, col=2)
fig.update_xaxes(title_text='花萼长度', row=1, col=1)
fig.update_yaxes(title_text='花萼宽度', row=1, col=1)
fig.update_xaxes(title_text='特征1', row=1, col=2)
fig.update_yaxes(title_text='特征2', row=1, col=2)
fig.show()
