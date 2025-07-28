from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
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

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
sns.scatterplot(x=data['data'][:, 0], y=data['data']
                [:, 1], hue='target', data=data, ax=ax[0])
ax[0].set_title('原始数据')
ax[0].set_xlabel('花萼长度')
ax[0].set_ylabel('花萼宽度')
sns.scatterplot(x='特征1', y='特征2', hue='target', data=reduced_data, ax=ax[1])
ax[1].set_title('PCA降维后的数据')
ax[1].set_xlabel('特征1')
ax[1].set_ylabel('特征2')
plt.show()
