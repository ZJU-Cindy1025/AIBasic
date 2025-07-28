from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# 读取iris数据集
data = datasets.load_iris()
# 标准化
scaler = StandardScaler()
data['data'] = scaler.fit_transform(data['data'])
# 肘方法看K值
inertia = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data['data'])
    inertia.append(kmeans.inertia_)
n_clusters = 3
# 计算聚类精度
model = KMeans(n_clusters=n_clusters)
model.fit(data['data'])

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].plot(range(1, 10), inertia)
ax[0].set_title('肘方法')
ax[0].set_xlabel('簇数')
ax[0].set_ylabel('簇内误差平方和')
sns.scatterplot(x=data['data'][:, 0], y=data['data']
                [:, 1], hue=model.labels_, data=data, ax=ax[1])
ax[1].scatter(model.cluster_centers_[:, 0],
              model.cluster_centers_[:, 1], c='red', marker='x')
ax[1].set_title('KMeans聚类')
ax[1].set_xlabel('花萼长度')
ax[1].set_ylabel('花萼宽度')
sns.scatterplot(x=data['data'][:, 0], y=data['data']
                [:, 1], hue='target', data=data, ax=ax[2])
ax[2].set_title('参考标签')
ax[2].set_xlabel('花萼长度')
ax[2].set_ylabel('花萼宽度')
plt.show()
plt.close()
