import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取图像
img = ski.data.chelsea()

# KMeans聚类
model = KMeans(n_clusters=3)
img_data = img.reshape((-1, 3))
model.fit(img_data)

# 聚类结果
labels = model.labels_
img_kmeans = np.zeros_like(img_data)
for i in range(3):
    img_kmeans[labels == i] = model.cluster_centers_[i]
img_kmeans = img_kmeans.reshape(img.shape)

# 显示结果
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[0].axis('off')
ax[0].set_title('原图像')
ax[1].imshow(img_kmeans)
ax[1].axis('off')
ax[1].set_title('KMeans聚类后图像')
plt.show()
