import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取图像
img = ski.data.chelsea()

# 简单集群分类法：基于概率密度函数估计的直接方法
model = GaussianMixture(n_components=3)
img_data = img.reshape((-1, 3))
model.fit(img_data)
# 聚类结果
labels = model.predict(img_data)
img_label = np.zeros_like(img_data)
for i in range(3):
    img_label[labels == i] = model.means_[i]
img_label = img_label.reshape(img.shape)

# 显示结果
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[0].axis('off')
ax[0].set_title('原图像')
ax[1].imshow(img_label)
ax[1].axis('off')
ax[1].set_title('简单集群聚类后图像')
plt.show()
