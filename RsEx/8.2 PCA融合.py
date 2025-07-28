import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取两幅图像
img1 = ski.data.astronaut()
img2 = ski.data.chelsea()

# 将img1调整为img2大小
img1 = ski.transform.resize(img1, img2.shape)

# 基于PCA的图像融合
pca = PCA(n_components=3)
pca.fit(img1.reshape(-1, 3))
img1_pca = pca.transform(img1.reshape(-1, 3))
img2_pca = pca.transform(img2.reshape(-1, 3))
# 保留img1的第一个主成分，img2的第二、三个主成分
img1_pca[:, 1:] = img2_pca[:, 1:]
img1_new = pca.inverse_transform(img1_pca).reshape(img2.shape)
# 将图像归一化到0-255
img1_new = 255*(img1_new - np.min(img1_new)) / \
    (np.max(img1_new) - np.min(img1_new))
img1_new = img1_new.astype(np.uint8)

# 显示结果
fig, ax = plt.subplots(1, 3)
ax[0].imshow(img1)
ax[0].axis('off')
ax[0].set_title('图像1')
ax[1].imshow(img2)
ax[1].axis('off')
ax[1].set_title('图像2')
ax[2].imshow(img1_new)
ax[2].axis('off')
ax[2].set_title('PCA融合后图像')
plt.show()
