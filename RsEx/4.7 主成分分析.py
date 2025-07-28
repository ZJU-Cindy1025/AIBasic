import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取示例图片
image = ski.data.astronaut()
image_2d = image.reshape(-1, 3)
# 创建PCA对象
pca = PCA(n_components=1)

# 训练PCA
reduced_image = pca.fit_transform(image_2d)

# 重构图像
reconstructed_image = pca.inverse_transform(reduced_image)
reconstructed_image = reconstructed_image.reshape(image.shape)
# 转为uint8类型
reconstructed_image = np.uint8(reconstructed_image)

# 显示原始和变换后的图片
fig, ax = plt.subplots(1, 2)
ax[0].axis('off')
ax[0].imshow(image)
ax[0].set_title('原图')
# 可视化PCA变换后的图片
ax[1].axis('off')
ax[1].imshow(reconstructed_image, cmap='gray')
ax[1].set_title('第一主成分假彩色影像')
plt.show()
