import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取示例图片
img1 = ski.data.camera()

# 生成高斯噪声
noise = np.random.normal(0, 50, img1.shape)

# 添加高斯噪声
img2 = img1 + noise
img2 = np.clip(img2, 0, 255).astype(np.uint8)
# 均值滤波
img3 = ski.filters.rank.mean(img2, np.ones((3, 3)), mask=None)
# 中值滤波
img4 = ski.filters.rank.median(img2, np.ones((3, 3)), mask=None)
# 最小值滤波
img5 = ski.filters.rank.minimum(img2, np.ones((3, 3)), mask=None)
# 最大值滤波
img6 = ski.filters.rank.maximum(img2, np.ones((3, 3)), mask=None)

# 显示结果
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax[0, 0].axis('off')
ax[0, 0].imshow(img1, cmap='gray')
ax[0, 0].set_title('原始图像')
ax[0, 1].axis('off')
ax[0, 1].imshow(img2, cmap='gray')
ax[0, 1].set_title('添加高斯噪声')
ax[0, 2].axis('off')
ax[0, 2].imshow(img3, cmap='gray')
ax[0, 2].set_title('均值滤波')
ax[1, 0].axis('off')
ax[1, 0].imshow(img4, cmap='gray')
ax[1, 0].set_title('中值滤波')
ax[1, 1].axis('off')
ax[1, 1].imshow(img5, cmap='gray')
ax[1, 1].set_title('最小值滤波')
ax[1, 2].axis('off')
ax[1, 2].imshow(img6, cmap='gray')
ax[1, 2].set_title('最大值滤波')
plt.show()
