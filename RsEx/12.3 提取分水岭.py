import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取图像
coins = ski.data.coins()
# 提取边缘
edges = ski.filters.sobel(coins)

# 生成种子
grid = ski.util.regular_grid(coins.shape, n_points=468)
seeds = np.zeros(coins.shape, dtype=int)
seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1

# 分水岭算法分割
watershed = ski.segmentation.watershed(edges, seeds)

fig, ax = plt.subplots(1, 2)
ax[0].axis('off')
ax[0].imshow(coins, cmap='gray')
ax[0].set_title('原图')
ax[1].axis('off')
ax[1].imshow(watershed, cmap='gray')
ax[1].set_title('分水岭算法分割')

plt.show()
