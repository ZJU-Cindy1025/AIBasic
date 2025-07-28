import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取图像
img1 = ski.data.astronaut()
# 平移
img2 = np.roll(img1, 100, axis=0)
# 旋转
img3 = np.rot90(img1)
# 缩放
img4 = ski.transform.resize(img1, (img1.shape[0]//2, img1.shape[1]//2))
# 翻转
img5 = np.flipud(img1)
# 镜像
img6 = np.fliplr(img1)

# 显示图像
fig, ax = plt.subplots(2, 3)
ax[0, 0].axis('off')
ax[0, 0].imshow(img1, cmap='gray')
ax[0, 0].set_title('原图')
ax[0, 1].axis('off')
ax[0, 1].imshow(img2, cmap='gray')
ax[0, 1].set_title('平移')
ax[0, 2].axis('off')
ax[0, 2].imshow(img3, cmap='gray')
ax[0, 2].set_title('旋转')
ax[1, 0].axis('off')
ax[1, 0].imshow(img4, cmap='gray')
ax[1, 0].set_title('缩放')
ax[1, 1].axis('off')
ax[1, 1].imshow(img5, cmap='gray')
ax[1, 1].set_title('翻转')
ax[1, 2].axis('off')
ax[1, 2].imshow(img6, cmap='gray')
ax[1, 2].set_title('镜像')
plt.show()
