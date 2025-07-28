import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取图像
img1 = ski.data.grass()
img2 = ski.data.gravel()

# 除运算
img3 = img1 / img2
# 将img3化为0-255
img3 = img3 * 255
img3 = img3.astype(np.uint8)

# 显示图像
fig, ax = plt.subplots(1, 3)
ax[0].axis('off')
ax[0].imshow(img1, cmap='gray')
ax[0].set_title('图像1')
ax[1].axis('off')
ax[1].imshow(img2, cmap='gray')
ax[1].set_title('图像2')
ax[2].axis('off')
ax[2].imshow(img3, cmap='gray')
ax[2].set_title('除运算图像')
plt.show()
