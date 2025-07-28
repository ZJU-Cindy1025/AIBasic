import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取图像
img1 = ski.data.chelsea()  # 彩色图像
img2 = ski.data.horse()  # 0或1

# 把img1调整为img2大小
img1 = ski.transform.resize(img1, img2.shape)
# img1每个通道分别与img2相乘
img3 = np.zeros_like(img1)
for i in range(3):
    img3[:, :, i] = img1[:, :, i] * img2

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
ax[2].set_title('乘运算图像')
plt.show()
