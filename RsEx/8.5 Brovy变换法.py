import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取两幅图像
img1 = ski.data.astronaut()
img2 = ski.data.chelsea()

# 将img1调整为img2大小
img1 = ski.transform.resize(img1, img2.shape)

# 将图像转换为灰度图
img1_gray = ski.color.rgb2gray(img1)
img2_gray = ski.color.rgb2gray(img2)
# 对img1进行多光谱波段颜色归一化
img1_norm = np.zeros_like(img1)
img1_norm[:, :, 0] = img1[:, :, 0] / np.sum(img1)
img1_norm[:, :, 1] = img1[:, :, 1] / np.sum(img1)
img1_norm[:, :, 2] = img1[:, :, 2] / np.sum(img1)
# 计算img2的全色影像的灰度值
img2_gray = img2_gray / 255
# 计算融合影像
img3 = np.zeros_like(img1)
img3[:, :, 0] = img2_gray * img1_norm[:, :, 0]
img3[:, :, 1] = img2_gray * img1_norm[:, :, 1]
img3[:, :, 2] = img2_gray * img1_norm[:, :, 2]
# 将图像归一化到0-255
img3 = 255*(img3 - np.min(img3)) / (np.max(img3) - np.min(img3))
img3 = img3.astype(np.uint8)

# 显示结果
fig, ax = plt.subplots(1, 3)
ax[0].imshow(img1)
ax[0].axis('off')
ax[0].set_title('图像1')
ax[1].imshow(img2)
ax[1].axis('off')
ax[1].set_title('图像2')
ax[2].imshow(img3)
ax[2].axis('off')
ax[2].set_title('Brovy后图像')
plt.show()
