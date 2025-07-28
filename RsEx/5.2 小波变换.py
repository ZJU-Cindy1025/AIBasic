import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import pywt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取示例图片
image = ski.data.camera()

# 设置小波系数
coeffs = pywt.dwt2(image, wavelet='haar')
cA, (cH, cV, cD) = coeffs

# 显示结果
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(image, cmap='gray')
ax[0, 0].axis('off')
ax[0, 0].set_title('原图')
ax[0, 1].imshow(cA, cmap='gray')
ax[0, 1].axis('off')
ax[0, 1].set_title('水平方向')
ax[1, 0].imshow(cV, cmap='gray')
ax[1, 0].axis('off')
ax[1, 0].set_title('垂直方向')
ax[1, 1].imshow(cD, cmap='gray')
ax[1, 1].axis('off')
ax[1, 1].set_title('对角线方向')
plt.show()
