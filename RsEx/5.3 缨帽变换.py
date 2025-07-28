import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取示例图片
image = ski.data.camera()

# K-T变换


def kt_transform(image, k, t):
    rows, cols = image.shape
    # 生成K-T矩阵
    kt_matrix = np.zeros((rows, cols), dtype=complex)
    for i in range(rows):
        for j in range(cols):
            kt_matrix[i, j] = np.exp(1j * (k * i + t * j))
    # K-T变换
    kt_image = np.fft.fft2(image) * kt_matrix
    kt_image = np.fft.ifft2(kt_image)
    kt_image = np.abs(kt_image)
    return kt_image


# 显示结果
fig, ax = plt.subplots(1, 3)
ax[0].imshow(image, cmap='gray')
ax[0].axis('off')
ax[0].set_title('原图')
ax[1].imshow(kt_transform(image, 0.1, 0.1), cmap='gray')
ax[1].axis('off')
ax[1].set_title('K=0.1, T=0.1')
ax[2].imshow(kt_transform(image, 0.1, 0.5), cmap='gray')
ax[2].axis('off')
ax[2].set_title('K=0.1, T=0.5')
plt.show()
