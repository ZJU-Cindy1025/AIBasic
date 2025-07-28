import skimage as ski
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取示例图片
img1 = ski.data.camera()

# 拉普拉斯锐化
img2 = ski.filters.laplace(img1)

# Roberts算子锐化
img3 = ski.filters.roberts(img1)

# Sobel算子锐化
img4 = ski.filters.sobel(img1)

# unsharp_mask锐化
img5 = ski.filters.unsharp_mask(img1, radius=1, amount=10)

# 显示结果
fig, ax = plt.subplots(1, 5, figsize=(15, 3))
ax[0].axis('off')
ax[0].imshow(img1, cmap='gray')
ax[0].set_title('原图')
ax[1].axis('off')
ax[1].imshow(img2, cmap='gray')
ax[1].set_title('拉普拉斯锐化')
ax[2].axis('off')
ax[2].imshow(img3, cmap='gray')
ax[2].set_title('Roberts算子锐化')
ax[3].axis('off')
ax[3].imshow(img4, cmap='gray')
ax[3].set_title('Sobel算子锐化')
ax[4].axis('off')
ax[4].imshow(img5, cmap='gray')
ax[4].set_title('unsharp_mask锐化')
plt.tight_layout()
plt.show()
