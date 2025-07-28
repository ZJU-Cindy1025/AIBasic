import skimage as ski
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取图像
img1 = ski.data.camera()

# 通过直方图找阈值
threhold1 = ski.filters.threshold_otsu(img1)
img2 = img1 > threhold1

# 基本全局阈值法
threhold2 = ski.filters.threshold_local(img1, method='mean')
img3 = img1 > threhold2

# P参数法
threhold3 = ski.filters.threshold_multiotsu(img1)
img4 = img1.copy()
img4[img1 < threhold3[0]] = 0
img4[(img1 >= threhold3[0]) & (img1 < threhold3[1])] = 128
img4[(img1 >= threhold3[1])] = 255

# 最优准则法
threhold4 = ski.filters.threshold_yen(img1)
img5 = img1 > threhold4

# 显示结果
fig, ax = plt.subplots(1, 5, figsize=(15, 3))
ax[0].imshow(img1, cmap='gray')
ax[0].axis('off')
ax[0].set_title('原图像')
ax[1].imshow(img2, cmap='gray')
ax[1].axis('off')
ax[1].set_title('直方图法')
ax[2].imshow(img3, cmap='gray')
ax[2].axis('off')
ax[2].set_title('基本全局阈值法')
ax[3].imshow(img4, cmap='gray')
ax[3].axis('off')
ax[3].set_title('P参数法')
ax[4].imshow(img5, cmap='gray')
ax[4].axis('off')
ax[4].set_title('最优准则法')
plt.tight_layout()
plt.show()
