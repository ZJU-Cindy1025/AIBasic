import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取示例图片
img1 = ski.data.astronaut()
hist1 = ski.exposure.histogram(img1)
# 图像反转
img2 = ski.util.invert(img1)
hist2 = ski.exposure.histogram(img2)
# 直方图均衡化
img3 = ski.exposure.equalize_hist(img1)
hist3 = ski.exposure.histogram(img3)
# 直方图拉伸
p, q = np.percentile(img1, (30, 70))
img4 = ski.exposure.rescale_intensity(img1, in_range=(p, q))
hist4 = ski.exposure.histogram(img4)
# 直方图规定化
img5 = ski.exposure.match_histograms(img1, ski.data.colorwheel())
hist5 = ski.exposure.histogram(img5)

# 显示结果
fig, ax = plt.subplots(2, 5, figsize=(15, 6))
ax[0, 0].imshow(img1)
ax[0, 0].axis('off')
ax[0, 0].set_title('原图')
ax[0, 1].imshow(img2)
ax[0, 1].axis('off')
ax[0, 1].set_title('反转')
ax[0, 2].imshow(img3)
ax[0, 2].axis('off')
ax[0, 2].set_title('直方图均衡化')
ax[0, 3].imshow(img4)
ax[0, 3].axis('off')
ax[0, 3].set_title('直方图拉伸')
ax[0, 4].imshow(img5)
ax[0, 4].axis('off')
ax[0, 4].set_title('直方图规定化')
ax[1, 0].bar(range(256), hist1[0])
ax[1, 0].set_title('原图直方图')
ax[1, 0].set_ylim([0, 10000])
ax[1, 1].bar(range(256), hist2[0])
ax[1, 1].set_title('反转直方图')
ax[1, 1].set_ylim([0, 10000])
ax[1, 2].bar(range(256), hist3[0])
ax[1, 2].set_title('直方图均衡化直方图')
ax[1, 2].set_ylim([0, 10000])
ax[1, 3].bar(range(256), hist4[0])
ax[1, 3].set_title('直方图拉伸直方图')
ax[1, 3].set_ylim([0, 10000])
ax[1, 4].bar(range(256), hist5[0])
ax[1, 4].set_title('直方图规定化直方图')
ax[1, 4].set_ylim([0, 10000])
plt.tight_layout()
plt.show()
