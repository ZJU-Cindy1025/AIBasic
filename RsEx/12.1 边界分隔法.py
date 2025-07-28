import skimage as ski
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取图像
img1 = ski.data.horse()
# 一阶微分找边
edge = ski.filters.sobel(img1)

# 寻找线
labels = ski.segmentation.find_boundaries(img1, mode='thick')
labels = ski.util.img_as_ubyte(labels)

# 显示结果
fig, ax = plt.subplots(1, 3)
ax[0].imshow(img1, cmap='gray')
ax[0].axis('off')
ax[0].set_title('原图像')
ax[1].imshow(edge, cmap='gray')
ax[1].axis('off')
ax[1].set_title('一阶微分法找边')
ax[2].imshow(labels, cmap='gray')
ax[2].axis('off')
ax[2].set_title('使用典型模板找线')
plt.tight_layout()
plt.show()
