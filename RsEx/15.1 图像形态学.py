import skimage as ski
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取图像
img = ski.data.horse()
# 反色，更好观察结果
img = ~img

# 结构元素
selem = ski.morphology.disk(10)

# 腐蚀
eroded = ski.morphology.erosion(img, selem)

# 膨胀
dilated = ski.morphology.dilation(img, selem)

# 开运算
opened = ski.morphology.opening(img, selem)

# 闭运算
closed = ski.morphology.closing(img, selem)

# 显示结果
fig, ax = plt.subplots(1, 5, figsize=(15, 3))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('原图')
ax[0].axis('off')
ax[1].imshow(eroded, cmap='gray')
ax[1].set_title('腐蚀')
ax[1].axis('off')
ax[2].imshow(dilated, cmap='gray')
ax[2].set_title('膨胀')
ax[2].axis('off')
ax[3].imshow(opened, cmap='gray')
ax[3].set_title('开运算')
ax[3].axis('off')
ax[4].imshow(closed, cmap='gray')
ax[4].set_title('闭运算')
ax[4].axis('off')
plt.tight_layout()
plt.show()
