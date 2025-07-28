import skimage as ski
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取图像
img1 = ski.data.camera()
img2 = ski.data.brick()

# 加运算
img3 = img1*0.5 + img2*0.5

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
ax[2].set_title('加运算图像')
plt.show()
