import skimage as ski
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取图像
img1 = ski.data.horse()
img2 = ski.data.checkerboard()

img1 = ski.transform.resize(img1, img2.shape)
img1 = ski.img_as_ubyte(img1)
img2 = ski.img_as_ubyte(img2)

# 反运算
img3 = 255-img1
# 或运算
img4 = img1 | img2
# 与运算
img5 = img1 & img2
# 异或运算
img6 = img1 ^ img2

# 显示图像
fig, ax = plt.subplots(2, 3)
ax[0, 0].axis('off')
ax[0, 0].imshow(img1, cmap='gray')
ax[0, 0].set_title('图像1')
ax[0, 1].axis('off')
ax[0, 1].imshow(img2, cmap='gray')
ax[0, 1].set_title('图像2')
ax[0, 2].axis('off')
ax[0, 2].imshow(img3, cmap='gray')
ax[0, 2].set_title('反运算图像')
ax[1, 0].axis('off')
ax[1, 0].imshow(img4, cmap='gray')
ax[1, 0].set_title('或运算图像')
ax[1, 1].axis('off')
ax[1, 1].imshow(img5, cmap='gray')
ax[1, 1].set_title('与运算图像')
ax[1, 2].axis('off')
ax[1, 2].imshow(img6, cmap='gray')
ax[1, 2].set_title('异或运算图像')
plt.show()
