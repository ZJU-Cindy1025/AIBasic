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

# 艾勒斯图像融合
img1hsv = ski.color.rgb2hsv(img1)
img2hsv = ski.color.rgb2hsv(img2)
img3 = np.zeros_like(img1)
# 对图像1的色彩进行高通滤波处理，图像2的色彩进行低通滤波处理，相加之后再取img2的饱和度和亮度信息
img3[:, :, 0] = ski.filters.butterworth(img1hsv, 0.05, high_pass=True)[
    :, :, 0]+ski.filters.butterworth(img2hsv, 0.05, high_pass=False)[:, :, 0]
img3[:, :, 1] = img2hsv[:, :, 1]
img3[:, :, 2] = img2hsv[:, :, 2]
img3 = ski.color.hsv2rgb(img3)

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
ax[2].set_title('艾勒斯融合后图像')
plt.show()
