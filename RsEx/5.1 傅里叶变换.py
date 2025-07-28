import skimage as ski
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 导入测试图片
img = ski.data.camera()
f = np.fft.fft2(img)    # 快速傅里叶变换算法得到频率分布
fshift = np.fft.fftshift(f)    # 默认结果中心点位置是左上角，将其转移到中间位置
fimg = np.log(np.abs(fshift))  # fft结果是复数，求其绝对值之后才是振幅

# 高通滤波
imgh = ski.filters.butterworth(img, 0.05, high_pass=True)
# 低通滤波
imgl = ski.filters.butterworth(img, 0.05, high_pass=False)

# 展示结果
fig, ax = plt.subplots(2, 2)
ax[0, 0].axis('off')
ax[0, 0].imshow(img, 'gray')
ax[0, 0].set_title('原图')
ax[0, 1].axis('off')
ax[0, 1].imshow(fimg, 'gray')
ax[0, 1].set_title('频谱图')
ax[1, 0].axis('off')
ax[1, 0].imshow(imgh, 'gray')
ax[1, 0].set_title('高通滤波')
ax[1, 1].axis('off')
ax[1, 1].imshow(imgl, 'gray')
ax[1, 1].set_title('低通滤波')
plt.show()
