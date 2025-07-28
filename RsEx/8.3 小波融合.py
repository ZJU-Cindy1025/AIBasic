import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import pywt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取两幅图像
img1 = ski.data.astronaut()
img2 = ski.data.chelsea()

# 将img1调整为img2大小
img1 = ski.transform.resize(img1, img2.shape)

# 基于小波的图像融合
coeffs1 = pywt.dwt2(img1, wavelet='haar')
coeffs2 = pywt.dwt2(img2, wavelet='haar')
cA1, (cH1, cV1, cD1) = coeffs1
cA2, (cH2, cV2, cD2) = coeffs2
# 将两幅图像的小波系数按照一定的权重相加
cA_new = (cA1 + cA2) / 2
cH_new = (cH1 + cH2) / 2
cV_new = (cV1 + cV2) / 2
cD_new = (cD1 + cD2) / 2
coeffs_new = cA_new, (cH_new, cV_new, cD_new)
img1_new = pywt.idwt2(coeffs_new, wavelet='haar')
img1_new = img1_new.astype(np.uint8)

# 显示结果
fig, ax = plt.subplots(1, 3)
ax[0].imshow(img1)
ax[0].axis('off')
ax[0].set_title('图像1')
ax[1].imshow(img2)
ax[1].axis('off')
ax[1].set_title('图像2')
ax[2].imshow(img1_new[:, :, :3])
ax[2].axis('off')
ax[2].set_title('小波融合后图像')
plt.show()
