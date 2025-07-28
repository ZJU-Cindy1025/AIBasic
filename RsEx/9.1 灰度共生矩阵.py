import skimage as ski
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取示例图片
img1 = ski.data.grass()
img2 = ski.data.gravel()

# 生成img1和img2的灰度共生矩阵，西北-东南向
glcm1 = ski.feature.graycomatrix(img1, [1], [1], 256)
glcm2 = ski.feature.graycomatrix(img2, [1], [1], 256)

# 显示灰度共生矩阵
fig, ax = plt.subplots(2, 2, figsize=(4, 4))
ax[0, 0].axis('off')
ax[0, 0].imshow(img1, cmap='gray')
ax[0, 0].set_title('grass')
ax[0, 1].axis('off')
ax[0, 1].imshow(img2, cmap='gray')
ax[0, 1].set_title('gravel')
ax[1, 0].axis('off')
ax[1, 0].imshow(glcm1[:, :, 0, 0], cmap='gray')
ax[1, 0].set_title('grass的灰度共生矩阵')
ax[1, 1].axis('off')
ax[1, 1].imshow(glcm2[:, :, 0, 0], cmap='gray')
ax[1, 1].set_title('gravel的灰度共生矩阵')
plt.show()

# 计算灰度共生矩阵的特征
contrast1 = ski.feature.graycoprops(glcm1, 'contrast')
correlation1 = ski.feature.graycoprops(glcm1, 'correlation')
energy1 = ski.feature.graycoprops(glcm1, 'energy')
homogeneity1 = ski.feature.graycoprops(glcm1, 'homogeneity')
dissimilarity1 = ski.feature.graycoprops(glcm1, 'dissimilarity')
contrast2 = ski.feature.graycoprops(glcm2, 'contrast')
correlation2 = ski.feature.graycoprops(glcm2, 'correlation')
energy2 = ski.feature.graycoprops(glcm2, 'energy')
homogeneity2 = ski.feature.graycoprops(glcm2, 'homogeneity')
dissimilarity2 = ski.feature.graycoprops(glcm2, 'dissimilarity')

# 显示灰度共生矩阵的特征
df = pd.DataFrame(columns=['影像', '对比度', '相关性', '能量', '一致性', '差异性'])
df.loc[0] = ['img1', contrast1[0, 0], correlation1[0, 0],
             energy1[0, 0], homogeneity1[0, 0], dissimilarity1[0, 0]]
df.loc[1] = ['img2', contrast2[0, 0], correlation2[0, 0],
             energy2[0, 0], homogeneity2[0, 0], dissimilarity2[0, 0]]
print(df)
