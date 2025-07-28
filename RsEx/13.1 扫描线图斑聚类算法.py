import skimage as ski
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 读取训练后数据
pData = ski.io.imread(r'RsEx/coins_result/coins_KNN_result.png', as_gray=True)

# 扫描线图斑聚类算法
pDstData = ski.measure.label(pData, background=0)
# 获取图斑信息
region = ski.measure.regionprops(ski.util.img_as_ubyte(pData))
df = pd.DataFrame(columns=[prop for prop in region[0]])
for i in range(len(region)):
    for prop in region[i]:
        df.loc[i, prop] = region[i][prop]
print(df)

# 显示结果
fig, ax = plt.subplots(1, 2)
ax[0].imshow(pData, cmap='gray')
ax[0].set_title('原始分类后数据')
ax[0].axis('off')
ax[1].imshow(pDstData, cmap='gray')
ax[1].set_title('扫描线聚类算法')
ax[1].axis('off')
plt.show()
