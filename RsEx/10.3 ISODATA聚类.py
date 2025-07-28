import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取图像
img = ski.data.chelsea()

# ISODATA聚类


def ISODATA(img, N, M, T, S):
    # N为初始聚类中心数，M为每个聚类中的样本数，T为最大迭代次数，S为最小样本方差
    # 初始化聚类中心
    center = np.random.randint(0, 256, (N, 3))
    # 初始化样本标签
    label = np.zeros((img.shape[0], img.shape[1]))
    # 初始化样本方差
    var = np.zeros(N)
    # 开始迭代
    for t in range(T):
        # 计算每个样本的标签
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                dist = np.zeros(N)
                for k in range(N):
                    dist[k] = np.linalg.norm(img[i, j] - center[k])
                label[i, j] = np.argmin(dist)
        # 计算每个聚类中的样本
        samples = []
        for k in range(N):
            samples.append(img[label == k])
        # 计算每个聚类中的样本方差
        for k in range(N):
            var[k] = np.var(samples[k])
        # 合并方差小于S的聚类
        if np.min(var) < S:
            center = np.delete(center, np.argmin(var), axis=0)
            N -= 1
            var = np.delete(var, np.argmin(var))
        # 计算每个聚类中的样本均值
        for k in range(N):
            center[k] = np.mean(samples[k], axis=0)
    # 生成聚类图像
    img_label = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_label[i, j] = center[int(label[i, j])]
    return img_label


# 聚类
img_label = ISODATA(img, 3, 2, 10, 10)
img_label = img_label.astype(np.uint8)
# 显示结果
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[0].axis('off')
ax[0].set_title('原图像')
ax[1].imshow(img_label)
ax[1].axis('off')
ax[1].set_title('ISODATA聚类后图像')
plt.show()
