import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

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
fig = make_subplots(rows=1, cols=2, subplot_titles=['原图像', 'ISODATA聚类后图像'])
fig.add_trace(px.imshow(img).data[0], row=1, col=1)
fig.add_trace(px.imshow(img_label).data[0], row=1, col=2)
fig.update_yaxes(autorange='reversed')
fig.update_traces(showscale=False, selector={'type': 'heatmap'})
fig.show()
