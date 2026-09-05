import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
# 读取图像
image = ski.data.coins()
# 转换训练标签
train_label = ski.io.imread('RsEx/coins_MIN_HEX/label.png')
# 生成每种颜色的对应标签
color_map = {}
k = 0
for i in range(train_label.shape[0]):
    for j in range(train_label.shape[1]):
        if (tuple(train_label[i, j, :]) not in color_map):
            color_map[tuple(train_label[i, j, :])] = k
            k += 1
# 对每种颜色进行标签映射
train_label_copy = np.zeros((train_label.shape[0], train_label.shape[1]))
for i in range(train_label.shape[0]):
    for j in range(train_label.shape[1]):
        train_label_copy[i, j] = color_map[tuple(train_label[i, j, :])]
train_label = train_label_copy.copy()
# 转换真实标签
true_label = ski.io.imread('RsEx/coins_label/label.png')
true_label = true_label[:, :, 0]

# 平行六面体分类法


def parallelepiped_classifier(train_data, train_label):
    # 求取train_label的唯一值
    unique_labels = np.unique(train_label)
    # 对每一个unique_label求取对应的平行六面体
    parallelepipeds = []
    for label in unique_labels:
        if (label == 0):
            continue
        # 求取label为该值的最大、最小x,y坐标
        x_min, y_min = np.min(np.where(train_label == label), axis=1)
        x_max, y_max = np.max(np.where(train_label == label), axis=1)
        parallelepipeds.append([x_min, x_max, y_min, y_max])
    # 对每一个像素点进行分类
    result_label = np.zeros(train_label.shape)
    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            for k in range(len(parallelepipeds)):
                x_min, x_max, y_min, y_max = parallelepipeds[k]
                # 在哪个六面体内就属于哪个
                if (i >= x_min and i <= x_max and j >= y_min and j <= y_max):
                    result_label[i, j] = unique_labels[k+1]
    return result_label


# 训练数据
result_label = parallelepiped_classifier(image, train_label)

# 显示分类结果
fig = make_subplots(rows=2, cols=2, subplot_titles=[
                    '原图', '训练标签', '真实标签', '平行六面体分类标签'])
for row, col, image_data in [(1, 1, image), (1, 2, train_label), (2, 1, true_label), (2, 2, result_label)]:
    fig.add_trace(
        px.imshow(image_data).data[0], row=row, col=col)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()
