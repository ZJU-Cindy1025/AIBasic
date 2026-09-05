import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy.spatial import distance
# 读取图像
image = ski.data.coins()
# 转换训练标签
train_label = ski.io.imread('RsEx/coins_MIN_HEX/label.png')
color_map = {}
# 生成每种颜色的对应标签
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

# 最小距离分类法


def min_distance_classifier(train_data, train_label):
    # 计算训练数据的类别中心
    unique_labels = np.unique(train_label)
    # 计算类别中心的x,y坐标
    class_centers = np.zeros((len(unique_labels), 2))
    for i in range(len(unique_labels)):
        if (unique_labels[i] == 0):
            continue
        x, y = np.where(train_label == unique_labels[i])
        class_centers[i] = np.mean(x), np.mean(y)
    # 计算每个像素点到类别中心的距离
    result_label = np.zeros(train_label.shape)
    for i in range(train_data.shape[0]):
        for j in range(train_data.shape[1]):
            distances = np.zeros(len(unique_labels))
            for k in range(len(unique_labels)):
                distances[k] = distance.euclidean([i, j], class_centers[k])
            # 取最小距离为分类结果
            result_label[i, j] = unique_labels[np.argmin(distances)]
    return result_label


# 训练数据
result_label = min_distance_classifier(image, train_label)

# 显示分类结果
fig = make_subplots(rows=2, cols=2, subplot_titles=[
                    '原图', '训练标签', '真实标签', '最小距离分类标签'])
for row, col, image_data in [(1, 1, image), (1, 2, train_label), (2, 1, true_label), (2, 2, result_label)]:
    fig.add_trace(
        px.imshow(image_data).data[0], row=row, col=col)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()
