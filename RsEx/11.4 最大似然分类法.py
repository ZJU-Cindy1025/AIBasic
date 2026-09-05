import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
# 读取图像
image = ski.data.coins()
# 转换训练标签
train_label = ski.io.imread('RsEx/coins_KNN_MLE/label.png')
train_label = train_label[:, :, 0]
# 转换真实标签
true_label = ski.io.imread('RsEx/coins_label/label.png')
true_label = true_label[:, :, 0]

# 最大似然分类法
clf = GaussianNB()
clf.fit(image.reshape(-1, 1), train_label.reshape(-1))

# 应用训练模型
result_label = clf.predict(image.reshape(-1, 1)).reshape(image.shape)

# 显示分类结果
fig = make_subplots(rows=2, cols=2, subplot_titles=[
                    '原图', '训练标签', '真实标签', '最大似然分类标签'])
for row, col, image_data in [(1, 1, image), (1, 2, train_label), (2, 1, true_label), (2, 2, result_label)]:
    fig.add_trace(
        px.imshow(image_data).data[0], row=row, col=col)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()
ski.io.imsave('RsEx/coins_result/coins_MLE_result.png', result_label)

# 计算混淆矩阵
matrix = confusion_matrix(true_label.reshape(-1), result_label.reshape(-1))
print('混淆矩阵:')
print(matrix)
print('查准率:', matrix[1, 1]/(matrix[1, 1]+matrix[0, 1]))
print('查全率:', matrix[1, 1]/(matrix[1, 1]+matrix[1, 0]))
