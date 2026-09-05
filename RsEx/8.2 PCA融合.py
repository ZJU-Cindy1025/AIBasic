import numpy as np
import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
# 读取两幅图像
img1 = ski.data.astronaut()
img2 = ski.data.chelsea()

# 将img1调整为img2大小
img1 = ski.transform.resize(img1, img2.shape)

# 基于PCA的图像融合
pca = PCA(n_components=3)
pca.fit(img1.reshape(-1, 3))
img1_pca = pca.transform(img1.reshape(-1, 3))
img2_pca = pca.transform(img2.reshape(-1, 3))
# 保留img1的第一个主成分，img2的第二、三个主成分
img1_pca[:, 1:] = img2_pca[:, 1:]
img1_new = pca.inverse_transform(img1_pca).reshape(img2.shape)
# 将图像归一化到0-255
img1_new = 255*(img1_new - np.min(img1_new)) / \
    (np.max(img1_new) - np.min(img1_new))
img1_new = img1_new.astype(np.uint8)

# 显示结果
fig = make_subplots(rows=1, cols=3, subplot_titles=['图像1', '图像2', 'PCA融合后图像'])
for column, image_data in enumerate([img1, img2, img1_new], 1):
    fig.add_trace(px.imshow(image_data).data[0], row=1, col=column)
fig.update_yaxes(autorange='reversed')
fig.update_traces(showscale=False, selector={'type': 'heatmap'})
fig.show()
