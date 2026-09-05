import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans

# 读取图像
img = ski.data.chelsea()

# KMeans聚类
model = KMeans(n_clusters=3)
img_data = img.reshape((-1, 3))
model.fit(img_data)

# 聚类结果
labels = model.labels_
img_kmeans = np.zeros_like(img_data)
for i in range(3):
    img_kmeans[labels == i] = model.cluster_centers_[i]
img_kmeans = img_kmeans.reshape(img.shape)

# 显示结果
fig = make_subplots(rows=1, cols=2, subplot_titles=['原图像', 'KMeans聚类后图像'])
fig.add_trace(px.imshow(img).data[0], row=1, col=1)
fig.add_trace(px.imshow(img_kmeans).data[0], row=1, col=2)
fig.update_yaxes(autorange='reversed')
fig.update_traces(showscale=False, selector={'type': 'heatmap'})
fig.show()
