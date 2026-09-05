import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from sklearn.mixture import GaussianMixture

# 读取图像
img = ski.data.chelsea()

# 简单集群分类法：基于概率密度函数估计的直接方法
model = GaussianMixture(n_components=3)
img_data = img.reshape((-1, 3))
model.fit(img_data)
# 聚类结果
labels = model.predict(img_data)
img_label = np.zeros_like(img_data)
for i in range(3):
    img_label[labels == i] = model.means_[i]
img_label = img_label.reshape(img.shape)

# 显示结果
fig = make_subplots(rows=1, cols=2, subplot_titles=['原图像', '简单集群聚类后图像'])
fig.add_trace(px.imshow(img).data[0], row=1, col=1)
fig.add_trace(px.imshow(img_label).data[0], row=1, col=2)
fig.update_yaxes(autorange='reversed')
fig.update_traces(showscale=False, selector={'type': 'heatmap'})
fig.show()
