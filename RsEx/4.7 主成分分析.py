import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from sklearn.decomposition import PCA

# 读取示例图片
image = ski.data.astronaut()
image_2d = image.reshape(-1, 3)
# 创建PCA对象
pca = PCA(n_components=1)

# 训练PCA
reduced_image = pca.fit_transform(image_2d)

# 重构图像
reconstructed_image = pca.inverse_transform(reduced_image)
reconstructed_image = reconstructed_image.reshape(image.shape)
# 转为uint8类型
reconstructed_image = np.uint8(reconstructed_image)

# 显示原始和变换后的图片
fig = make_subplots(rows=1, cols=2, subplot_titles=['原图', '第一主成分假彩色影像'])
fig.add_trace(px.imshow(image).data[0], row=1, col=1)
fig.add_trace(px.imshow(reconstructed_image).data[0], row=1, col=2)
fig.update_yaxes(autorange='reversed')
fig.update_traces(showscale=False, selector={'type': 'heatmap'})
fig.show()
