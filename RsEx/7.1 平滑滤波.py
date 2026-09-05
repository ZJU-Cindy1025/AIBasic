import numpy as np
import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots

# 读取示例图片
img1 = ski.data.camera()

# 生成高斯噪声
noise = np.random.normal(0, 50, img1.shape)

# 添加高斯噪声
img2 = img1 + noise
img2 = np.clip(img2, 0, 255).astype(np.uint8)
# 均值滤波
img3 = ski.filters.rank.mean(img2, np.ones((3, 3)), mask=None)
# 中值滤波
img4 = ski.filters.rank.median(img2, np.ones((3, 3)), mask=None)
# 最小值滤波
img5 = ski.filters.rank.minimum(img2, np.ones((3, 3)), mask=None)
# 最大值滤波
img6 = ski.filters.rank.maximum(img2, np.ones((3, 3)), mask=None)

# 显示结果
fig = make_subplots(rows=2, cols=3, subplot_titles=[
                    '原始图像', '添加高斯噪声', '均值滤波', '中值滤波', '最小值滤波', '最大值滤波'])
for index, image_data in enumerate([img1, img2, img3, img4, img5, img6]):
    fig.add_trace(
        px.imshow(image_data).data[0], row=index // 3 + 1, col=index % 3 + 1)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()
