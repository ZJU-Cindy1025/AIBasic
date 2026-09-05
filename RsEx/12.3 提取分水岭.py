import numpy as np
import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
# 读取图像
coins = ski.data.coins()
# 提取边缘
edges = ski.filters.sobel(coins)

# 生成种子
grid = ski.util.regular_grid(coins.shape, n_points=468)
seeds = np.zeros(coins.shape, dtype=int)
seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1

# 分水岭算法分割
watershed = ski.segmentation.watershed(edges, seeds)

fig = make_subplots(rows=1, cols=2, subplot_titles=['原图', '分水岭算法分割'])
for column, image_data in enumerate([coins, watershed], 1):
    fig.add_trace(px.imshow(
        image_data).data[0], row=1, col=column)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()
