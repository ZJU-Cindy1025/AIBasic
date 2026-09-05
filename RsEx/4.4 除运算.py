import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
# 读取图像
img1 = ski.data.grass()
img2 = ski.data.gravel()

# 除运算
img3 = img1 / img2
# 将img3化为0-255
img3 = img3 * 255
img3 = img3.astype(np.uint8)

# 显示图像
fig = make_subplots(rows=1, cols=3, subplot_titles=['图像1', '图像2', '除运算图像'])
fig.add_trace(
    px.imshow(img1).data[0], row=1, col=1)
fig.add_trace(
    px.imshow(img2).data[0], row=1, col=2)
fig.add_trace(
    px.imshow(img3).data[0], row=1, col=3)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()
