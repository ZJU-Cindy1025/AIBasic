import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
# 读取图像
img1 = ski.data.chelsea()  # 彩色图像
img2 = ski.data.horse()  # 0或1

# 把img1调整为img2大小
img1 = ski.transform.resize(img1, img2.shape)
# img1每个通道分别与img2相乘
img3 = np.zeros_like(img1)
for i in range(3):
    img3[:, :, i] = img1[:, :, i] * img2

# 显示图像
fig = make_subplots(rows=1, cols=3, subplot_titles=['图像1', '图像2', '乘运算图像'])
fig.add_trace(px.imshow(img1).data[0], row=1, col=1)
fig.add_trace(
    px.imshow(img2).data[0], row=1, col=2)
fig.add_trace(px.imshow(img3).data[0], row=1, col=3)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()
