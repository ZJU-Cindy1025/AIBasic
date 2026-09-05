import numpy as np
import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
import pywt
# 读取示例图片
image = ski.data.camera()

# 设置小波系数
coeffs = pywt.dwt2(image, wavelet='haar')
cA, (cH, cV, cD) = coeffs

# 显示结果
fig = make_subplots(rows=2, cols=2, subplot_titles=[
                    '原图', '水平方向', '垂直方向', '对角线方向'])
for index, image_data in enumerate([image, cA, cV, cD]):
    fig.add_trace(
        px.imshow(image_data).data[0], row=index // 2 + 1, col=index % 2 + 1)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()
