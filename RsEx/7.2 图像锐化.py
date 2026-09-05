import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots

# 读取示例图片
img1 = ski.data.camera()

# 拉普拉斯锐化
img2 = ski.filters.laplace(img1)

# Roberts算子锐化
img3 = ski.filters.roberts(img1)

# Sobel算子锐化
img4 = ski.filters.sobel(img1)

# unsharp_mask锐化
img5 = ski.filters.unsharp_mask(img1, radius=1, amount=10)

# 显示结果
fig = make_subplots(rows=1, cols=5, subplot_titles=[
                    '原图', '拉普拉斯锐化', 'Roberts算子锐化', 'Sobel算子锐化', 'unsharp_mask锐化'])
image_traces = [
    px.imshow(img1).data[0],
    px.imshow(img2).data[0],
    px.imshow(img3).data[0],
    px.imshow(img4).data[0],
    px.imshow(img5).data[0],
]
image_traces[0].colorscale = 'gray'
image_traces[1].colorscale = 'gray'
image_traces[2].colorscale = 'gray'
image_traces[3].colorscale = 'gray'
image_traces[4].colorscale = 'gray'
for trace in image_traces:
    trace.coloraxis = None
image_traces[1].zmid = 0
image_traces[2].zmid = 0
image_traces[3].zmid = 0
for column, trace in enumerate(image_traces, 1):
    fig.add_trace(trace, row=1, col=column)
fig.update_yaxes(autorange='reversed')
fig.update_traces(showscale=False, selector={'type': 'heatmap'})
fig.show()
