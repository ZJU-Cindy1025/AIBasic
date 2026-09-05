import numpy as np
import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots

# 读取示例图片
img1 = ski.data.astronaut()
# 图像反转
img2 = ski.util.invert(img1)
# 直方图均衡化
img3 = np.stack([
    ski.exposure.equalize_hist(img1[:, :, channel])
    for channel in range(img1.shape[2])
], axis=-1)
# 直方图拉伸
p, q = np.percentile(img1, (30, 70))
img4 = ski.exposure.rescale_intensity(img1, in_range=(p, q))
# 直方图规定化
reference_image = ski.data.colorwheel()
img5 = np.stack([
    ski.exposure.match_histograms(
        img1[:, :, channel], reference_image[:, :, channel])
    for channel in range(img1.shape[2])
], axis=-1)

# 显示结果
fig = make_subplots(rows=2, cols=5, subplot_titles=[
                    '原图', '反转', '直方图均衡化', '直方图拉伸', '直方图规定化', '原图直方图', '反转直方图', '直方图均衡化直方图', '直方图拉伸直方图', '直方图规定化直方图'])
fig.add_trace(px.imshow(img1).data[0], row=1, col=1)
fig.add_trace(px.imshow(img2).data[0], row=1, col=2)
fig.add_trace(px.imshow(img3).data[0], row=1, col=3)
fig.add_trace(px.imshow(img4).data[0], row=1, col=4)
fig.add_trace(px.imshow(img5).data[0], row=1, col=5)


def add_rgb_histogram(image, column):
    channel_names = ['R', 'G', 'B']
    for channel, name in enumerate(channel_names):
        channel_data = image[:, :, channel]
        value_range = (0, 255) if np.issubdtype(channel_data.dtype, np.integer) else (
            float(channel_data.min()), float(channel_data.max()))
        counts, bin_edges = np.histogram(
            channel_data.ravel(), bins=256, range=value_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        trace = px.bar(x=bin_centers, y=counts, labels={
            'x': '像素值', 'y': '像素数量'}).data[0]
        trace.name = name
        trace.legendgroup = name
        trace.marker.color = {'R': 'red', 'G': 'green', 'B': 'blue'}[name]
        trace.opacity = 0.55
        fig.add_trace(trace, row=2, col=column)


for column, image in enumerate([img1, img2, img3, img4, img5], 1):
    add_rgb_histogram(image, column)
fig.update_layout(barmode='overlay')
fig.update_yaxes(autorange='reversed', row=1)
fig.update_traces(showscale=False, selector={'type': 'heatmap'})
fig.show()
