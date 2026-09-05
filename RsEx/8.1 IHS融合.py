import numpy as np
import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
# 读取两幅图像
img1 = ski.data.astronaut()
img2 = ski.data.chelsea()

# 将img1调整为img2大小
img1 = ski.transform.resize(img1, img2.shape)

# 计算IHS变换
img1hsv = ski.color.rgb2hsv(img1)
img2hsv = ski.color.rgb2hsv(img2)
img3 = np.zeros_like(img1)
# 保留img1的亮度信息，img2的色彩信息
img3[:, :, 0] = img1hsv[:, :, 0]
img3[:, :, 1] = img2hsv[:, :, 1]
img3[:, :, 2] = img2hsv[:, :, 2]
img3 = ski.color.hsv2rgb(img3)

# 显示结果
fig = make_subplots(rows=1, cols=3, subplot_titles=['图像1', '图像2', 'IHS融合后图像'])
for column, image_data in enumerate([img1, img2, img3], 1):
    fig.add_trace(px.imshow(image_data).data[0], row=1, col=column)
fig.update_yaxes(autorange='reversed')
fig.update_traces(showscale=False, selector={'type': 'heatmap'})
fig.show()
