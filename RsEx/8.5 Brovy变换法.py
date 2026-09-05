import numpy as np
import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
# 读取两幅图像
img1 = ski.data.astronaut()
img2 = ski.data.chelsea()

# 将img1调整为img2大小
img1 = ski.transform.resize(img1, img2.shape)

# 将图像转换为灰度图
img1_gray = ski.color.rgb2gray(img1)
img2_gray = ski.color.rgb2gray(img2)
# 对img1进行多光谱波段颜色归一化
img1_norm = np.zeros_like(img1)
img1_norm[:, :, 0] = img1[:, :, 0] / np.sum(img1)
img1_norm[:, :, 1] = img1[:, :, 1] / np.sum(img1)
img1_norm[:, :, 2] = img1[:, :, 2] / np.sum(img1)
# 计算img2的全色影像的灰度值
img2_gray = img2_gray / 255
# 计算融合影像
img3 = np.zeros_like(img1)
img3[:, :, 0] = img2_gray * img1_norm[:, :, 0]
img3[:, :, 1] = img2_gray * img1_norm[:, :, 1]
img3[:, :, 2] = img2_gray * img1_norm[:, :, 2]
# 将图像归一化到0-255
img3 = 255*(img3 - np.min(img3)) / (np.max(img3) - np.min(img3))
img3 = img3.astype(np.uint8)

# 显示结果
fig = make_subplots(rows=1, cols=3, subplot_titles=['图像1', '图像2', 'Brovy后图像'])
for column, image_data in enumerate([img1, img2, img3], 1):
    fig.add_trace(px.imshow(image_data).data[0], row=1, col=column)
fig.update_yaxes(autorange='reversed')
fig.update_traces(showscale=False, selector={'type': 'heatmap'})
fig.show()
