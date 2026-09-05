import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# 读取图像
img1 = ski.data.astronaut()
# 平移
img2 = np.roll(img1, 100, axis=0)
# 旋转
img3 = np.rot90(img1)
# 缩放
img4 = ski.transform.resize(img1, (img1.shape[0]//2, img1.shape[1]//2))
# 翻转
img5 = np.flipud(img1)
# 镜像
img6 = np.fliplr(img1)

# 显示图像
fig = make_subplots(rows=2, cols=3, subplot_titles=[
                    '原图', '平移', '旋转', '缩放', '翻转', '镜像'])
for index, image_data in enumerate([img1, img2, img3, img4, img5, img6]):
    fig.add_trace(
        px.imshow(image_data).data[0], row=index // 3 + 1, col=index % 3 + 1)
fig.update_yaxes(autorange='reversed')
fig.update_traces(showscale=False, selector={'type': 'heatmap'})
fig.show()
