import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
# 读取图像
img = ski.data.horse()
# 反色，更好观察结果
img = ~img

# 结构元素
selem = ski.morphology.disk(10)

# 腐蚀
eroded = ski.morphology.erosion(img, selem)

# 膨胀
dilated = ski.morphology.dilation(img, selem)

# 开运算
opened = ski.morphology.opening(img, selem)

# 闭运算
closed = ski.morphology.closing(img, selem)

# 显示结果
fig = make_subplots(rows=1, cols=5, subplot_titles=[
                    '原图', '腐蚀', '膨胀', '开运算', '闭运算'])
for column, image_data in enumerate([img, eroded, dilated, opened, closed], 1):
    fig.add_trace(px.imshow(
        image_data).data[0], row=1, col=column)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()
