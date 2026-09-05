import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots

# 读取图像
img1 = ski.data.horse()
# 一阶微分找边
edge = ski.filters.sobel(img1)

# 寻找线
labels = ski.segmentation.find_boundaries(img1, mode='thick')
labels = ski.util.img_as_ubyte(labels)

# 显示结果
fig = make_subplots(rows=1, cols=3, subplot_titles=[
                    '原图像', '一阶微分法找边', '使用典型模板找线'])
for column, image_data in enumerate([img1, edge, labels], 1):
    fig.add_trace(px.imshow(
        image_data).data[0], row=1, col=column)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()
