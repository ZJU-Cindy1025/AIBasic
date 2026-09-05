import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots

# 读取图像
img1 = ski.data.camera()

# 通过直方图找阈值
threhold1 = ski.filters.threshold_otsu(img1)
img2 = img1 > threhold1

# 基本全局阈值法
threhold2 = ski.filters.threshold_local(img1, method='mean')
img3 = img1 > threhold2

# P参数法
threhold3 = ski.filters.threshold_multiotsu(img1)
img4 = img1.copy()
img4[img1 < threhold3[0]] = 0
img4[(img1 >= threhold3[0]) & (img1 < threhold3[1])] = 128
img4[(img1 >= threhold3[1])] = 255

# 最优准则法
threhold4 = ski.filters.threshold_yen(img1)
img5 = img1 > threhold4

# 显示结果
fig = make_subplots(rows=1, cols=5, subplot_titles=[
                    '原图像', '直方图法', '基本全局阈值法', 'P参数法', '最优准则法'])
for column, image_data in enumerate([img1, img2, img3, img4, img5], 1):
    fig.add_trace(px.imshow(
        image_data).data[0], row=1, col=column)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()
