import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
# 读取训练后数据
pData = ski.io.imread(r'RsEx/coins_result/coins_KNN_result.png', as_gray=True)

# 扫描线图斑聚类算法
pDstData = ski.measure.label(pData, background=0)
# 获取图斑信息
region = ski.measure.regionprops(ski.util.img_as_ubyte(pData))
df = pd.DataFrame(columns=[prop for prop in region[0]])
for i in range(len(region)):
    for prop in region[i]:
        df.loc[i, prop] = region[i][prop]
print(df)

# 显示结果
fig = make_subplots(rows=1, cols=2, subplot_titles=['原始分类后数据', '扫描线聚类算法'])
for column, image_data in enumerate([pData, pDstData], 1):
    fig.add_trace(px.imshow(
        image_data).data[0], row=1, col=column)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()
