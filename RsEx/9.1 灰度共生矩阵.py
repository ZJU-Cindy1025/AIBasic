import skimage as ski
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
# 读取示例图片
img1 = ski.data.grass()
img2 = ski.data.gravel()

# 生成img1和img2的灰度共生矩阵，西北-东南向
glcm1 = ski.feature.graycomatrix(img1, [1], [1], 256)
glcm2 = ski.feature.graycomatrix(img2, [1], [1], 256)

# 显示灰度共生矩阵
fig = make_subplots(rows=2, cols=2, subplot_titles=[
                    'grass', 'gravel', 'grass的灰度共生矩阵', 'gravel的灰度共生矩阵'])
fig.add_trace(
    px.imshow(img1).data[0], row=1, col=1)
fig.add_trace(
    px.imshow(img2).data[0], row=1, col=2)
fig.add_trace(px.imshow(
    glcm1[:, :, 0, 0]).data[0], row=2, col=1)
fig.add_trace(px.imshow(
    glcm2[:, :, 0, 0]).data[0], row=2, col=2)
fig.update_yaxes(autorange='reversed')
fig.update_traces(colorscale='gray', coloraxis=None,
                  showscale=False, selector={'type': 'heatmap'})
fig.show()

# 计算灰度共生矩阵的特征
contrast1 = ski.feature.graycoprops(glcm1, 'contrast')
correlation1 = ski.feature.graycoprops(glcm1, 'correlation')
energy1 = ski.feature.graycoprops(glcm1, 'energy')
homogeneity1 = ski.feature.graycoprops(glcm1, 'homogeneity')
dissimilarity1 = ski.feature.graycoprops(glcm1, 'dissimilarity')
contrast2 = ski.feature.graycoprops(glcm2, 'contrast')
correlation2 = ski.feature.graycoprops(glcm2, 'correlation')
energy2 = ski.feature.graycoprops(glcm2, 'energy')
homogeneity2 = ski.feature.graycoprops(glcm2, 'homogeneity')
dissimilarity2 = ski.feature.graycoprops(glcm2, 'dissimilarity')

# 显示灰度共生矩阵的特征
df = pd.DataFrame(columns=['影像', '对比度', '相关性', '能量', '一致性', '差异性'])
df.loc[0] = ['img1', contrast1[0, 0], correlation1[0, 0],
             energy1[0, 0], homogeneity1[0, 0], dissimilarity1[0, 0]]
df.loc[1] = ['img2', contrast2[0, 0], correlation2[0, 0],
             energy2[0, 0], homogeneity2[0, 0], dissimilarity2[0, 0]]
print(df)
