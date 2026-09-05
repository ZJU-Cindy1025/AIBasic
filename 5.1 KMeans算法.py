from sklearn.cluster import KMeans
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 读取iris数据集
data = datasets.load_iris()
# 标准化
scaler = StandardScaler()
data['data'] = scaler.fit_transform(data['data'])
# 肘方法看K值
inertia = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data['data'])
    inertia.append(kmeans.inertia_)
n_clusters = 3
# 计算聚类精度
model = KMeans(n_clusters=n_clusters)
model.fit(data['data'])

fig = make_subplots(rows=1, cols=3, subplot_titles=['肘方法', 'KMeans聚类', '参考标签'])
fig.add_trace(px.line(x=list(range(1, 10)), y=inertia).data[0], row=1, col=1)
for label, group in pd.DataFrame({'x': data['data'][:, 0], 'y': data['data'][:, 1], 'label': model.labels_}).groupby('label'):
    fig.add_trace(px.scatter(group, x='x', y='y',
                  title=f'cluster {label}').data[0], row=1, col=2)
centers = px.scatter(x=model.cluster_centers_[
                     :, 0], y=model.cluster_centers_[:, 1]).data[0]
centers.marker.symbol = 'x'
centers.marker.color = 'red'
fig.add_trace(centers, row=1, col=2)
for label, group in pd.DataFrame({'x': data['data'][:, 0], 'y': data['data'][:, 1], 'label': data['target']}).groupby('label'):
    fig.add_trace(px.scatter(group, x='x', y='y',
                  title=f'target {label}').data[0], row=1, col=3)
fig.update_xaxes(title_text='簇数', row=1, col=1)
fig.update_yaxes(title_text='簇内误差平方和', row=1, col=1)
fig.update_xaxes(title_text='花萼长度', row=1, col=2)
fig.update_yaxes(title_text='花萼宽度', row=1, col=2)
fig.update_xaxes(title_text='花萼长度', row=1, col=3)
fig.update_yaxes(title_text='花萼宽度', row=1, col=3)
fig.show()
