from sklearn.mixture import GaussianMixture
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

# 生成数据
np.random.seed(0)
X = np.concatenate([np.random.normal(loc=1, scale=0.5, size=(50, 1)),
                    np.random.normal(loc=3, scale=0.5, size=(50, 1))])

# 初始化GaussianMixture模型
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)

# 使用EM算法拟合数据
gmm.fit(X)

# 输出各组成分的权重和均值方差
print("权重s:", gmm.weights_)
print("均值:", gmm.means_)
print("方差:", gmm.covariances_)

# 绘图
fig = make_subplots(rows=1, cols=2, subplot_titles=['数据分布', 'GMM拟合的数据分布'])
fig.add_trace(px.histogram(
    x=X.ravel(), histnorm='probability density').data[0], row=1, col=1)
fig.add_trace(px.histogram(x=gmm.sample(
    100)[0].ravel(), histnorm='probability density').data[0], row=1, col=2)
fig.show()
