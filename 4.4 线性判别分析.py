from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
import plotly.express as px

# 生成数据
data = pd.DataFrame({
    '运动频率': np.array([0.01*i for i in range(1, 101)])+np.random.normal(0, 0.16, 100),
    '吸烟频率': np.array([0.01*i for i in range(1, 101)])+np.random.normal(0, 0.16, 100),
    '患病情况': [0 if i <= 50 else 1 for i in range(1, 101)],
})
data.loc[data['运动频率'] < 0, '运动频率'] = 0
data.loc[data['运动频率'] > 1, '运动频率'] = 1
data.loc[data['吸烟频率'] < 0, '吸烟频率'] = 0
data.loc[data['吸烟频率'] > 1, '吸烟频率'] = 1

# 生成LDA模型
lda = LinearDiscriminantAnalysis()
lda.fit(data.iloc[:, 0:2], data.iloc[:, 2])
print('回归参数a的值：', lda.intercept_)
print('回归参数b的值：', lda.coef_)

# 绘图
fig = px.scatter(data, x='运动频率', y='吸烟频率', color='患病情况')
# 绘制投影直线w
w = lda.coef_[0]
a = w[1] / w[0]
xx = np.linspace(data['运动频率'].min(), data['运动频率'].max())
yy = a * xx - (lda.intercept_[0]) / w[1]
fig.add_trace(px.line(x=xx, y=yy).data[0])
fig.add_trace(px.scatter(x=lda.means_[:, 0], y=lda.means_[:, 1]).data[0])
fig.show()
