from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

data1 = pd.DataFrame({
    '商品价格': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    '拒绝意愿': [0, 0, 0, 0.1, 0.5, 0.9, 1, 1, 1],
})
data2 = pd.DataFrame({
    '商品价格': [1, 2, 3, 4, 5, 6, 7, 8, 9, 15],
    '拒绝意愿': [0, 0, 0, 0.1, 0.5, 0.9, 1, 1, 1, 1],
})
x1 = data1['商品价格'].values.reshape(-1, 1)
y1 = data1['拒绝意愿'].values
x2 = data2['商品价格'].values.reshape(-1, 1)
y2 = data2['拒绝意愿'].values

# 逻辑斯蒂模型，使用对数似然函数作为损失函数，使用梯度下降法求解参数
model = LogisticRegression(solver='liblinear')

# 给y1设置二分类标签
y1 = Binarizer(threshold=0.5).fit_transform(y1.reshape(1, -1)).ravel()
model.fit(x1, y1)
a1 = model.intercept_
b1 = model.coef_[0]
print('回归参数a1的值：', a1)
print('回归参数b1的值：', b1)

# 给y2设置二分类标签
y2 = Binarizer(threshold=0.5).fit_transform(y2.reshape(1, -1)).ravel()
# 逻辑斯蒂回归
model.fit(x2, y2)
a2 = model.intercept_
b2 = model.coef_
print('回归参数a2的值：', a2)
print('回归参数b2的值：', b2)

fig = make_subplots(rows=2, cols=2, subplot_titles=[
                    'data1线性回归', 'data2线性回归', 'data1逻辑斯蒂回归', 'data2逻辑斯蒂回归'])
for row, col, frame in [(1, 1, data1), (1, 2, data2)]:
    x_values = frame['商品价格']
    y_values = frame['拒绝意愿']
    line_x = np.linspace(x_values.min(), x_values.max(), 100)
    line_y = np.polyval(np.polyfit(x_values, y_values, 1), line_x)
    fig.add_trace(px.scatter(x=x_values, y=y_values).data[0], row=row, col=col)
    fig.add_trace(px.line(x=line_x, y=line_y).data[0], row=row, col=col)
for row, col, frame in [(2, 1, data1), (2, 2, data2)]:
    labels = (frame['拒绝意愿'] > 0.5).astype(int)
    logistic_model = LogisticRegression(
        solver='liblinear').fit(frame[['商品价格']], labels)
    line_x = np.linspace(frame['商品价格'].min(), frame['商品价格'].max(), 100)
    line_x_frame = pd.DataFrame({'商品价格': line_x})
    line_y = logistic_model.predict_proba(line_x_frame)[:, 1]
    fig.add_trace(px.scatter(
        x=frame['商品价格'], y=frame['拒绝意愿']).data[0], row=row, col=col)
    fig.add_trace(px.line(x=line_x, y=line_y).data[0], row=row, col=col)
fig.show()
