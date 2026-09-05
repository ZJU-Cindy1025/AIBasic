from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import plotly.graph_objects as go

data = pd.DataFrame({
    '气温温度(x)': [5.1, 8.2, 11.5, 13.9, 15.1, 16.2, 19.6, 23.3],
    '风力(z)': [4.5, 5.8, 4, 6.3, 4, 7.2, 6.3, 8.5],
    '火灾影响面积(y)': [2.14, 4.62, 8.24, 11.24, 13.99, 16.33, 19.23, 28.74],
})

x = data[['气温温度(x)', '风力(z)']]
y = data['火灾影响面积(y)']

model = LinearRegression()
model.fit(x, y)

a = model.intercept_
b1, b2 = model.coef_
print('回归参数a的值：', a)
print('回归参数b1的值：', b1)
print('回归参数b2的值：', b2)

# 绘制三维图
fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=data['气温温度(x)'],
    y=data['风力(z)'],
    z=data['火灾影响面积(y)'],
    mode='markers',
))
# 添加回归平面
x1 = data['气温温度(x)']
x2 = data['风力(z)']
x1, x2 = np.meshgrid(x1, x2)
y = a+b1*x1+b2*x2
fig.add_trace(go.Surface(x=x1, y=x2, z=y, opacity=0.1))
fig.update_layout(
    scene=dict(
        xaxis_title='气温温度(x)',
        yaxis_title='风力(z)',
        zaxis_title='火灾影响面积(y)',
    )
)

fig.show()
