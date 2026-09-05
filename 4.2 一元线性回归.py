from sklearn.linear_model import LinearRegression
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

data = pd.DataFrame({
    '气温温度(x)': [5.1, 8.2, 11.5, 13.9, 15.1, 16.2, 19.6, 23.3],
    '火灾影响面积(y)': [2.14, 4.62, 8.24, 11.24, 13.99, 16.33, 19.23, 28.74]
})

x = data['气温温度(x)'].values.reshape(-1, 1)
y = data['火灾影响面积(y)'].values

model = LinearRegression()
model.fit(x, y)

a = model.intercept_
b = model.coef_
print('回归参数a的值：', a)
print('回归参数b的值：', b)

fig = make_subplots(rows=1, cols=1, subplot_titles=['一元线性回归绘图'])
fig.add_trace(px.scatter(data, x='气温温度(x)',
              y='火灾影响面积(y)').data[0], row=1, col=1)
fig.add_trace(px.line(x=data['气温温度(x)'], y=a+b *
              data['气温温度(x)']).data[0], row=1, col=1)
fig.update_xaxes(title_text='气温温度(x)')
fig.update_yaxes(title_text='火灾影响面积(y)')
fig.show()
