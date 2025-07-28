from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['气温温度(x)'], data['风力(z)'], data['火灾影响面积(y)'])
ax.set_xlabel('气温温度(x)')
ax.set_ylabel('风力(z)')
ax.set_zlabel('火灾影响面积(y)')
# 添加回归平面
x1 = data['气温温度(x)']
x2 = data['风力(z)']
x1, x2 = np.meshgrid(x1, x2)
y = a+b1*x1+b2*x2
ax.plot_surface(x1, x2, y, color='blue', alpha=0.1)

plt.show()
