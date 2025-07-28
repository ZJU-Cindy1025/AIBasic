from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].scatter(data['气温温度(x)'], data['火灾影响面积(y)'])
ax[0].plot(data['气温温度(x)'], a+b*data['气温温度(x)'], color='red')
ax[0].set_xlabel('气温温度(x)')
ax[0].set_ylabel('火灾影响面积(y)')
ax[0].set_title('Matplotlib绘图')

ax[1] = sns.regplot(x='气温温度(x)', y='火灾影响面积(y)', data=data, ax=ax[1])
ax[1].set_title('Seaborn绘图')

plt.show()
