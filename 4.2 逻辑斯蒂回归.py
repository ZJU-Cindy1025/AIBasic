from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Binarizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
# 输出data1的线性回归
sns.regplot(x='商品价格', y='拒绝意愿', data=data1, ax=ax[0, 0], logistic=False)
ax[0, 0].set_title('data1线性回归')
# 输出data2的线性回归
sns.regplot(x='商品价格', y='拒绝意愿', data=data2, ax=ax[0, 1], logistic=False)
ax[0, 1].set_title('data2线性回归')
# 输出data1的逻辑斯蒂回归
sns.regplot(x='商品价格', y='拒绝意愿', data=data1, ax=ax[1, 0], logistic=True)
ax[1, 0].set_title('data1逻辑斯蒂回归')
# 输出data2的逻辑斯蒂回归
sns.regplot(x='商品价格', y='拒绝意愿', data=data2, ax=ax[1, 1], logistic=True)
ax[1, 1].set_title('data2逻辑斯蒂回归')
plt.tight_layout()
plt.show()
