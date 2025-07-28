from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
sns.scatterplot(x='运动频率', y='吸烟频率', hue='患病情况', data=data)
# 绘制投影直线w
w = lda.coef_[0]
a = w[1] / w[0]
xx = np.linspace(data['运动频率'].min(), data['运动频率'].max())
yy = a * xx - (lda.intercept_[0]) / w[1]
plt.plot(xx, yy)
plt.scatter(lda.means_[:, 0], lda.means_[:, 1], s=100, c='r', marker='*')
plt.show()
