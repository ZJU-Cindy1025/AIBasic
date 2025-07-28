from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 生成数据
data = pd.DataFrame({
    'x': np.r_[np.random.normal(10, 3, 100), np.random.normal(20, 3, 100)],
    'y': np.r_[np.random.normal(10, 3, 100), np.random.normal(20, 3, 100)],
    'label': np.r_[np.repeat(0, 100), np.repeat(1, 100)]
})

model = svm.LinearSVC(loss='hinge')
model.fit(data[['x', 'y']], data['label'])

# 获取分割超平面
k = -model.coef_[0][0]/model.coef_[0][1]
xx = np.linspace(np.min(data['x']), np.max(data['x']))
yy = k*xx-model.intercept_/model.coef_[0][1]

# 绘图
sns.scatterplot(x='x', y='y', hue='label', data=data)
plt.plot(xx, yy)
plt.show()
