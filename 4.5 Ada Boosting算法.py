from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
data = pd.DataFrame({
    'x': np.linspace(-9, 9, 10),
    'y': [-1, -1, 1, 1, -1, -1, -1, -1, 1, 1],
})

x = data['x'].values.reshape(-1, 1)
y = data['y'].values

clf = AdaBoostClassifier(n_estimators=3)
clf.fit(x, y)

# 输出模型的有关信息
print('各个特征的权重:', clf.feature_importances_)
print('每个弱分类器的权重:', clf.estimator_weights_)
print('每个弱分类器的信息:', clf.estimators_)
print('预测结果:', clf.predict(x))
print('预测概率:', clf.predict_proba(x))
