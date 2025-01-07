# 非线性回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 生成示例数据
np.random.seed(0)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 创建多项式回归模型（这里使用3次多项式）
model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())

# 训练模型
model.fit(X_train, y_train)

# 使用模型进行预测  X_test
X_test = sorted(X_test)  # 排序
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean squared error (MSE): {mse}")
print(f"R^2 score: {r2}")

# 可视化结果
plt.scatter(X, y, color='blue', label='Data')
# plt.scatter(X_test, y_pred, color='red', linewidth=2, label='Polynomial regression')
plt.plot(X_test, y_pred, color='red', linewidth=2,
         label='Polynomial regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
