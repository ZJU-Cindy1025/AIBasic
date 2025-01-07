# 多元线性回归
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 假设我们有一些数据（这里用随机数据作为示例）
# X 是特征矩阵，y 是目标向量
np.random.seed(0)  # 为了结果的可重复性
X = np.random.rand(100, 3)  # 100个样本，每个样本有3个特征
y = X @ np.array([1.5, -2., 1.]) + np.random.randn(100) * 0.5  # 真实关系加上一些噪声

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R² Score: {r2}")

# 打印模型的系数和截距
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
