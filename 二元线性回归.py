# 二元线性回归，并画出3D图
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 假设我们有一些数据（这里用随机数据作为示例）
np.random.seed(0)  # 为了结果的可重复性
X = np.random.rand(100, 2)  # 100个样本，每个样本有2个特征
true_coefficients = [1.5, -2.]
y = X @ true_coefficients + np.random.randn(100) * 0.5  # 真实关系加上一些噪声

# 分割数据为训练集和测试集（这里我们只用训练集来拟合模型并绘制3D图）
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.3, random_state=42)  # test_size=0.0意味着全部用作训练

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 创建网格以绘制3D图
x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x1, x2 = np.meshgrid(x1, x2)
X_grid = np.c_[x1.ravel(), x2.ravel()]
y_grid_pred = model.predict(X_grid)
y_grid_pred = y_grid_pred.reshape(x1.shape)

# 绘制3D图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Actual Data')
ax.plot_surface(x1, x2, y_grid_pred, cmap='viridis', alpha=0.7,
                rstride=100, cstride=100, label='Fitted Plane')

# 添加图例和标签
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('Binary Linear Regression 3D Plot')

# 显示图形
plt.show()

# 打印模型的系数和截距
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
