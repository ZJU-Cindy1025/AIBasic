# 二元非线性回归，并作3D图
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# 假设我们有一些数据（这里用非线性关系的数据作为示例）
np.random.seed(0)
X = np.random.rand(100, 2)  # 100个样本，每个样本有2个特征
# 创建一个非线性关系：y = x1^2 - x2^2 + 噪声
def true_function(x): return x[:, 0]**2 - x[:, 1]**2


y = true_function(X) + np.random.randn(100) * 0.5

# 创建多项式回归模型（这里使用二次多项式）
degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# 拟合模型
model.fit(X, y)

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
                rstride=100, cstride=100, label='Fitted Surface')

# 添加图例和标签
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('Binary Nonlinear Regression (Polynomial) 3D Plot')

# 显示图形
plt.show()

# 打印模型的系数（注意：由于使用了多项式特征，系数数量会增加）
print("Model coefficients (including polynomial terms):")
print(model.named_steps['linearregression'].coef_)
print("Intercept:")
print(model.named_steps['linearregression'].intercept_)
