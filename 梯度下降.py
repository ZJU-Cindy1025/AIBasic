# 二维曲面的梯度下降过程
import numpy as np
import matplotlib.pyplot as plt

# 定义损失函数（例如，一个简单的二次函数）


def loss_function(x, y):
    return (x - 2)**2 + (y - 3)**2  # 这是一个以(2, 3)为最小点的二次函数

# 定义损失函数的梯度


def gradient(x, y):
    grad_x = 2 * (x - 2)
    grad_y = 2 * (y - 3)
    return grad_x, grad_y

# 梯度下降算法


def gradient_descent(initial_point, learning_rate, num_iterations):
    x, y = initial_point
    trajectory = [(x, y)]  # 记录梯度下降过程中的点
    for _ in range(num_iterations):
        grad_x, grad_y = gradient(x, y)
        x -= learning_rate * grad_x
        y -= learning_rate * grad_y
        trajectory.append((x, y))
    return trajectory


# 初始化参数
initial_point = (0, 0)  # 起始点
learning_rate = 0.1  # 学习率
num_iterations = 50  # 迭代次数

# 执行梯度下降
trajectory = gradient_descent(initial_point, learning_rate, num_iterations)

# 转换为numpy数组以便绘图
trajectory = np.array(trajectory)

# 创建网格以绘制损失函数的曲面
x_vals = np.linspace(-1, 5, 400)
y_vals = np.linspace(-1, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = loss_function(X, Y)

# 绘制损失函数的曲面
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Loss Function Surface')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Loss')

# 绘制梯度下降的路径
ax2 = fig.add_subplot(122)
ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.plot(trajectory[:, 0], trajectory[:, 1], 'r-',
         linewidth=2, label='Gradient Descent Path')
ax2.scatter(trajectory[0, 0], trajectory[0, 1],
            color='blue', label='Start Point')
ax2.scatter(trajectory[-1, 0], trajectory[-1, 1],
            color='green', label='End Point')
ax2.set_title('Gradient Descent Path on Loss Function Contour')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.legend()

# 显示图形
plt.tight_layout()
plt.show()
