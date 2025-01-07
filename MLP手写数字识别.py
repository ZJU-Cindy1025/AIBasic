from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# 加载MNIST数据集
mtData = fetch_openml('mnist_784', version=1)
# 归一化处理，每个像素的取值范围是0~255
X, y = mtData.data / 255.0, mtData.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

print(X_train.shape)  # 查看数据形状

# 创建并训练MLP分类器
# hidden_layer_sizes隐藏层的数量和每层的神经元数量
# max_iter迭代次数
# random_state随机数种子，用于训练的可重复性
myMLP = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=10, random_state=42)

# 模型训练
myMLP.fit(X_train, y_train)

# 评估模型
score = myMLP.score(X_test, y_test)
# 输出准确率
print(f"Model accuracy: {score:.2f}")
