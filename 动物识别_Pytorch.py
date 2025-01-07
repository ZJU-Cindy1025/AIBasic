import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# 定义MLP模型


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 500)  # 输入层节点数 = 32*32*3（图片尺寸）
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)  # 输出层节点数 = 10（类别数）

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # 展平图片，降维到1维
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def MLP_DEMO():
    # 数据预处理
    transform = transforms.Compose([
        # 输出张量格式的数据
        transforms.ToTensor(),
        # 归一化RGB三个通道的数据，平均值=0.5，标准差=0.5
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据集
    train_dataset = datasets.CIFAR10(
        root='./cifar_data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(
        root='./cifar_data', train=False, download=True, transform=transform)

    # 批量加载数据用于后续训练加速。并打乱数据次序
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MLP()

    # 设定超参数
    learning_rate = 0.001  # 学习率
    epochs = 10  # 迭代次数
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 自带softmax算法
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in tqdm(range(epochs)):
        for images, labels in train_loader:
            optimizer.zero_grad()  # 很重要，误差不累积，置零
            outputs = model(images)  # 前向计算
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 误差反向传播
            optimizer.step()  # 用选定的优化器更新模型参数

        print(f'\nEpoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # 测试模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')


MLP_DEMO()
