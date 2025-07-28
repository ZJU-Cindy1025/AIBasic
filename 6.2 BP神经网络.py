import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 数据
x = torch.tensor([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
    [1, 1, 0],
], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [1], [0], [0], [0], [1]], dtype=torch.float32)

# 前馈神经网络


class BP(nn.Module):
    def __init__(self):
        super(BP, self).__init__()
        # 每个输入数组长度为3
        self.fc1 = nn.Linear(3, 2)
        # 对应输出数组长度为1
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        # 二分类使用sigmoid激活函数
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(device)
# 模型
net = BP()
net = net.to(device)
# 使用均方损失函数
criterion = nn.MSELoss()
criterion = criterion.to(device)
# 使用随机梯度下降
optimaizer = optim.SGD(net.parameters(), lr=0.1)
losses = []
# 训练
for epoch in tqdm(range(1000)):
    # 求输出
    out = net(x)
    # 计算损失函数
    loss = criterion(out, y)
    # 梯度清零
    optimaizer.zero_grad()
    # 反向传播
    loss.backward()
    # 优化参数
    optimaizer.step()
    # 记录损失值
    losses.append(loss.item())

# 预测
predict = net(x)
print(predict.cpu().detach().numpy())
plt.plot(losses)
plt.show()
