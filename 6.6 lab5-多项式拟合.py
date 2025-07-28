import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
# 输入数据
a = [i for i in range(21)]
b = [i**2-20*i+40 for i in a]
# 数据量减少有利于拟合，但模型通用性不强

# 转换为tensor
a = torch.tensor(a)
b = torch.tensor(b)


# 定义网络，这个网络用于拟合y=f(x)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(len(a), 10)
        self.fc2 = nn.Linear(10, len(b))
        # 增加参数，拟合速度加快，效果变好，但更容易出现过拟合

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 实例化网络
net = Net()

# 定义优化器和损失函数
optimizer = optim.Adam(
    net.parameters(), lr=0.1
)  # 提高学习率，前面的学习速度加快，但后面更容易出问题
criterion = nn.MSELoss()

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = a.to(device)
b = b.to(device)
net = net.to(device)
criterion = criterion.to(device)

losses = []
# 绘图
fig, ax = plt.subplots(2, 5, figsize=(20, 8))
# 训练网络
for epoch in tqdm(range(1, 101)):
    # 梯度清零
    optimizer.zero_grad()
    # 前向传播
    output = net(a.float())
    # 计算损失
    loss = criterion(output, b.float())
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()
    losses.append(loss.item())
    if epoch % 10 == 0:
        output = net(a.float())
        ax[(epoch - 1) // 10 // 5, (epoch - 1) // 10 % 5].plot(
            a.cpu().numpy(), output.cpu().detach().numpy().reshape(-1)
        )
        ax[(epoch - 1) // 10 // 5, (epoch - 1) // 10 % 5].scatter(
            a.cpu().numpy(), b.cpu().numpy()
        )
        ax[(epoch - 1) // 10 // 5, (epoch - 1) // 10 % 5].set_title(
            "第%d次拟合" % epoch
        )
plt.show()
plt.close()
# 测试网络
output = net(a.float())
result = pd.DataFrame({"输入值": a.cpu().numpy(), "模型输出值": output.cpu(
).detach().numpy().reshape(-1), "标签": b.cpu().numpy()})
result.set_index("输入值", inplace=True)
print(result)
plt.plot(losses)
plt.show()
