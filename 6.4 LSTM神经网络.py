import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import plotly.express as px
from tqdm import tqdm


# 模拟时间序列数据集
class _Dataset(Dataset):
    def __init__(self, data_num):
        # 输入：[n,n+1,n+2,n+3,n+4]；结果：[n+5]，共data_num组测试数据
        x = [
            [i % 5, i % 5 + 1, i % 5 + 2, i % 5 + 3, i % 5 + 4] for i in range(data_num)
        ]
        y = [[i % 5 + 5] for i in range(data_num)]
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y


# LSTM神经网络
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        # LSTM层，input_size为输入x的数组列数，hidden_size和num_layers为模型自身参数
        self.lstm = nn.LSTM(input_size=5, hidden_size=3,
                            num_layers=1, batch_first=True)
        # 全连接层，输入为LSTM层的输出（隐藏层个数），输出为每个y的列数（这里是1）
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 数据集
TestDataset = _Dataset(1000)
train_loader = DataLoader(TestDataset, batch_size=16, shuffle=True)
net = LSTM()
criterion = nn.MSELoss()
net = net.to(device)
criterion = criterion.to(device)
losses = []
for epoch in tqdm(range(100)):
    trainning_loss = 0
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        # 求输出
        outputs = net(x)
        # 梯度清零
        optimizer.zero_grad()
        # 计算损失函数
        loss = criterion(outputs, y)
        # 反向传播
        loss.backward()
        # 优化参数
        optimizer.step()
        # 记录损失值
        trainning_loss += loss.item()
    trainning_loss /= len(train_loader)
    losses.append(trainning_loss)

# 验证
for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)
    print(x.cpu().detach().numpy())
    print(net(x).cpu().detach().numpy().reshape(1, -1))
    print(y.cpu().detach().numpy().reshape(1, -1))

fig = px.line(y=losses, labels={'x': '迭代次数', 'y': '损失'})
fig.show()
