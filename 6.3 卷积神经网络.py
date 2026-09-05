import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import plotly.express as px
from plotly.subplots import make_subplots
from skimage import data
from tqdm import tqdm
import numpy as np


# 初始影像
origin_data = data.binary_blobs()
# 标签
label_data = origin_data.copy()
label_data = label_data.astype(np.uint8)
# 训练图像
train_data = np.stack((origin_data/3, origin_data/2, origin_data/4), axis=-1)
train_data = train_data*255
train_data = train_data.astype(np.uint8)
train_data[train_data == 0] = 200
# 转换数据为 PyTorch Tensor
# 卷积神经网络输入的数组结构是[batch, channel, height, width]，其中 batch 是输入的数量，channel 是输入的通道数，height 和 width 是输入的高和宽，所以首先要把通道数放在第二个维度，然后增加一个 batch 维度（batch=1）
x = torch.tensor(train_data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
y = torch.tensor(label_data, dtype=torch.float32).unsqueeze(0)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 输入 3 通道，输出 6 通道，卷积核 3x3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=3, stride=1, padding=1)
        # 卷积神经网络的输出的数组结构是[batch, channel, height, width]，其中 channel 是卷积核的数量，height 和 width 是卷积后的图像的高和宽
        # 池化层，2x2 平均池化
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 线性层，输入卷积后的 6*256*256，输出 10
        self.fc1 = nn.Linear(6*256*256, 10)
        # 线性层，输入 10，输出供预测的 512*512
        self.fc2 = nn.Linear(10, 512*512)

    def forward(self, x):
        # 卷积
        x = self.conv1(x)
        # 池化
        x = self.pool(x)
        # 展平
        x = x.view(-1, 6*256*256)
        # 使用 softmax 激活函数
        x = F.softmax(self.fc1(x))
        x = self.fc2(x)
        return x


# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(device)
# 模型
net = CNN()
net = net.to(device)
# 二分类交叉熵损失函数
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)
# Adam 优化器
optimaizer = optim.Adam(net.parameters(), lr=0.1)
losses = []
# 训练
for epoch in tqdm(range(100)):
    out = net(x)
    loss = criterion(out.view(-1), y.view(-1))
    optimaizer.zero_grad()
    loss.backward()
    optimaizer.step()
    losses.append(loss.item())

# 预测
predict = net(x)
fig = make_subplots(rows=2, cols=2, subplot_titles=['原图', '标签', '预测', '损失'])
fig.add_trace(px.imshow(train_data).data[0], row=1, col=1)
fig.add_trace(px.imshow(label_data).data[0], row=1, col=2)
fig.add_trace(px.imshow(np.array(predict.cpu().detach().numpy()
                                 ).reshape(512, 512)).data[0], row=2, col=1)
fig.add_trace(px.line(y=losses).data[0], row=2, col=2)
fig.show()
