import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import plotly.express as px
from plotly.subplots import make_subplots
from tqdm import tqdm
import numpy as np
from skimage import data


class _Dataset(Dataset):
    def __init__(self, crop_size, overlap):
        # 影像和标签集合
        self.images = []
        self.labels = []
        # 初始影像
        origin_data = data.binary_blobs()
        # 标签
        label_data = origin_data.copy()
        self.label_data = label_data.astype(np.float32)
        # 训练图像
        train_data = np.stack(
            (origin_data / 3, origin_data / 2, origin_data / 4), axis=-1
        )
        train_data = train_data * 255
        train_data[train_data == 0] = 200
        self.train_data = train_data.astype(np.float32)
        # 裁剪图像
        overlap = 0.5
        # 计算裁剪的步长
        step = int(crop_size * (1 - overlap))
        # 进行裁剪
        for i in range(0, label_data.shape[0] - crop_size + 1, step):
            for j in range(0, label_data.shape[1] - crop_size + 1, step):
                # 切片操作进行裁剪
                self.images.append(
                    self.train_data[i: i + crop_size, j: j + crop_size, :]
                )
                self.labels.append(
                    self.label_data[i: i + crop_size, j: j + crop_size]
                )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].transpose(2, 0, 1)
        lab = self.labels[idx]
        return img, lab


# 定义unet模型相关类
class conv_conv(nn.Module):
    """conv_conv: (conv[3*3] + BN + ReLU) *2"""

    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        super(conv_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, X):
        X = self.conv(X)
        return X


class downconv(nn.Module):
    """downconv: conv_conv => maxpool[2*2]"""

    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        super(downconv, self).__init__()
        self.conv = conv_conv(in_channels, out_channels, bn_momentum)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        X = self.conv(X)
        pool_X = self.pool(X)
        return pool_X, X


class upconv_concat(nn.Module):
    """upconv_concat: upconv[2*2] => cat => conv_conv"""

    def __init__(self, in_channels, out_channels, bn_momentum=0.1):
        super(upconv_concat, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = conv_conv(in_channels, out_channels, bn_momentum)

    def forward(self, X1, X2):
        X1 = self.upconv(X1)
        feature_map = torch.cat((X2, X1), dim=1)
        X1 = self.conv(feature_map)
        return X1


# 层数：3
class UNet(nn.Module):
    """UNet(3-level): downconv *3 => conv_conv => upconv *3 => conv[1*1]"""

    def __init__(self, in_channels, out_channels, starting_filters=32, bn_momentum=0.1):
        super(UNet, self).__init__()
        self.conv1 = downconv(in_channels, starting_filters, bn_momentum)
        self.conv2 = downconv(
            starting_filters, starting_filters * 2, bn_momentum)
        self.conv3 = downconv(starting_filters * 2,
                              starting_filters * 4, bn_momentum)
        self.convconv = conv_conv(
            starting_filters * 4, starting_filters * 8, bn_momentum
        )
        self.upconv3 = upconv_concat(
            starting_filters * 8, starting_filters * 4, bn_momentum
        )
        self.upconv2 = upconv_concat(
            starting_filters * 4, starting_filters * 2, bn_momentum
        )
        self.upconv1 = upconv_concat(
            starting_filters * 2, starting_filters, bn_momentum
        )
        self.conv_out = nn.Conv2d(
            starting_filters, out_channels, kernel_size=1, padding=0, stride=1
        )

    def forward(self, X):
        X, conv1 = self.conv1(X)
        X, conv2 = self.conv2(X)
        X, conv3 = self.conv3(X)
        X = self.convconv(X)
        X = self.upconv3(X, conv3)
        X = self.upconv2(X, conv2)
        X = self.upconv1(X, conv1)
        X = self.conv_out(X)
        return X


# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = UNet(in_channels=3, out_channels=1)
net.to(device)
TestDataset = _Dataset(16, 0.2)
losses = []
train_loader = DataLoader(TestDataset, batch_size=16)
optimizer = optim.Adam(net.parameters(), lr=0.01)
critertion = nn.BCEWithLogitsLoss()
critertion = critertion.to(device)
for epoch in tqdm(range(100)):
    total_loss = 0
    count = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)  # 将影像输入网络得到输出
        optimizer.zero_grad()  # 将梯度置为0
        loss = critertion(output, labels.unsqueeze(1))
        total_loss += loss.item()
        count += 1
        loss.backward()  # 损失反向传播
        optimizer.step()
    losses.append(total_loss/count)
# 预测
Test_image = torch.tensor(
    TestDataset.train_data.transpose(2, 0, 1)).unsqueeze(0)  # 转换维度
Test_image = Test_image.to(device)
output = net(Test_image)
output = output.cpu().detach().numpy()
output = output.squeeze()

fig = make_subplots(rows=2, cols=2, subplot_titles=['原始影像', '标签', '预测', '损失'])
fig.add_trace(px.imshow(TestDataset.train_data.astype(
    np.uint8)).data[0], row=1, col=1)
fig.add_trace(px.imshow(TestDataset.label_data.astype(
    np.uint8), color_continuous_scale='gray').data[0], row=1, col=2)
fig.add_trace(px.imshow((255-output).astype(np.uint8)).data[0], row=2, col=1)
fig.add_trace(px.line(y=losses).data[0], row=2, col=2)
fig.update_traces(colorscale='gray', coloraxis=None, showscale=False,
                  selector={'type': 'heatmap'}, row=1, col=2)
fig.show()
