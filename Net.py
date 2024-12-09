import torch.nn as nn


class AlexNet(nn.Module):
    """
    AlexNet模型
    """

    def __init__(self, num_classes: int, linear_num: int, p: float, activation):
        """初始化函数
        :param num_classes：输出类别数
        :param linear_num：全连接层神经元数
        :param p：Dropout概率
        :param activation：激活函数
        """
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=5, padding=1), activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=1), activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), activation,
            nn.Conv2d(384, 384, kernel_size=3, padding=1), activation,
            nn.Conv2d(384, 256, kernel_size=3, padding=1), activation,
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(4096, linear_num), activation,
            nn.Dropout(p),
            nn.Linear(linear_num, linear_num), activation,
            nn.Dropout(p),
            nn.Linear(linear_num, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.permute(0, 3, 1, 2)
        x = self.net(x)
        return x


class MLP(nn.Module):
    """
    MLP模型
    """

    def __init__(self, num_classes: int, linear_num: list, activation):
        """初始化函数
        :param num_classes：输出类别数
        :param linear_num：每层MLP的神经元数量，以数组形式呈现
        :param activation：激活函数
        """
        super(MLP, self).__init__()
        assert len(linear_num) == 4
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 48, linear_num[0]),
            activation,
            nn.Linear(linear_num[0], linear_num[1]),
            activation,
            nn.Linear(linear_num[1], linear_num[2]),
            activation,
            nn.Linear(linear_num[2], linear_num[3]),
            activation,
            nn.Linear(linear_num[3], num_classes)
        )

    def forward(self, x):
        x = self.net(x)
        return x
