import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
# 定义AlexNet模型
# pyTorch的torchvision.models模块中已经有AlexNet的预定义实现
# 也可以从头开始实现
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 卷积核96个，输出尺寸55*55：(224+2*2-11)/4+1=55，
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # 输出尺寸27*27：(55-3)/2+1=27
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 卷积核256个，输出尺寸27*27：(27+2*2-5)/1+1=27
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # 输出尺寸13*13：(27-3)/2+1=13
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 卷积核384个，输出尺寸13*13：(13+1*2-3)/1+1=13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 卷积核384个，输出尺寸13*13：(13+1*2-3)/1+1=13
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 卷积核256个，输出尺寸13*13：13+1*2-3)/1+1=13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输出尺寸6*6：(13-3)/2+1=6
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 最终输出尺寸：6*6，卷积核256个，特征值数量：6*6*256=9126
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
            # nn.Softmax() 与交叉熵损失函数使用时，可省略。

        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def AlexNet_Demo():
    model = AlexNet(10)

    # 加载图像数据
    transform = transforms.Compose(
        [torchvision.transforms.Resize((224, 224)),
         torchvision.transforms.RandomGrayscale(p=0.3),
         # 将Image对象转换为Tensor张量
         torchvision.transforms.ToTensor()]
    )
    trainset = torchvision.datasets.CIFAR10(
        root='./cifar_data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./cifar_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    print("start")
    # 训练模型
    for epoch in tqdm(range(2),desc='epoch'):  # 通常需要多个轮数来训练网络
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0),desc='batch',total=len(trainloader)):
            # 获取输入
            inputs, labels = data

            # 把梯度设置为零，这样才能进行累加
            optimizer.zero_grad()
            # 正向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 优化模型参数
            optimizer.step()

            # 打印信息
            running_loss += loss.item()
            if (i + 1) % 10 == 0:  # 每2000个批次打印一次
                print('\n[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
    print('Finished Training')

if(__name__ == "__main__"):
    AlexNet_Demo()
