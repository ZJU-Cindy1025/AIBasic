import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# BP神经网络


class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=True,
                   download=True, transform=transform),
    batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=False, transform=transform),
    batch_size=1000, shuffle=False
)

# 模型训练
print('模型训练')
model_train = BPNet()
criterion_train = nn.CrossEntropyLoss()
optimizer_train = optim.Adam(model_train.parameters(), lr=1e-5)
model_train = model_train.to(device)
criterion_train = criterion_train.to(device)
Losses_train = []
for epoch in tqdm(range(20)):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer_train.zero_grad()
        output = model_train(data)
        loss = criterion_train(output, target)
        loss.backward()
        optimizer_train.step()
    Losses_train.append(loss.item())

# 模型测试
print('模型测试')
for image, label in test_loader:
    image = image.to(device)
    label = label.to(device)
    output = model_train(image)
    pred = output.argmax(dim=1, keepdim=True)
    pred = pred.cpu().numpy().reshape(-1)
    label = label.cpu().numpy().reshape(-1)
    # 寻找pred和label不同的样本
    diff = np.where(pred != label)
    print(len(diff[0]), '个样本分类错误')
    break

# 对抗样本生成网络


class FGSMNet(nn.Module):
    def __init__(self):
        super(FGSMNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 28*28)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 初始化对抗样本生成网络并训练
print('对抗样本生成网络训练')
model_fgsm = FGSMNet().to(device)
optimizer_fgsm = optim.Adam(model_fgsm.parameters(), lr=1e-5)
model_fgsm.eval()
criterion_fgsm = nn.CrossEntropyLoss()
criterion_fgsm = criterion_fgsm.to(device)
Losses_fgsm = []
for epoch in tqdm(range(20)):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        # 随机选择一批样本作为对抗样本生成的基础
        data, target = data.to(device), target.to(device)
        # 将生成器的输出与原始图像结合，生成对抗样本
        optimizer_fgsm.zero_grad()
        adversarial_examples = model_fgsm(data)
        adversarial_examples = torch.sigmoid(
            adversarial_examples)  # 确保输出在[0,1]范围
        # 将生成的对抗样本传递给识别网络
        output = model_train(adversarial_examples)
        # 计算损失：我们希望生成的对抗样本被识别为目标标签
        target_labels = target.clone().detach()
        target_labels[target_labels == 9] = 3  # 将标签9替换为3
        loss = criterion_fgsm(output, target_labels)
        # 反向传播和优化
        loss.backward()
        optimizer_fgsm.step()
        running_loss += loss.item()
    Losses_fgsm.append(running_loss)

# 测试生成的对抗样本
for image, label in test_loader:
    image = image.to(device)
    model_fgsm.eval()
    with torch.no_grad():
        adversarial_example = model_fgsm(image)
        adversarial_example = torch.sigmoid(adversarial_example).to(device)
        output = model_train(adversarial_example)
        pred = output.argmax(dim=1, keepdim=True).cpu().numpy().reshape(-1)
        label = label.cpu().numpy().reshape(-1)
        diff = np.where(pred != label)
        print(len(diff[0]), '个样本分类错误')
    break

# 显示对抗样本和损失函数
fig, ax = plt.subplots(2, 8, figsize=(16, 4))
for i in range(8):
    ax[0, i].imshow(image[i].cpu().numpy().reshape(28, 28), cmap='gray')
    ax[0, i].axis('off')
    ax[0, i].set_title('原始样本 %d' % label[i])
    ax[1, i].imshow(adversarial_example[i].cpu(
    ).numpy().reshape(28, 28), cmap='gray')
    ax[1, i].axis('off')
    ax[1, i].set_title('对抗样本 %d' % pred[i])
plt.tight_layout()
plt.show()
plt.close()
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].plot(Losses_train)
ax[0].set_title('训练损失')
ax[1].plot(Losses_fgsm)
ax[1].set_title('对抗样本生成损失')
plt.show()
